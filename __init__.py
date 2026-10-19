"""SeedVR2 image-space multiplier bridge for ComfyUI.

Decode an image latent with its image VAE, optionally resize the decoded IMAGE,
then encode that image with the SeedVR2 VAE. The multiplier therefore changes
the image resolution *before* SeedVR2 encoding; it does not interpolate an
already-encoded SeedVR2 latent.
"""

import torch
import torch.nn.functional as F


class SeedVR2ImageSpaceMultiplierBridge:
    SCALE_METHODS = ["bilinear", "bicubic", "lanczos", "nearest", "nearest-exact", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "image_vae": ("VAE",),
                "seedvr2_vae": ("VAE",),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "round": 0.1,
                }),
                "scaling_method": (cls.SCALE_METHODS, {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("seedvr2_latent",)
    FUNCTION = "convert"
    CATEGORY = "latent/SeedVR2"
    DESCRIPTION = (
        "Decodes the image latent, optionally resizes the decoded image, then encodes "
        "at that target resolution with the SeedVR2 VAE. Scaling occurs in image space "
        "before SeedVR2 encoding, not by interpolating the SeedVR2 latent."
    )

    @staticmethod
    def _target_dimension(value, factor, alignment=8):
        """Scale a pixel dimension and round to the nearest VAE-friendly multiple of 8."""
        scaled = max(alignment, int(round((value * factor) / alignment)) * alignment)
        return scaled

    @staticmethod
    def _resize_image(image, target_h, target_w, method):
        # ComfyUI IMAGE layout is [B,H,W,C]. Interpolate in channel-first layout.
        original_dtype = image.dtype
        nchw = image[..., :3].permute(0, 3, 1, 2)
        # Keep interpolation stable for half/bfloat16 image tensors, then restore dtype.
        work = nchw.float() if nchw.dtype in (torch.float16, torch.bfloat16) else nchw

        if method == "lanczos":
            # Separable Lanczos-3 resize, applied to image pixels rather than latent features.
            def lanczos_weights(input_size, output_size, device):
                scale = input_size / output_size
                filter_scale = max(scale, 1.0)
                positions = (torch.arange(output_size, device=device, dtype=torch.float32) + 0.5) * scale - 0.5
                source = torch.arange(input_size, device=device, dtype=torch.float32)
                distance = (positions[:, None] - source[None, :]) / filter_scale
                weights = torch.sinc(distance) * torch.sinc(distance / 3.0)
                weights = torch.where(distance.abs() < 3.0, weights, torch.zeros_like(weights))
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                return weights

            hw = lanczos_weights(work.shape[-2], target_h, work.device)
            ww = lanczos_weights(work.shape[-1], target_w, work.device)
            resized = torch.einsum("oh,bchw->bcow", hw, work)
            resized = torch.einsum("bcow,pw->bcop", resized, ww)
        else:
            kwargs = {"size": (target_h, target_w), "mode": method}
            if method in ("bilinear", "bicubic"):
                kwargs["align_corners"] = False
            resized = F.interpolate(work, **kwargs)

        return resized.to(original_dtype).permute(0, 2, 3, 1).contiguous()

    def convert(self, samples, image_vae, seedvr2_vae, multiplier=1.0, scaling_method="bilinear"):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Input must be a ComfyUI LATENT containing a 'samples' tensor.")

        image_latent = samples["samples"]
        if getattr(image_latent, "is_nested", False):
            image_latent = image_latent.unbind()[0]
        if image_latent.ndim != 4:
            raise ValueError(
                "Expected a 4-D image latent [B,C,H,W], "
                f"but received shape {tuple(image_latent.shape)}."
            )

        factor = float(multiplier)
        if not 1.0 <= factor <= 4.0:
            raise ValueError(f"multiplier must be between 1.0 and 4.0, got {factor}.")
        if scaling_method not in self.SCALE_METHODS:
            raise ValueError(f"Unsupported scaling_method: {scaling_method}.")

        # Decode the image latent to ComfyUI IMAGE [B,H,W,C].
        image = image_vae.decode(image_latent)
        if image.ndim == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(
                "Image VAE did not return an IMAGE tensor in [B,H,W,C] layout; "
                f"received shape {tuple(image.shape)}."
            )
        image = image[..., :3]

        # The crucial fix: resize pixels BEFORE SeedVR2 VAE encoding.
        # Align target dimensions to 8 pixels, matching the usual spatial VAE stride.
        if factor != 1.0:
            old_h, old_w = image.shape[1:3]
            target_h = self._target_dimension(old_h, factor)
            target_w = self._target_dimension(old_w, factor)
            image = self._resize_image(image, target_h, target_w, scaling_method)

        seedvr2_latent = seedvr2_vae.encode(image)
        if seedvr2_latent.ndim != 5:
            raise ValueError(
                "SeedVR2 VAE encode should return a 5-D video latent [B,C,T,H,W], "
                f"but returned shape {tuple(seedvr2_latent.shape)}. Check that the "
                "selected VAE is the SeedVR2 VAE."
            )
        if seedvr2_latent.shape[1] != 16:
            raise ValueError(
                "SeedVR2 VAE encode returned an unexpected channel count: "
                f"shape {tuple(seedvr2_latent.shape)}. SeedVR2 expects 16 latent channels. "
                "Check the VAE connection and ComfyUI/SeedVR2 versions."
            )

        # Do not rescale the encoded latent: its spatial size now follows the resized image.
        return ({"samples": seedvr2_latent},)


NODE_CLASS_MAPPINGS = {"SeedVR2ImageSpaceMultiplierBridge": SeedVR2ImageSpaceMultiplierBridge}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2ImageSpaceMultiplierBridge": "SeedVR2 Image-Space Multiplier Bridge"
}

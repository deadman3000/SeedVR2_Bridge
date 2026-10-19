# SeedVR2 Image-Space Multiplier Bridge

This ComfyUI custom node decodes an image latent with its associated image VAE, applies the multiplier in pixel space, then encodes the result with the SeedVR2 VAE.

## Inputs
- `samples`: image `LATENT` from a compatible sampler
- `image_vae`: VAE associated with the input image latent
- `seedvr2_vae`: SeedVR2 VAE
- `multiplier`: image-space scale factor, default `1.0`, range `1.0`–`4.0`, step `0.1`
- `scaling_method`: `bilinear` (default), `bicubic`, `lanczos`, `nearest`, `nearest-exact`, or `area`

## What changed
1. Decode the input image latent using `image_vae`.
2. If `multiplier` is above `1.0`, resize the decoded image pixels internally.
3. Encode that resized image using the SeedVR2 VAE.
4. Return the SeedVR2 latent without any post-encode latent interpolation.

This means a 1440p image at `1.5×` is resized to approximately 2160p **before** SeedVR2 encoding. The target width and height are rounded to the nearest multiple of 8 pixels for VAE compatibility. For example, 2560×1440 becomes 3840×2160.

## Workflow notes
- Keep the SeedVR2 conditioning node connected to the bridge's `seedvr2_latent` output as in the existing working workflow.
- Set `multiplier=1.0` to preserve the original no-resize bridge path.
- The interpolation mode now applies to decoded image pixels, not latent feature maps.
- This archive has been syntax-checked and tested with mock VAEs for expected dimensions, but it has not been executed inside the user's ComfyUI installation. Back up the existing node folder before replacing it.

## Install
Stop ComfyUI, back up and remove the previous bridge folder so ComfyUI does not load two copies of the node. Extract this archive into `ComfyUI/custom_nodes/`, creating `seedvr2_image_space_multiplier_bridge`, then restart ComfyUI.

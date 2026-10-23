# SeedVR2 Image-Space Multiplier Bridge

A ComfyUI node that decodes a source image latent with `image_vae`, resizes the decoded pixels in image space, and re-encodes them with `seedvr2_vae` for the SeedVR2 refinement path.

## Inputs

- `samples`: source `LATENT` (4-D image latent or 5-D single-frame latent with T=1)
- `image_vae`: VAE used to decode the source latent
- `seedvr2_vae`: SeedVR2 VAE used to encode the resized image
- `multiplier`: image-space scale multiplier (default 1.0)
- `scaling_method`: resize method (default Lanczos)
- `short_side`: optional base short-side resolution in pixels (default 0, which uses the decoded image's current short side)

The original widgets remain in their original order (`multiplier`, then `scaling_method`). The new `short_side` widget is appended after them to preserve saved workflow widget values when updating the node.

Output `seedvr2_latent` connects to the SeedVR2 sampler/refinement path. Connect `output_resolution` to a Show Text node if you want to display the calculated output dimensions.

Output dimensions are calculated from the decoded image aspect ratio, scaled by `short_side × multiplier` (or by the decoded image's own short side when `short_side=0`), then rounded to the nearest multiple of 16.

Example: source image 3392×1440, `short_side=1440`, `multiplier=1.2` → `4064 × 1728`.


New: added a manual short side field that is automatically multiplied. 0 sets to default behaviour based upon input latent. 
Added report output to view the final resolution.
 
### Existing workflow compatibility (v12)

This version includes a small ComfyUI frontend migration that repairs an invalid/missing `short_side` value to `0` when an older saved workflow is loaded. It also repairs an invalid `multiplier` value to `1.0`. Valid saved values are left unchanged. The original widget order remains `multiplier`, `scaling_method`, then `short_side`.

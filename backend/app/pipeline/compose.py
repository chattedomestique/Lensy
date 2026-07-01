"""Stage 7 — Compose. Premultiplied `F·α` **over** the blurred background (§7.2.7), linear light.

Two modes:
- default (`subject_dof=False`): the subject is composited **sharp** — clean, crisp edges, the
  classic in-focus subject over a depth-graded background. This is the reliable, artifact-free look.
- `subject_dof=True`: the subject is run through the SAME depth-of-field as the background, so
  parts of it off the focal plane soften too. More cinematic, but only as clean as the depth map.

Premultiplied compositing means the decontaminated soft edges blend without a fringe."""

from __future__ import annotations

import numpy as np

from .blur import BlurParams, blur_foreground_dof
from .color import linear_to_srgb, srgb_to_linear


def compose(
    fg_srgb: np.ndarray,        # F, foreground color float32 [0,1] sRGB (from decontaminate)
    alpha: np.ndarray,          # soft matte float32 [0,1]
    blurred_bg_u8: np.ndarray,  # depth-blurred background plate, sRGB uint8
    disparity: np.ndarray,
    p: BlurParams,
) -> np.ndarray:
    """Return the final composited image, uint8 RGB."""
    bg_lin = srgb_to_linear(blurred_bg_u8.astype(np.float32) / 255.0)

    if p.subject_dof:
        fg_premult, fg_alpha = blur_foreground_dof(fg_srgb, alpha, disparity, p)
    else:
        fg_lin = srgb_to_linear(np.clip(fg_srgb, 0.0, 1.0))
        fg_alpha = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
        fg_premult = fg_lin * fg_alpha

    out_lin = fg_premult + bg_lin * (1.0 - fg_alpha)  # premultiplied OVER
    return (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

"""Stage 7 — Compose. Premultiplied `F·α` **over** the blurred background (§7.2.7), all in
linear light.

For a real depth-of-field (not a sharp sticker on a blurred plate), the subject is run through
the **same** depth-driven blur as the background (`blur_foreground_dof`): parts of the subject
off the focal plane soften, the focal parts stay razor sharp. Both layers were produced by one
continuous CoC-from-depth field, so the whole scene grades with distance. Premultiplied
compositing means the (decontaminated) soft edges blend without a fringe."""

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

    # subject with its own depth-of-field → premultiplied linear color + blurred coverage
    fg_premult, fg_alpha = blur_foreground_dof(fg_srgb, alpha, disparity, p)

    # premultiplied OVER the (already depth-blurred) background
    out_lin = fg_premult + bg_lin * (1.0 - fg_alpha)
    return (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

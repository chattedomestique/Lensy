"""Stage 7 — Compose. Premultiplied `F·α` **over** the blurred background (§7.2.7), linear light.

The subject is subject to the SAME lens as everything else: it blurs by its own per-pixel circle
of confusion (`radius_fg`). On the focal plane that CoC is ~0 and the scatter passes it through
sharp; off the plane it defocuses like any object at its distance — no "always in focus" override.

Premultiplied compositing means the decontaminated soft edges blend without a fringe.
(Legacy simple path — the layered renderer in layered.py is what the pipeline uses.)"""

from __future__ import annotations

import numpy as np

from .blur import BlurParams, blur_foreground_dof
from .color import linear_to_srgb, srgb_to_linear


def compose(
    fg_srgb: np.ndarray,        # F, foreground color float32 [0,1] sRGB (from decontaminate)
    alpha: np.ndarray,          # soft matte float32 [0,1]
    blurred_bg_u8: np.ndarray,  # depth-blurred background plate, sRGB uint8
    radius_fg: np.ndarray,      # subject's per-pixel CoC radius (px)
    p: BlurParams,
) -> np.ndarray:
    """Return the final composited image, uint8 RGB."""
    bg_lin = srgb_to_linear(blurred_bg_u8.astype(np.float32) / 255.0)
    fg_premult, fg_alpha = blur_foreground_dof(fg_srgb, alpha, radius_fg, p)
    out_lin = fg_premult + bg_lin * (1.0 - fg_alpha)  # premultiplied OVER
    return (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

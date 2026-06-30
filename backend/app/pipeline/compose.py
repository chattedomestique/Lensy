"""Stage 7 — Compose. Premultiplied `F·α` **over** the blurred background (§7.2.7), all in
linear light. A *mild* blur is applied to subject regions that sit off the focal plane so the
cutout doesn't read as a flat sticker pasted on a blurred plate.

Compositing is premultiplied so soft hair edges blend without a dark or bright fringe — the
payoff of the decontamination step that produced a clean F."""

from __future__ import annotations

import cv2
import numpy as np

from .blur import BlurParams
from .color import linear_to_srgb, srgb_to_linear


def compose(
    fg_srgb: np.ndarray,       # F, foreground color float32 [0,1] sRGB (from decontaminate)
    alpha: np.ndarray,         # soft matte float32 [0,1]
    blurred_bg_u8: np.ndarray,  # blurred background plate, sRGB uint8
    disparity: np.ndarray,
    p: BlurParams,
) -> np.ndarray:
    """Return the final composited image, uint8 RGB."""
    h, w = alpha.shape[:2]
    a = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)

    fg_lin = srgb_to_linear(np.clip(fg_srgb, 0.0, 1.0))
    bg_lin = srgb_to_linear(blurred_bg_u8.astype(np.float32) / 255.0)

    # --- anti-sticker: mild defocus on subject regions away from the focal plane ---
    subj_coc = np.abs(disparity.astype(np.float32) - float(p.disp_focus))
    norm = max(float(p.disp_focus), 1.0 - float(p.disp_focus), 1e-3)
    subj_blur_amt = float(np.clip(subj_coc.mean() / norm, 0.0, 1.0))
    if p.k > 0 and subj_blur_amt > 0.04:
        sigma = 0.6 + 2.5 * subj_blur_amt * (p.k / 100.0)  # gentle, never the full lens blur
        fg_soft = cv2.GaussianBlur(fg_lin, (0, 0), sigmaX=sigma)
        # blend toward softened FG proportionally to how off-plane the subject sits
        mix = np.clip(subj_coc / norm, 0.0, 1.0)[..., None] * 0.6
        fg_lin = fg_lin * (1.0 - mix) + fg_soft * mix

    # --- premultiplied OVER ---
    out_lin = fg_lin * a + bg_lin * (1.0 - a)
    return (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

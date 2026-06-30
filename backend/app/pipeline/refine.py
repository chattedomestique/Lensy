"""Stage 2 + 3 — Refine the matte to true edges, then decontaminate foreground color.

- **Refine** (§7.2.2): `cv2.ximgproc.guidedFilter(guide=RGB, src=α)` snaps α to image edges
  (matting-Laplacian link) without halos. Bilateral fallback if ximgproc is absent.
- **Decontaminate** (§7.2.3): pymatting `estimate_foreground_ml(image, α)` recovers the true
  foreground color F, stripping background tint from edge pixels. **This is the single most
  important anti-halo step.** Fallback is a passthrough (F = image), which is honest but weaker."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .runtime import ModelBundle

log = logging.getLogger("lensy.refine")


def refine_alpha(rgb_u8: np.ndarray, alpha: np.ndarray, radius: int = 8, eps: float = 1e-4) -> np.ndarray:
    """Edge-snap the soft alpha to the guide image. Returns float32 [0,1]."""
    guide = rgb_u8.astype(np.float32) / 255.0
    a = alpha.astype(np.float32)
    try:
        gf = cv2.ximgproc.guidedFilter  # contrib build
        out = gf(guide=guide, src=a, radius=radius, eps=eps)
    except AttributeError:
        log.info("ximgproc unavailable; refine via joint-bilateral fallback")
        # joint bilateral approximates the edge-aware behaviour
        out = cv2.bilateralFilter(a, d=radius * 2 + 1, sigmaColor=0.1, sigmaSpace=radius)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def decontaminate(rgb_u8: np.ndarray, alpha: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Estimate true foreground color F (float32 [0,1], HxWx3). Anti-halo cornerstone."""
    img = rgb_u8.astype(np.float32) / 255.0
    if bundle.has_pymatting:
        try:
            from pymatting import estimate_foreground_ml

            fg = estimate_foreground_ml(img, np.clip(alpha, 0.0, 1.0))
            return np.clip(fg, 0.0, 1.0).astype(np.float32)
        except Exception as e:
            log.warning("estimate_foreground_ml failed (%s); F = image passthrough", e.__class__.__name__)
    # honest fallback: use the observed image as F. Edges keep some background tint, but we
    # never invent color. Downstream feather + inpaint still suppress most haloing.
    return img.astype(np.float32)

"""Stage 2 + 3 — Refine the matte to true edges, then decontaminate foreground color.

- **Refine** (§7.2.2): `cv2.ximgproc.guidedFilter(guide=RGB, src=α)` snaps α to image edges
  (matting-Laplacian link) without halos. Bilateral fallback if ximgproc is absent.
- **Decontaminate** (§7.2.3): pymatting `estimate_foreground_ml(image, α)` recovers the true
  foreground color F, stripping background tint from edge pixels. **This is the single most
  important anti-halo step.** Fallback is a passthrough (F = image), which is honest but weaker."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .runtime import ModelBundle

log = logging.getLogger("lensy.refine")

# BiRefNet mattes can carry a very wide, low-slope transition band (5%+ of the frame). Composited
# over a blurred background that wide band lets the bright background glow through as a HALO. We
# steepen the transition with a smoothstep so the edge stays soft (a few px, keeps hair) without
# the broad feather. Tunable via LENSY_MATTE_TIGHTEN=lo,hi (or "off").
_TIGHTEN = os.environ.get("LENSY_MATTE_TIGHTEN", "0.35,0.65")


def _tighten(a: np.ndarray) -> np.ndarray:
    if _TIGHTEN.lower() in ("off", "0", ""):
        return a
    try:
        lo, hi = (float(x) for x in _TIGHTEN.split(","))
    except ValueError:
        lo, hi = 0.35, 0.65
    t = np.clip((a - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep


def refine_alpha(rgb_u8: np.ndarray, alpha: np.ndarray, radius: int = 8, eps: float = 1e-4) -> np.ndarray:
    """Edge-snap the soft alpha to the guide image, then steepen the transition to kill halos.
    Returns float32 [0,1]."""
    guide = rgb_u8.astype(np.float32) / 255.0
    a = alpha.astype(np.float32)
    try:
        gf = cv2.ximgproc.guidedFilter  # contrib build
        out = gf(guide=guide, src=a, radius=radius, eps=eps)
    except AttributeError:
        log.info("ximgproc unavailable; refine via joint-bilateral fallback")
        # joint bilateral approximates the edge-aware behaviour
        out = cv2.bilateralFilter(a, d=radius * 2 + 1, sigmaColor=0.1, sigmaSpace=radius)
    out = _tighten(np.clip(out, 0.0, 1.0).astype(np.float32))
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

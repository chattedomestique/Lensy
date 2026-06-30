"""Stage 4 — Depth. Apple Depth Pro → disparity, chosen for best boundary accuracy + thin
structure (hair) recall (§7.2.4). Returns **disparity** normalized to [0,1] where larger =
nearer (so it sits naturally in `CoC = K·(disparity − disp_focus)`).

Fallback when Depth Pro isn't loaded: a smooth synthetic disparity from a vertical gradient
blended with luminance — crude, but gives a believable "background falls away" focal field
so the blur stage still has something depth-like to grade against."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .runtime import ModelBundle

log = logging.getLogger("lensy.depth")


def estimate_disparity(rgb_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Return disparity float32 [0,1], shape (H, W). 1 = nearest, 0 = farthest."""
    if bundle.depth_model is not None:
        try:
            return _depth_pro_disparity(rgb_u8, bundle)
        except Exception as e:
            log.warning("Depth Pro failed (%s); using radial fallback", e.__class__.__name__)
    return _radial_disparity(rgb_u8)


def _normalize(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    lo, hi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def _depth_pro_disparity(rgb_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    import torch

    h, w = rgb_u8.shape[:2]
    image = bundle.depth_transform(rgb_u8)
    with torch.no_grad():
        out = bundle.depth_model.infer(image)
    depth = out["depth"].detach().cpu().numpy().astype(np.float32)  # metric depth, meters
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    disparity = 1.0 / np.clip(depth, 1e-3, None)  # near => large
    return _normalize(disparity)


def _radial_disparity(rgb_u8: np.ndarray) -> np.ndarray:
    """No depth model: assume the subject is centered and the background recedes outward and
    upward. Blend a center-weighted radial falloff with inverted luminance (darker often =
    deeper) for a touch of image awareness."""
    h, w = rgb_u8.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h * 0.55
    r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
    radial = np.clip(1.0 - r, 0.0, 1.0)  # near center => near
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lum = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(h, w) / 64.0)
    disparity = 0.7 * radial + 0.3 * lum
    return _normalize(disparity)

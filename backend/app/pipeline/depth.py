"""Stage 4 — Depth. Depth Anything V2 → disparity, normalized to [0,1] where larger = nearer
(so it sits naturally in `CoC = K·(disparity − disp_focus)`). Depth Anything's `predicted_depth`
is already disparity-like (near = large), so it is used directly — no inversion.

(The brief's first pick was Apple Depth Pro for boundary/hair-thin recall, but it was 60-130s
and leaked MPS memory on a 16GB M4; depth here only grades the blur falloff, not the edge, so
Depth Anything V2 is the practical choice — see runtime._load_depth.)

Fallback when no depth model is loaded: a smooth synthetic disparity from a center-weighted
radial falloff blended with luminance — crude, but gives a believable "background falls away"
focal field so the blur stage still has something depth-like to grade against."""

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
            return _model_disparity(rgb_u8, bundle)
        except Exception as e:
            log.warning("depth model failed (%s); using radial fallback", e.__class__.__name__)
    return _radial_disparity(rgb_u8)


def _sigma(disp: np.ndarray) -> float:
    h, w = disp.shape[:2]
    return max(2.0, min(h, w) / 130.0)  # ~12px at 1536


def smooth_depth(disp: np.ndarray) -> np.ndarray:
    """Plain Gaussian smoothing so the DoF grades cleanly rather than blotching from per-pixel
    model noise. NOT edge-aware on purpose — locking depth to image texture (tattoos, patterns)
    would create false depth steps → sharp/blur seams. Range-preserving (metres or [0,1])."""
    return cv2.GaussianBlur(disp.astype(np.float32), (0, 0), sigmaX=_sigma(disp))


def background_depth(disp: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    """Depth field for the BACKGROUND: fill the subject's region with surrounding background
    depth (so the subject's near-depth can't bleed outward and leave a sharp ring around the
    silhouette), then smooth. Keeps the true depth discontinuity AT the silhouette. Works in the
    depth's own units (metres for Depth Pro, [0,1] otherwise)."""
    h, w = disp.shape[:2]
    grow = max(3, (min(h, w) // 120) | 1)
    hole = cv2.dilate((subject_mask > 0).astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    lo, rng = float(disp.min()), max(float(disp.max() - disp.min()), 1e-6)
    d8 = np.clip((disp - lo) / rng * 255.0, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(d8, hole, max(3, grow), cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    filled = filled * rng + lo  # back to original units
    return cv2.GaussianBlur(filled, (0, 0), sigmaX=_sigma(disp))


def _normalize(d: np.ndarray) -> np.ndarray:
    """Normalize a depth/disparity signal to [0,1]. A plain min/max stretch collapses everything
    into a sliver when one object is much nearer than the rest (e.g. a pole right by the lens) —
    then subjects and background people end up numerically ~equal and the DoF can't separate the
    focal planes. So we blend the linear stretch with a **rank (histogram) equalization**, which
    spreads the populated depth range perceptually and keeps distinct planes distinct. Kept partly
    linear so smooth regions don't over-amplify into spurious gradients."""
    d = d.astype(np.float32)
    lo, hi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    lin = np.clip((d - lo) / (hi - lo), 0.0, 1.0)

    flat = lin.ravel()
    order = np.argsort(flat)
    rank = np.empty(flat.size, np.float32)
    rank[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    rank = rank.reshape(d.shape)
    return (0.7 * rank + 0.3 * lin).astype(np.float32)


def normalize01(d: np.ndarray) -> np.ndarray:
    """Public: percentile-normalize any depth/disparity signal to [0,1]."""
    return _normalize(d)


def _da3_disparity(rgb_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Depth Anything V3 mono path: inference([rgb]) → Prediction.depth (metric, near = small).
    Returned as metric depth; run_pipeline inverts it to disparity (depth_metric=True)."""
    import os

    import torch

    h, w = rgb_u8.shape[:2]
    res = int(os.environ.get("LENSY_DA3_RES", "768"))
    with torch.no_grad():
        pred = bundle.depth_model.inference([rgb_u8], process_res=res)
    d = np.asarray(pred.depth).squeeze().astype(np.float32)
    if bundle.device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 0.05, None).astype(np.float32)


def _model_disparity(rgb_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    if getattr(bundle, "depth_backend", "hf") == "da3":
        return _da3_disparity(rgb_u8, bundle)

    import torch
    from PIL import Image

    h, w = rgb_u8.shape[:2]
    processor = bundle.depth_transform
    inputs = processor(images=Image.fromarray(rgb_u8), return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        out = bundle.depth_model(**inputs)
    post = processor.post_process_depth_estimation(out, target_sizes=[(h, w)])
    pred = post[0]["predicted_depth"].detach().cpu().float().numpy().astype(np.float32)
    if bundle.device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    if bundle.depth_metric:
        # Depth Pro → metric depth in METERS. Keep it metric (do NOT normalize) so the blur can
        # use real distances for a true optical falloff. Clip only wild outliers.
        return np.clip(pred, 0.1, 100.0).astype(np.float32)
    # Depth Anything → relative disparity-like (near = large); normalize to [0,1] as a proxy.
    return _normalize(pred)


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

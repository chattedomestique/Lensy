"""Stage 1 — Matte. BiRefNet (HR) → soft alpha. Falls back to GrabCut + feather when the
model isn't loaded. **Always a soft alpha in [0,1], never a hard mask** (§7.2.1)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .runtime import ModelBundle

log = logging.getLogger("lensy.matte")

_BIREFNET_SIZE = 1024  # BiRefNet trained square input


def estimate_alpha(rgb_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Return soft alpha float32 [0,1], shape (H, W), at the input resolution."""
    if bundle.matte_model is not None:
        try:
            return _birefnet_alpha(rgb_u8, bundle)
        except Exception as e:  # never let a model hiccup kill the render
            log.warning("BiRefNet failed (%s); using GrabCut fallback", e.__class__.__name__)
    return _grabcut_alpha(rgb_u8)


def _birefnet_alpha(rgb_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    h, w = rgb_u8.shape[:2]
    x = cv2.resize(rgb_u8, (_BIREFNET_SIZE, _BIREFNET_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = ((t - mean) / std).to(bundle.device)

    with torch.no_grad():
        out = bundle.matte_model(t)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        pred = pred.sigmoid().cpu()
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    alpha = pred[0, 0].numpy().astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def _grabcut_alpha(rgb_u8: np.ndarray) -> np.ndarray:
    """Classic fallback: GrabCut seeded from a centered rect, then feather the hard mask into
    a soft alpha so downstream refine/decontam still have a gradient to work with."""
    h, w = rgb_u8.shape[:2]
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        hard = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except Exception:
        # last-ditch: luminance-thresholded center blob
        hard = np.zeros((h, w), np.float32)
        hard[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = 1.0
    # feather: distance-based soft edge so it is not a hard mask
    k = max(3, (min(h, w) // 200) * 2 + 1)
    soft = cv2.GaussianBlur(hard, (k, k), 0)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)

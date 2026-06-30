"""Stage 5 — Inpaint. Remove the subject and fill its hole so the background blur near the
silhouette only ever averages *real* background (§7.2.5). LaMa (big-lama) is the primary;
`cv2.inpaint` (Telea) is the cheap, always-available fallback.

Critical: we dilate the subject mask before filling so no sliver of contaminated edge pixel
survives at the silhouette to be smeared back into the blur."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .runtime import ModelBundle

log = logging.getLogger("lensy.inpaint")


def fill_background(rgb_u8: np.ndarray, alpha: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Return a clean background plate (uint8 RGB, HxWx3) with the subject removed + filled."""
    h, w = rgb_u8.shape[:2]
    # subject region = where alpha is meaningfully present, dilated to swallow edge mix pixels
    grow = max(3, (min(h, w) // 150) | 1)
    hole = (alpha > 0.05).astype(np.uint8) * 255
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))

    if bundle.inpaint_model is not None:
        try:
            return _lama_fill(rgb_u8, hole, bundle)
        except Exception as e:
            log.warning("LaMa failed (%s); using cv2.inpaint fallback", e.__class__.__name__)
    return _cv2_fill(rgb_u8, hole)


def _lama_fill(rgb_u8: np.ndarray, hole_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(rgb_u8)
    mask = Image.fromarray(hole_u8)
    out = bundle.inpaint_model(img, mask)  # simple_lama returns a PIL image
    arr = np.array(out.convert("RGB"))
    if arr.shape[:2] != rgb_u8.shape[:2]:
        arr = cv2.resize(arr, (rgb_u8.shape[1], rgb_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _cv2_fill(rgb_u8: np.ndarray, hole_u8: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    radius = max(3, hole_u8.shape[0] // 200)
    filled = cv2.inpaint(bgr, hole_u8, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

"""Stage 5 — Inpaint. Remove the subject and fill its hole so the background blur near the
silhouette only ever averages *real* background (§7.2.5). LaMa (big-lama) is the primary;
`cv2.inpaint` (Telea) is the cheap, always-available fallback.

Critical: we dilate the subject mask before filling so no sliver of contaminated edge pixel
survives at the silhouette to be smeared back into the blur."""

from __future__ import annotations

import logging
import os

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


# LaMa runs on CPU and scales ~quadratically with resolution. The inpainted plate is only
# ever seen *blurred* and *behind* the sharp subject, so a reduced-res fill is invisible in
# the final image — we cap LaMa's working long-edge and upscale the result. Big speedup.
_LAMA_MAX_EDGE = int(os.environ.get("LENSY_LAMA_MAX_EDGE", "768"))
# The user-facing "erase an object" fill is seen SHARP (not blurred behind the subject), so it
# runs at a higher resolution than the background plate — quality matters more than speed here.
_ERASE_MAX_EDGE = int(os.environ.get("LENSY_ERASE_MAX_EDGE", "1536"))


def erase_region(rgb_u8: np.ndarray, mask_u8: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """Remove whatever the mask covers (white = erase) and fill it plausibly. Used by the
    interactive object-removal tool, so the fill is kept high-res. Returns uint8 RGB."""
    h, w = rgb_u8.shape[:2]
    grow = max(3, (min(h, w) // 200) | 1)  # small grow so no rim of the object survives
    hole = cv2.dilate(
        (mask_u8 > 127).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)),
    )
    if int((hole > 0).sum()) == 0:
        return rgb_u8
    if bundle.inpaint_model is not None:
        try:
            return _lama_fill(rgb_u8, hole, bundle, max_edge=_ERASE_MAX_EDGE)
        except Exception as e:
            log.warning("LaMa erase failed (%s); using cv2.inpaint fallback", e.__class__.__name__)
    return _cv2_fill(rgb_u8, hole)


def _lama_fill(
    rgb_u8: np.ndarray, hole_u8: np.ndarray, bundle: ModelBundle, max_edge: int = _LAMA_MAX_EDGE
) -> np.ndarray:
    from PIL import Image

    h, w = rgb_u8.shape[:2]
    scale = min(1.0, max_edge / float(max(h, w)))
    if scale < 1.0:
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize(rgb_u8, (sw, sh), interpolation=cv2.INTER_AREA)
        # dilate-then-resize the mask so the downsampled hole never under-covers the subject
        mask_small = cv2.resize(hole_u8, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        small, mask_small = rgb_u8, hole_u8

    out = bundle.inpaint_model(Image.fromarray(small), Image.fromarray(mask_small))
    arr = np.array(out.convert("RGB"))
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    # keep the original (sharp) pixels outside the hole; only the filled hole comes from LaMa
    keep = (hole_u8 == 0)
    arr[keep] = rgb_u8[keep]
    return arr


def _cv2_fill(rgb_u8: np.ndarray, hole_u8: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    radius = max(3, hole_u8.shape[0] // 200)
    filled = cv2.inpaint(bgr, hole_u8, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

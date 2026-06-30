"""Color-space helpers. The pipeline works in **linear light**; we convert sRGB↔linear
only at the boundaries (§6, §7.3). All arrays are float32 in [0, 1] unless noted."""

from __future__ import annotations

import numpy as np

# --- sRGB <-> linear (IEC 61966-2-1) -------------------------------------------------


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded [0,1] floats to linear light."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode linear-light [0,1] floats back to sRGB."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)  # linear-HDR may exceed 1.0 from highlight scatter; tone below
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


# --- byte <-> float ------------------------------------------------------------------


def to_float(img_u8: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]."""
    return (img_u8.astype(np.float32)) / 255.0


def to_u8(img_f: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8 [0,255], clamped."""
    return (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def reinhard_tonemap(linear_hdr: np.ndarray, white: float = 3.0) -> np.ndarray:
    """Extended Reinhard: roll HDR linear values that overshot 1.0 (from highlight scatter)
    back into range while keeping bloom bright. `white` is the luminance that maps to 1.0;
    values below ~1 are left almost untouched, so only blown-out highlights compress.
    Applied just before re-encoding to sRGB."""
    x = np.clip(linear_hdr, 0.0, None).astype(np.float32)
    w2 = float(white) * float(white)
    return np.clip(x * (1.0 + x / w2) / (1.0 + x), 0.0, 1.0).astype(np.float32)

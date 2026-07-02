"""Modeled film grain — a NumPy port of the GLSL noise pass (grainimplementation.md).

Key properties preserved from the spec:
  • PIXEL-scale cells (1–2.5 px), not UV-scaled blobs — floor() cells, no interpolation, so the
    grain keeps its sharp, crystalline character at any resolution.
  • Luminance response: parabola peaking near lum≈0.4 (pow 0.65) — grain lives in the lower
    midtones and vanishes in deep shadows and bright highlights (halide threshold/saturation).
  • Two scales: fine (per-grain) + coarse (2×, ISO clustering), blended by amount² · 0.45.
  • Per-channel independent patterns (three dye layers); blue boosted ×1.3 (Fuji's blue-sensitive
    layer is the noisiest) → subtle cyan-yellow chromatic noise.
  • Quadratic amplitude ramp: amount² · 0.20 in midtones — imperceptible low, strong high.
  • Static per image: the seed comes from the analysis (not the render call), so slider edits
    re-render with identical grain — film doesn't flicker.
"""

from __future__ import annotations

import numpy as np


def _hash(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Port of the GLSL hash: p=fract(p·(127.1,311.7)); p+=dot(p,p.yx+19.19); fract(p.x·p.y)."""
    px = np.modf(x * 127.1)[0]
    py = np.modf(y * 311.7)[0]
    d = px * (py + 19.19) + py * (px + 19.19)
    px = px + d
    py = py + d
    return np.modf(px * py)[0].astype(np.float32)


def _cell_noise(px: np.ndarray, py: np.ndarray, cell: float, sx: float, sy: float) -> np.ndarray:
    """Signed [-1,1] noise constant within floor(px/cell) cells, offset by a seed pair."""
    cx = np.floor(px / cell) + sx
    cy = np.floor(py / cell) + sy
    return _hash(cx, cy) * 2.0 - 1.0


def apply_grain(img_u8: np.ndarray, amount: float, size: float, seed: float) -> np.ndarray:
    """Apply modeled film grain to the final sRGB uint8 frame. amount/size in [0,1]; seed in [0,1)."""
    if amount <= 0.001:
        return img_u8
    h, w = img_u8.shape[:2]
    col = img_u8.astype(np.float32) / 255.0

    # luminance response — peak in the lower midtones
    lum = col @ np.array([0.299, 0.587, 0.114], np.float32)
    lum_curve = np.power(np.clip(4.0 * lum * (1.0 - lum), 0.0, None), 0.65)

    # cell sizes in real output pixels; `size` is the stock's grain character (independent of amount)
    fine = 1.0 + 2.0 * float(size)   # 1.0 (very fine) → 3.0 (coarse)
    coarse = fine * 2.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # per-channel independent seed pairs (three dye layers), from the per-image seed
    t = float(seed)
    seeds = {
        "r": (np.floor(t * 13.7 + 1.3), np.floor(t * 29.1 + 5.7)),
        "g": (np.floor(t * 43.9 + 7.1), np.floor(t * 11.3 + 2.9)),
        "b": (np.floor(t * 67.3 + 3.9), np.floor(t * 53.7 + 8.1)),
    }

    coarse_mix = amount * amount * 0.45
    grain = np.empty((h, w, 3), np.float32)
    for i, ch in enumerate(("r", "g", "b")):
        sx, sy = seeds[ch]
        f = _cell_noise(xx, yy, fine, sx, sy)
        c = _cell_noise(xx, yy, coarse, sx + 100.0, sy + 100.0)
        grain[..., i] = f * (1.0 - coarse_mix) + c * coarse_mix
    grain[..., 2] *= 1.3  # blue-sensitive layer grains hardest

    amplitude = (amount * amount * 0.20) * lum_curve
    out = col + grain * amplitude[..., None]
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

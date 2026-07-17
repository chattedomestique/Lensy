"""Modeled film grain — a physically-motivated approximation of analog colour-negative grain,
tuned toward Fujifilm's fine, tight, faintly-cool character.

Why this reads as film and not tiled digital noise:
  • Grain is a BAND-LIMITED Gaussian field, not per-cell value noise. Real silver-halide grain is a
    dense random cloud whose autocorrelation is ~Gaussian with a correlation length set by the grain
    size. We build it by band-pass filtering white noise to that scale (a difference of Gaussians)
    and renormalizing the variance — organic, tile-free, no grid, no repeats. This is the fast
    Gaussian approximation of Newson et al., "Realistic Film Grain Rendering" (IPOL, 2017), whose
    exact model is a Boolean/Poisson process of tiny random grains (too slow for interactive use).
  • Two octaves (fine + coarse) model the spread of grain sizes / high-ISO clumping.
  • Tonal response: the Boolean model's grain-coverage variance ∝ u·(1−u), so grain peaks in the mid
    densities and vanishes in deep shadow and blown highlight (halide threshold + saturation).
  • Colour: colour film exposes three dye layers that grain largely TOGETHER (a shared luminance
    grain) with a little independent per-layer speckle; the blue-sensitive (yellow) layer is the
    noisiest. The result is mostly monochrome grain with a subtle, slightly-cool chroma — the Fuji
    look — not rainbow confetti.
  • Static per image: the seed comes from the analysis id, so slider edits re-render identical grain
    (film doesn't flicker frame to frame).
  • Applied at the FINAL export resolution (see layered._output_optics) so the grain is crisp at
    native size rather than an upscale of a working-res pattern.
"""

from __future__ import annotations

import cv2
import numpy as np

# relative chroma grain per dye layer (R, G, B). Kept small so the grain is mostly monochrome (the
# three layers grain largely together); the blue-sensitive / yellow layer grains hardest and green
# edges out red, giving Fuji's faintly-cool speckle rather than neutral grey or rainbow confetti.
_CHROMA_GAIN = np.array([0.09, 0.13, 0.20], np.float32)


def _grain_layer(noise: np.ndarray, sigma: float) -> np.ndarray:
    """One octave of grain: band-pass white noise to a correlation length ~sigma via a difference
    of Gaussians (removes both the per-pixel fizz and the slow large-scale drift → organic clumps),
    then renormalize to unit variance. Cheap: both blurs use small sigmas."""
    lo = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(sigma, 0.35))
    hi = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(sigma * 3.0, 1.05))
    g = lo - hi
    return g / (float(g.std()) + 1e-6)


def apply_grain(
    img_u8: np.ndarray,
    amount: float,
    size: float,
    seed: float,
    blend: float = 0.0,
    defocus: np.ndarray | None = None,
) -> np.ndarray:
    """Apply modeled film grain to the final sRGB uint8 frame.

    amount/size/blend in [0,1]; seed in [0,1). `blend` optionally confines the grain to the
    out-of-focus regions: 0 = grain everywhere, 1 = grain only where blurred, between = a lerp.
    `defocus` (float [0,1], 0 = sharp/subject → 1 = fully blurred) is the mask that drives `blend`;
    it is resized to the frame if needed."""
    if amount <= 0.001:
        return img_u8
    h, w = img_u8.shape[:2]
    col = img_u8.astype(np.float32) / 255.0
    rng = np.random.default_rng(int(abs(float(seed)) * 1e6) % (2**32))

    # grain correlation length in output px — Fuji is fine/tight: ~0.7 px (very fine) → ~3 px (coarse)
    r_fine = 0.7 + 2.3 * float(np.clip(size, 0.0, 1.0))
    r_coarse = r_fine * 2.3
    coarse_mix = 0.4 * float(np.clip(amount, 0.0, 1.0))  # ISO clumping grows with amount

    def octaves() -> np.ndarray:
        fine = _grain_layer(rng.standard_normal((h, w), dtype=np.float32), r_fine)
        coarse = _grain_layer(rng.standard_normal((h, w), dtype=np.float32), r_coarse)
        return fine * (1.0 - coarse_mix) + coarse * coarse_mix

    # shared luminance grain (the dominant, mostly-monochrome component) + subtle per-layer speckle
    lum = octaves()
    grain = np.empty((h, w, 3), np.float32)
    for i in range(3):
        grain[..., i] = lum + _CHROMA_GAIN[i] * octaves()
    # renormalize each channel to unit std so `amp` below is a predictable RMS in [0,1]
    grain -= grain.mean(axis=(0, 1), keepdims=True)
    grain /= grain.std(axis=(0, 1), keepdims=True) + 1e-6

    # tonal response: Boolean-model coverage variance ∝ u·(1−u) → midtone-weighted, ~0 at extremes
    lum_img = col @ np.array([0.299, 0.587, 0.114], np.float32)
    resp = np.sqrt(np.clip(4.0 * lum_img * (1.0 - lum_img), 0.0, 1.0))

    # RMS grain amplitude in [0,1] — perceptible when low, strong-but-not-blown at amount=1
    rms = 0.015 + 0.10 * float(amount) * float(amount)
    amp = rms * resp[..., None]

    # blend: fade the grain toward the defocused regions only. 1 − blend·(1 − defocus): at blend=0
    # the factor is 1 everywhere; at blend=1 it equals the defocus mask (sharp/subject → no grain).
    b = float(np.clip(blend, 0.0, 1.0))
    if b > 0.0 and defocus is not None:
        d = defocus.astype(np.float32)
        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        amp = amp * (1.0 - b * (1.0 - np.clip(d, 0.0, 1.0)))[..., None]

    out = col + grain * amp
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

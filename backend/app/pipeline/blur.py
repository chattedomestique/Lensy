"""Stage 6 — Lens blur. A depth-graded **scatter** renderer in **linear light** (§7.3).
MPS-safe (pure NumPy/OpenCV, no CUDA kernels). This is the primary renderer.

Why these choices read as a *lens* and not a Gaussian smudge:
  1. Linear light + HDR headroom  → highlights stay bright as they spread (bokeh balls exist).
  2. Scatter, not gather          → a bright point spreads its energy into an aperture disk
                                     instead of being averaged down. We approximate true
                                     per-pixel splatting with **per-CoC-bin convolution by an
                                     aperture kernel**: each depth layer's energy is spread by
                                     a kernel that sums to 1, so an isolated highlight blooms
                                     into a full-brightness aperture shape.
  3. Signed CoC from depth        → CoC = K·(disparity − disp_focus); |CoC| is the blur radius.
  4. Energy-conserving accumulate → premultiplied color and weight spread together, divide at
                                     the end. No brightening/darkening as CoC changes.
  5. Layer by depth, back-to-front→ far layers composited under near ones; no cross-contam.
  6. Aperture = the splat kernel  → disk, or N-gon for N iris blades → polygonal bokeh.
  7. Optional lens character      → cat's-eye (optical vignetting) toward the frame edges."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import linear_to_srgb, srgb_to_linear, tonemap_highlights


@dataclass
class BlurParams:
    k: float = 60.0            # blur strength slider 0..100 → max CoC radius
    disp_focus: float = 0.7    # focal plane in disparity space [0,1] (1 = nearest)
    blades: int = 0            # 0 => circular aperture; >=3 => N-gon bokeh
    rotation: float = 0.0      # aperture rotation, radians
    highlight_boost: float = 0.18  # localized bloom strength for true highlights (0 = off)
    cat_eye: float = 0.2       # optical vignetting toward edges (0 = off)
    n_bins: int = 10           # CoC quantization layers (smoother gradient)


# ---- aperture kernels ---------------------------------------------------------------


def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    d = 2 * r + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    k = (xx * xx + yy * yy <= (r + 0.5) ** 2).astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else k


def _ngon_kernel(radius: int, blades: int, rotation: float) -> np.ndarray:
    r = max(1, int(radius))
    d = 2 * r + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    ang = np.arctan2(yy, xx) - rotation
    rad = np.sqrt(xx * xx + yy * yy)
    # regular polygon: distance to nearest edge along each ray (apothem form)
    seg = np.pi / blades
    phase = np.mod(ang, 2 * seg) - seg
    poly_r = (r + 0.5) * np.cos(seg) / np.clip(np.cos(phase), 1e-3, None)
    k = (rad <= poly_r).astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else k


def _aperture_kernel(radius: int, p: BlurParams) -> np.ndarray:
    if p.blades and p.blades >= 3:
        return _ngon_kernel(radius, p.blades, p.rotation)
    return _disk_kernel(radius)


def _apply_cat_eye(kernel: np.ndarray, fx: float, fy: float, strength: float) -> np.ndarray:
    """Squash + slide the aperture toward an oval pointing at frame center → cat's-eye bokeh
    that intensifies toward the edges (optical vignetting). fx,fy in [-1,1] = frame position."""
    rr = float(np.hypot(fx, fy))
    if strength <= 0 or rr < 1e-3:
        return kernel
    amount = strength * min(rr, 1.0)
    d = kernel.shape[0]
    r = d // 2
    # direction from center toward this pixel's frame position
    ux, uy = fx / (rr + 1e-6), fy / (rr + 1e-6)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    # squeeze perpendicular to the radial direction
    perp = -uy * xx + ux * yy
    para = ux * xx + uy * yy
    squeezed = (para * para) + (perp * perp) / max(1e-3, (1.0 - 0.7 * amount) ** 2)
    mask = (squeezed <= (r + 0.5) ** 2).astype(np.float32)
    out = kernel * mask
    s = out.sum()
    return out / s if s > 0 else kernel


# ---- main renderer ------------------------------------------------------------------


def render_lens_blur(bg_u8: np.ndarray, disparity: np.ndarray, p: BlurParams) -> np.ndarray:
    """Blur the clean background plate. Returns uint8 RGB (HxWx3), still in sRGB."""
    h, w = bg_u8.shape[:2]
    diag = float(np.hypot(h, w))
    max_radius = max(1.0, (p.k / 100.0) * diag * 0.045)  # K=100 ≈ 4.5% of diagonal

    # signed CoC; magnitude is the blur radius in px
    coc = disparity.astype(np.float32) - float(p.disp_focus)
    norm = max(float(p.disp_focus), 1.0 - float(p.disp_focus), 1e-3)
    radius_px = np.clip(np.abs(coc) / norm, 0.0, 1.0) * max_radius

    bg_lin = srgb_to_linear(bg_u8.astype(np.float32) / 255.0)

    # Bloom source: ONLY the energy above a high threshold (true highlights) glows. Dark and mid
    # tones contribute nothing, so the blur never lifts blacks or hazes the frame. The excess is
    # spread with the same depth-of-field as the blur and added back on top (localized, additive)
    # — not a global multiplicative boost, which is what caused the old wash.
    HI_THRESH = 0.82
    excess = np.clip(bg_lin - HI_THRESH, 0.0, None)
    do_bloom = p.highlight_boost > 0.0

    # quantize CoC into bins, composite far→near (largest radius first)
    edges = np.linspace(0.0, max_radius, int(p.n_bins) + 1)
    out_c = np.zeros((h, w, 3), np.float32)  # premultiplied accumulated color
    out_w = np.zeros((h, w, 1), np.float32)
    bloom = np.zeros((h, w, 3), np.float32)

    for i in range(int(p.n_bins), 0, -1):  # far (big radius) → near (small)
        r_lo, r_hi = edges[i - 1], edges[i]
        r_mid = 0.5 * (r_lo + r_hi)
        if r_mid < 0.75:
            # near-focus layer: effectively sharp, handled by the sharp pass below
            continue
        member = ((radius_px >= r_lo) & (radius_px < r_hi)).astype(np.float32)
        if member.sum() < 1.0:
            continue
        kernel = _aperture_kernel(int(round(r_mid)), p)
        if p.cat_eye > 0:
            # representative frame position = mean location of this bin's pixels
            ys, xs = np.where(member > 0)
            fx = (xs.mean() / w) * 2 - 1
            fy = (ys.mean() / h) * 2 - 1
            kernel = _apply_cat_eye(kernel, float(fx), float(fy), p.cat_eye)

        m3 = member[..., None]
        # energy-conserving scatter: color and weight spread together (kernel sums to 1)
        spread_c = cv2.filter2D(bg_lin * m3, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        spread_w = cv2.filter2D(member, -1, kernel, borderType=cv2.BORDER_REPLICATE)[..., None]
        cov = np.clip(spread_w, 0.0, 1.0)
        out_c = spread_c + out_c * (1.0 - cov)  # premultiplied OVER
        out_w = spread_w + out_w * (1.0 - cov)
        if do_bloom:
            bloom += cv2.filter2D(excess * m3, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # sharp (in-focus) pixels: composite the original linear bg under everything
    sharp_w = np.clip(1.0 - out_w, 0.0, 1.0)
    out_c = out_c + bg_lin * sharp_w
    out_w = out_w + sharp_w
    result_lin = out_c / np.clip(out_w, 1e-4, None)

    if do_bloom:  # additive glow, localized to bright out-of-focus areas only
        result_lin = result_lin + (p.highlight_boost * 2.0) * bloom

    result_lin = tonemap_highlights(result_lin)  # roll only true highlights toward white
    return (np.clip(linear_to_srgb(result_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

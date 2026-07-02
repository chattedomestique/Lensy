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
    n_bins: int = 10           # CoC quantization layers (legacy scatter path)
    focus_range: float = 0.12  # half-width (diopters, or normalized disparity) of the in-focus zone
    subject_dof: bool = False  # (cinematic removed) subject composited sharp
    n_layers: int = 22         # depth slices for the layered occlusion renderer
    chroma: float = 0.0        # lateral chromatic aberration in the bokeh (0 = off; ~0.01 subtle)
    swirl: float = 0.0         # Petzval swirly bokeh — tangential smear growing toward the edges
    sweet: float = 0.0         # Lensbaby sweet-spot intensity — extra radial blur outside the spot
    sweet_size: float = 0.35   # radius (0..1 of half-diagonal) of the sharp sweet spot
    halation: float = 0.0      # reddish film glow bleeding out of the highlights
    halation_size: float = 0.4 # how far the halation glow spreads
    ca: float = 0.0            # lateral chromatic aberration — colour fringing toward the edges
    distortion: float = 0.0    # barrel lens distortion (paired with CA in the UI)


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


# ---- depth → circle-of-confusion, and the shared scatter core -----------------------

_HI_THRESH = 0.82  # linear luminance above which highlights bloom
_DIOPTER_GAIN = 0.16  # px-per-diopter scaling at K=100 (× diagonal); tunes how fast blur grows


def focal_radius(signal: np.ndarray, focus: float, metric: bool, p: BlurParams) -> np.ndarray:
    """Per-pixel blur radius (px) — the circle of confusion. This is the physically-correct
    depth-of-field falloff and the SINGLE field used for both background and subject.

    metric (Depth Pro, meters): CoC ∝ |1/D − 1/D_focus| — the real thin-lens near-field model.
    Blur grows with the DIOPTER difference from the focal plane, so an object a bit past the
    subject blurs a little and one far behind blurs a lot — true distance falloff (the floor at
    your feet ≠ the door 15 ft back). `focus_range` is an in-focus dead zone, in diopters.

    non-metric (Depth Anything, relative): fall back to normalized-disparity distance — no true
    metres, so the falloff is only approximate."""
    h, w = signal.shape[:2]
    diag = float(np.hypot(h, w))
    max_radius = max(1.0, (p.k / 100.0) * diag * 0.11)  # ceiling so the far field saturates softly

    if metric:
        f_dpt = 1.0 / max(float(focus), 0.08)
        d_dpt = 1.0 / np.clip(signal.astype(np.float32), 0.08, None)
        diff = np.abs(d_dpt - f_dpt)  # diopters from the focal plane
        eff = np.clip(diff - float(p.focus_range), 0.0, None)
        radius = (p.k / 100.0) * diag * _DIOPTER_GAIN * eff
    else:
        norm = max(float(focus), 1.0 - float(focus), 1e-3)
        diff = np.abs(signal.astype(np.float32) - float(focus))
        eff = np.clip((diff - float(p.focus_range)) / norm, 0.0, 1.0)
        radius = eff * max_radius

    return np.minimum(radius, max_radius).astype(np.float32)


def scatter_dof(
    color_lin: np.ndarray,      # linear RGB (HxWx3), NOT premultiplied
    weight: np.ndarray,         # opacity per pixel (HxWx1); ones for an opaque plate, alpha for a cutout
    radius_px: np.ndarray,      # per-pixel blur radius
    p: BlurParams,
    bloom_excess: np.ndarray | None = None,
):
    """Energy-conserving, aperture-shaped scatter, quantized into CoC bins and composited
    far→near. Returns (premultiplied_color, coverage, bloom). Works for any layer — the
    background (weight=1) or a soft-alpha subject (weight=α) — so the same optics apply to both."""
    h, w = radius_px.shape
    premult = color_lin * weight  # premultiplied so soft edges blend without fringing
    max_r = float(radius_px.max())
    edges = np.linspace(0.0, max(max_r, 1e-3), int(p.n_bins) + 1)

    out_c = np.zeros((h, w, 3), np.float32)
    out_w = np.zeros((h, w, 1), np.float32)
    bloom = np.zeros((h, w, 3), np.float32)

    for i in range(int(p.n_bins), 0, -1):  # far (big radius) → near (small)
        r_lo, r_hi = edges[i - 1], edges[i]
        r_mid = 0.5 * (r_lo + r_hi)
        if r_mid < 0.75:
            continue  # near-focus layer handled by the sharp pass below
        member = ((radius_px >= r_lo) & (radius_px < r_hi)).astype(np.float32)
        if member.sum() < 1.0:
            continue
        kernel = _aperture_kernel(int(round(r_mid)), p)
        if p.cat_eye > 0:
            ys, xs = np.where(member > 0)
            fx = (xs.mean() / w) * 2 - 1
            fy = (ys.mean() / h) * 2 - 1
            kernel = _apply_cat_eye(kernel, float(fx), float(fy), p.cat_eye)

        m3 = member[..., None]
        spread_c = cv2.filter2D(premult * m3, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        # filter2D collapses a (H,W,1) input to (H,W) — restore the channel axis
        spread_w = cv2.filter2D(weight * m3, -1, kernel, borderType=cv2.BORDER_REPLICATE)[..., None]
        cov = np.clip(spread_w, 0.0, 1.0)
        out_c = spread_c + out_c * (1.0 - cov)  # premultiplied OVER
        out_w = spread_w + out_w * (1.0 - cov)
        if bloom_excess is not None:
            bloom += cv2.filter2D(bloom_excess * m3, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    sharp = (radius_px < 0.75)[..., None].astype(np.float32)
    out_c = out_c + premult * sharp
    out_w = out_w + weight * sharp
    return out_c, out_w, bloom


# ---- background + foreground renderers ----------------------------------------------


def render_lens_blur(bg_u8: np.ndarray, radius_px: np.ndarray, p: BlurParams) -> np.ndarray:
    """Blur the clean (inpainted) background plate given a per-pixel CoC radius. Returns uint8 sRGB."""
    h, w = bg_u8.shape[:2]
    bg_lin = srgb_to_linear(bg_u8.astype(np.float32) / 255.0)
    excess = np.clip(bg_lin - _HI_THRESH, 0.0, None) if p.highlight_boost > 0 else None

    out_c, out_w, bloom = scatter_dof(bg_lin, np.ones((h, w, 1), np.float32), radius_px, p, excess)
    result_lin = out_c / np.clip(out_w, 1e-4, None)
    if excess is not None:  # additive glow, only on bright out-of-focus areas
        result_lin = result_lin + (p.highlight_boost * 2.0) * bloom
    result_lin = tonemap_highlights(result_lin)
    return (np.clip(linear_to_srgb(result_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def blur_foreground_dof(fg_srgb: np.ndarray, alpha: np.ndarray, radius_px: np.ndarray, p: BlurParams):
    """Apply the SAME depth-of-field to the subject given its per-pixel CoC radius: off-focal parts
    soften, focal parts stay sharp — so the person isn't a flat sticker. Returns (premultiplied
    linear color, coverage α) for premultiplied compositing.

    The blurred coverage is clamped to the (sharp) matte so the subject softens *internally* but
    can't spread its edge out over the background — that outward spread is what read as a translucent
    rim / glow. Result: clean silhouette, depth-softened interior."""
    fg_lin = srgb_to_linear(np.clip(fg_srgb, 0.0, 1.0))
    a = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
    out_c, out_w, _ = scatter_dof(fg_lin, a, radius_px, p, None)
    cov = np.minimum(out_w, a)                       # never exceed the matte
    scale = cov / np.clip(out_w, 1e-4, None)         # keep premultiplied color consistent
    return out_c * scale, cov

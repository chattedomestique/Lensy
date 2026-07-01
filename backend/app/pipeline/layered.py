"""Layered occlusion-aware depth-of-field renderer — physically-based lens/aperture modeling.

This is the "do it right" renderer (§7.3). It follows the classical layered scatter approach
(Barsky & Kosloff survey; Google's mobile Portrait Mode, Wadhwa et al. 2018; Dr.Bokeh's layered
scene idea) adapted to what Lensy already has — a soft matte and a LaMa-inpainted background:

  1. Circle of confusion from METRIC depth via the thin-lens near-field model:
        CoC ∝ |1/D − 1/D_focus|                    (see blur.focal_radius)
     so blur grows with true distance from the focal plane — real falloff.

  2. The scene is two occlusion sheets, each with complete color:
        • BACKGROUND — the LaMa-inpainted plate (opaque; exists even behind the subject),
        • FOREGROUND — the decontaminated subject with the soft matte as its coverage α.

  3. Each sheet is rendered by decomposing it into many DEPTH slices (fine, with soft *tent*
     membership so slices blend — no banding, and objects spanning depth get smooth internal
     falloff). Each slice is scattered by ITS circle of confusion with an energy-conserving,
     aperture-shaped kernel (kernel sums to 1 → a point becomes a disk of 1/area brightness:
     true bokeh), then composited **back-to-front** with premultiplied alpha. Near slices occlude
     far ones; a blurred near slice keeps soft (α<1) edges that correctly reveal what's behind.

  4. Compose FOREGROUND over BACKGROUND (premultiplied). The subject's blurred edges reveal the
     already-blurred, inpainted background — occlusion-correct, no halo, no chewed edge.

Optics modeled: aperture shape (disk / N-gon iris blades + rotation), optical vignetting
(cat's-eye toward the frame edge), highlight bloom (bokeh balls), and optional lateral chromatic
aberration (per-channel CoC) for bokeh fringing."""

from __future__ import annotations

import cv2
import numpy as np

from .blur import BlurParams, _apply_cat_eye, _aperture_kernel, focal_radius
from .color import linear_to_srgb, srgb_to_linear, tonemap_highlights

_HI_THRESH = 0.82  # linear luminance above which highlights bloom


def _layer_coord(signal: np.ndarray, metric: bool) -> np.ndarray:
    """Per-pixel ordering coordinate — larger = NEARER. In diopters for metric depth (so slices
    are perceptually uniform in blur), or the raw disparity-like signal otherwise."""
    if metric:
        return (1.0 / np.clip(signal.astype(np.float32), 0.08, None))  # diopters, near = large
    return signal.astype(np.float32)  # disparity-like, near = large


def _kernel_for(radius: int, p: BlurParams, member: np.ndarray, h: int, w: int) -> np.ndarray:
    kernel = _aperture_kernel(radius, p)
    if p.cat_eye > 0:
        ys, xs = np.where(member > 0.05)
        if xs.size:
            fx = (xs.mean() / w) * 2 - 1
            fy = (ys.mean() / h) * 2 - 1
            kernel = _apply_cat_eye(kernel, float(fx), float(fy), p.cat_eye)
    return kernel


def _scatter_layer(premult: np.ndarray, la: np.ndarray, kernel: np.ndarray, p: BlurParams):
    """Spread one depth slice's premultiplied color + coverage by the aperture kernel. Optional
    lateral chromatic aberration: blur the R and B channels with a slightly larger/smaller kernel
    so out-of-focus edges fringe (a real fast-lens artifact)."""
    sa = cv2.filter2D(la, -1, kernel, borderType=cv2.BORDER_REPLICATE)[..., None]
    if p.chroma <= 0:
        sc = cv2.filter2D(premult, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return sc, sa
    r = cv2.filter2D(premult[:, :, 0], -1, _scale_kernel(kernel, 1.0 + p.chroma), borderType=cv2.BORDER_REPLICATE)
    g = cv2.filter2D(premult[:, :, 1], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    b = cv2.filter2D(premult[:, :, 2], -1, _scale_kernel(kernel, 1.0 - p.chroma), borderType=cv2.BORDER_REPLICATE)
    return np.stack([r, g, b], axis=-1), sa


def _scale_kernel(kernel: np.ndarray, s: float) -> np.ndarray:
    d = kernel.shape[0]
    nd = max(3, int(round(d * s)) | 1)
    k = cv2.resize(kernel, (nd, nd), interpolation=cv2.INTER_LINEAR)
    tot = k.sum()
    return k / tot if tot > 0 else kernel


def render_sheet(
    color_lin: np.ndarray,     # linear RGB, complete over the sheet's support
    base_alpha: np.ndarray,    # per-pixel opacity (1 for the opaque background, matte α for subject)
    coord: np.ndarray,         # ordering coordinate (near = large)
    radius_px: np.ndarray,     # per-pixel CoC radius
    p: BlurParams,
    bloom_excess: np.ndarray | None = None,
):
    """Render one occlusion sheet: fine depth slices, energy-conserving aperture scatter,
    composited back-to-front. Returns (premultiplied color, coverage α, bloom)."""
    h, w = radius_px.shape
    K = max(2, int(p.n_layers))
    lo, hi = float(np.percentile(coord, 1)), float(np.percentile(coord, 99))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    t = (coord - lo) / (hi - lo) * (K - 1)  # continuous slice coordinate in [0, K-1]
    # HARD assignment (each pixel to exactly one slice) so an opaque pixel keeps α=1 — soft/tent
    # splitting plus back-to-front OVER would treat one surface as two occluding layers and bleed
    # coverage away (the subject would go translucent). Blurred layers overlap, so no banding.
    slot = np.clip(np.rint(t).astype(np.int32), 0, K - 1)

    acc_c = np.zeros((h, w, 3), np.float32)
    acc_a = np.zeros((h, w, 1), np.float32)
    bloom = np.zeros((h, w, 3), np.float32)

    # composite far → near (small coord → large): near slices go OVER far ones (occlusion)
    for k in range(K):
        member = (slot == k).astype(np.float32)
        msum = float(member.sum())
        if msum < 1.0:
            continue
        la = (base_alpha.astype(np.float32) * member)[..., None]
        if float(la.sum()) < 1e-3:
            continue
        r_k = float((radius_px * member).sum() / msum)  # this slice's representative CoC

        if r_k < 0.75:  # in-focus slice — no spread
            sc, sa = color_lin * la, la
        else:
            kernel = _kernel_for(int(round(r_k)), p, member, h, w)
            sc, sa = _scatter_layer(color_lin * la, la, kernel, p)
            if bloom_excess is not None:
                bloom += cv2.filter2D(bloom_excess * la, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        if bloom_excess is not None and r_k < 0.75:
            bloom += bloom_excess * la

        cov = np.clip(sa, 0.0, 1.0)
        acc_c = sc + acc_c * (1.0 - cov)   # premultiplied OVER (near over far)
        acc_a = sa + acc_a * (1.0 - cov)

    return acc_c, np.clip(acc_a, 0.0, 1.0), bloom


def _field(h: int, w: int):
    """Normalized field radius (0 at center → ~1 at the corners) and the center coords."""
    cx, cy = w / 2.0, h / 2.0
    halfdiag = 0.5 * float(np.hypot(w, h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / halfdiag).astype(np.float32), cx, cy


def _apply_swirl(img: np.ndarray, strength: float) -> np.ndarray:
    """Petzval: optical-vignetting swirl approximated as a tangential (spin) blur whose angle grows
    with field radius — off-axis bokeh streaks tangentially, curling around the sharp center.
    Uses many gaussian-weighted samples so the streak is a smooth arc (no discrete ghost repeats)."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    r_frac, cx, cy = _field(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    max_ang = 0.38 * float(strength)  # radians at the corner
    n = 21
    offs = np.linspace(-1.0, 1.0, n)
    wts = np.exp(-((offs * 1.6) ** 2))  # gaussian taper → smooth streak, not stacked copies
    wts = wts / wts.sum()
    acc = np.zeros_like(img)
    for off, wt in zip(offs, wts):
        ang = (off * max_ang) * (r_frac ** 1.5)  # per-pixel rotation, stronger toward edges
        c, s = np.cos(ang), np.sin(ang)
        srcx = np.ascontiguousarray(cx + dx * c - dy * s, dtype=np.float32)
        srcy = np.ascontiguousarray(cy + dx * s + dy * c, dtype=np.float32)
        acc += float(wt) * cv2.remap(img, srcx, srcy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return acc.astype(np.float32)


def _var_radial_blur(arr: np.ndarray, amt: np.ndarray, max_sig: float) -> np.ndarray:
    """Per-pixel variable gaussian blur via a smooth multi-level blend by `amt` (0..1)."""
    levels = [
        arr,
        cv2.GaussianBlur(arr, (0, 0), sigmaX=max_sig * 0.35),
        cv2.GaussianBlur(arr, (0, 0), sigmaX=max_sig * 0.7),
        cv2.GaussianBlur(arr, (0, 0), sigmaX=max_sig),
    ]
    idx = amt * (len(levels) - 1)
    lo = np.clip(np.floor(idx).astype(np.int32), 0, len(levels) - 2)
    frac = idx - lo
    if arr.ndim == 3:
        frac = frac[..., None]
    out = np.array(levels[0])
    for k in range(len(levels) - 1):
        m = lo == k
        if arr.ndim == 3:
            m = m[..., None]
        out = np.where(m, levels[k] * (1.0 - frac) + levels[k + 1] * frac, out)
    return out.astype(np.float32)


def _sweet_amount(h: int, w: int, p: BlurParams, cx: float, cy: float) -> np.ndarray:
    """Lensbaby field falloff: 0 inside the sweet spot → `sweet` at the far edge, measured from
    the sweet-spot center (cx, cy) — the SUBJECT's centroid, so blur radiates out from the person."""
    halfdiag = 0.5 * float(np.hypot(w, h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / halfdiag
    t = np.clip((r - p.sweet_size) / max(1.0 - p.sweet_size, 1e-3), 0.0, 1.0)
    return ((t * t * (3.0 - 2.0 * t)) * float(p.sweet)).astype(np.float32)


def _apply_halation(img_lin: np.ndarray, strength: float, size: float) -> np.ndarray:
    """Film halation: light scatters inside the emulsion and reflects off the base, so a warm
    RED-ORANGE glow bleeds out of the HIGHLIGHTS only (unlike bloom, which glows everywhere). We
    isolate the bright pixels, blur them, tint red-orange, and add back — the physical falloff."""
    if strength <= 0:
        return img_lin
    h, w = img_lin.shape[:2]
    lum = img_lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = np.clip((lum - 0.7) / 0.3, 0.0, 1.0)  # highlights only
    sigma = max(2.0, float(size) * float(np.hypot(h, w)) * 0.02)
    glow = cv2.GaussianBlur(hi, (0, 0), sigmaX=sigma)[..., None]
    tint = np.array([1.0, 0.32, 0.12], np.float32)  # red-orange
    return (img_lin + (strength * 0.9) * glow * tint).astype(np.float32)


def _apply_ca(img_u8: np.ndarray, amount: float) -> np.ndarray:
    """Lateral (transverse) chromatic aberration: the R and B images are magnified slightly
    differently about the optical axis, so colour fringes appear and grow toward the frame edges."""
    if amount <= 0:
        return img_u8
    h, w = img_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    k = float(amount) * 0.012  # max magnification delta

    def scaled(ch: np.ndarray, s: float) -> np.ndarray:
        srcx = np.ascontiguousarray(cx + dx / s, dtype=np.float32)
        srcy = np.ascontiguousarray(cy + dy / s, dtype=np.float32)
        return cv2.remap(ch, srcx, srcy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    r = scaled(img_u8[:, :, 0], 1.0 + k)  # red image slightly larger
    b = scaled(img_u8[:, :, 2], 1.0 - k)  # blue slightly smaller
    return np.stack([r, img_u8[:, :, 1], b], axis=-1)


def render_layered_dof(
    fg_srgb: np.ndarray,
    alpha: np.ndarray,
    clean_bg_u8: np.ndarray,
    fg_signal: np.ndarray,
    bg_signal: np.ndarray,
    focus: float,
    metric: bool,
    p: BlurParams,
) -> np.ndarray:
    """Full occlusion-aware render → final uint8 sRGB image."""
    h, w = alpha.shape[:2]

    # --- background sheet (opaque, complete via inpaint) ---
    bg_lin = srgb_to_linear(clean_bg_u8.astype(np.float32) / 255.0)
    bg_radius = focal_radius(bg_signal, focus, metric, p)
    bg_coord = _layer_coord(bg_signal, metric)
    excess = np.clip(bg_lin - _HI_THRESH, 0.0, None) if p.highlight_boost > 0 else None
    bgc, bga, bgbloom = render_sheet(bg_lin, np.ones((h, w), np.float32), bg_coord, bg_radius, p, excess)
    bg_rendered = bgc / np.clip(bga, 1e-4, None)  # un-premultiply the opaque plate
    if excess is not None:
        bg_rendered = bg_rendered + (p.highlight_boost * 2.0) * bgbloom
    if p.swirl > 0:  # Petzval swirl — tangential smear of the background, subject stays sharp.
        # a touch of smoothing first blends the discrete depth-slice steps so they don't streak
        # into visible bands under the swirl.
        bg_rendered = cv2.GaussianBlur(bg_rendered, (0, 0), sigmaX=1.4)
        bg_rendered = _apply_swirl(bg_rendered, p.swirl)

    # --- foreground sheet (subject) ---
    fg_lin = srgb_to_linear(np.clip(fg_srgb, 0.0, 1.0))
    if p.subject_dof:
        # cinematic: the subject gets the same layered DoF (off-focal parts soften, occlusion-correct)
        fg_radius = focal_radius(fg_signal, focus, metric, p)
        fg_coord = _layer_coord(fg_signal, metric)
        fgc, fga, _ = render_sheet(fg_lin, np.clip(alpha, 0.0, 1.0), fg_coord, fg_radius, p, None)
    else:
        # keep the subject perfectly sharp — clean cutout over the depth-graded background
        a = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
        fgc, fga = fg_lin * a, a

    # --- Lensbaby sweet spot: blur everything outside a sweet spot centered on the SUBJECT.
    # Applied to the premultiplied sheets (fg color+alpha AND background) BEFORE compositing, so a
    # softened subject edge fades to transparent rather than bleeding a bright halo (no glow). ---
    if p.sweet > 0:
        a2 = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        tot = float(a2.sum())
        if tot > 1.0:  # sweet spot centers on the subject's centroid
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = float((xx * a2).sum() / tot), float((yy * a2).sum() / tot)
        else:
            cx, cy = w / 2.0, h / 2.0
        amt = _sweet_amount(h, w, p, cx, cy)
        max_sig = max(1.0, (p.k / 100.0) * float(np.hypot(h, w)) * 0.05)
        fgc = _var_radial_blur(fgc, amt, max_sig)
        fga = _var_radial_blur(fga[..., 0], amt, max_sig)[..., None]
        bg_rendered = _var_radial_blur(bg_rendered, amt, max_sig)

    # --- compose foreground OVER background (premultiplied) ---
    out_lin = fgc + bg_rendered * (1.0 - fga)
    if p.halation > 0:  # warm film glow out of the highlights (linear light, before tonemap)
        out_lin = _apply_halation(out_lin, p.halation, p.halation_size)
    out_lin = tonemap_highlights(out_lin)
    out = (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if p.ca > 0:  # lateral chromatic aberration — geometric channel shift, last
        out = _apply_ca(out, p.ca)
    return out

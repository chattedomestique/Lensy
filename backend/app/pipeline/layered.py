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

    # --- compose foreground OVER background (premultiplied) ---
    out_lin = fgc + bg_rendered * (1.0 - fga)
    out_lin = tonemap_highlights(out_lin)
    return (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

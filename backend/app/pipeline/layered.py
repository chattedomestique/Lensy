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

import os

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


def _apply_bloom(img_lin: np.ndarray, strength: float) -> np.ndarray:
    """Bloom: bright areas spill a soft NEUTRAL glow across the frame (a lens/veiling-glare look).
    Unlike halation (highlights only, warm), bloom is white and reads on any bright region — skin
    highlights, sky, specular. Multi-scale so it's soft, not a hard ring. Applied in linear light."""
    if strength <= 0:
        return img_lin
    h, w = img_lin.shape[:2]
    diag = float(np.hypot(h, w))
    lum = img_lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = np.clip(lum - 0.55, 0.0, None)  # everything reasonably bright (linear)
    g1 = cv2.GaussianBlur(hi, (0, 0), sigmaX=max(2.0, diag * 0.008))
    g2 = cv2.GaussianBlur(hi, (0, 0), sigmaX=max(4.0, diag * 0.028))
    glow = (0.6 * g1 + 0.4 * g2)[..., None]
    return (img_lin + (strength * 4.5) * glow).astype(np.float32)


def _apply_halation(img_lin: np.ndarray, strength: float, size: float) -> np.ndarray:
    """Film halation: light scatters inside the emulsion and reflects off the base, so a warm
    RED-ORANGE glow bleeds out of the HIGHLIGHTS only (unlike bloom, which glows everywhere). We
    isolate the bright pixels, blur them, tint red-orange, and add back — the physical falloff."""
    if strength <= 0:
        return img_lin
    h, w = img_lin.shape[:2]
    lum = img_lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = np.clip((lum - 0.5) / 0.5, 0.0, 1.0) ** 1.5  # highlights, softly ramped in
    sigma = max(3.0, float(size) * float(np.hypot(h, w)) * 0.03)
    glow = cv2.GaussianBlur(hi, (0, 0), sigmaX=sigma)[..., None]
    tint = np.array([1.0, 0.28, 0.1], np.float32)  # red-orange
    return (img_lin + (strength * 2.6) * glow * tint).astype(np.float32)


def _apply_ca(img_u8: np.ndarray, amount: float) -> np.ndarray:
    """Lateral (transverse) chromatic aberration: the R and B images are magnified slightly
    differently about the optical axis, so colour fringes appear and grow toward the frame edges."""
    if amount <= 0:
        return img_u8
    h, w = img_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    k = float(amount) * 0.018  # max magnification delta

    def scaled(ch: np.ndarray, s: float) -> np.ndarray:
        srcx = np.ascontiguousarray(cx + dx / s, dtype=np.float32)
        srcy = np.ascontiguousarray(cy + dy / s, dtype=np.float32)
        return cv2.remap(ch, srcx, srcy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    r = scaled(img_u8[:, :, 0], 1.0 + k)  # red image slightly larger
    b = scaled(img_u8[:, :, 2], 1.0 - k)  # blue slightly smaller
    return np.stack([r, img_u8[:, :, 1], b], axis=-1)


def _apply_distortion(img_u8: np.ndarray, amount: float) -> np.ndarray:
    """Barrel lens distortion — straight lines bow outward, the wide-angle/phone-lens look. A
    radial resample about the optical axis; a compensating zoom keeps the corners inside frame."""
    if amount <= 0:
        return img_u8
    h, w = img_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx * nx + ny * ny
    k = float(amount) * 0.28  # barrel strength
    factor = (1.0 + k * r2) / (1.0 + k)  # compensating zoom so corners stay in bounds
    srcx = np.ascontiguousarray(cx + (xx - cx) * factor, dtype=np.float32)
    srcy = np.ascontiguousarray(cy + (yy - cy) * factor, dtype=np.float32)
    return cv2.remap(img_u8, srcx, srcy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _fill_sharp_islands(radius_px: np.ndarray, a_sub: np.ndarray) -> np.ndarray:
    """Kill SHARP ISLANDS in the background CoC map: a background region that is sharp (low CoC)
    but ringed by heavy blur — a bystander/pole standing at the subject's focal distance — renders
    as a hard-edged sharp patch floating in the blur (the "hard stop, zero falloff"). Depth can't
    fix it (the object really is at the focal plane) and a distance floor can't (it's adjacent to
    the subject). But it's obviously wrong for a portrait: nothing in the *background* should stay
    tack-sharp while everything around it is blurred.

    So detect pixels whose surroundings are much blurrier than they are, and raise them toward the
    surrounding blur — smooth falloff, no hard edge. The SUBJECT is composited from a SEPARATE sharp
    layer and never reads this map, so this can't soften the subject. The ground at the subject's
    feet is a low-CoC region CONNECTED to the near field (it grades to blur), not an enclosed island,
    so the surrounding-blur estimate there is itself low and it's left alone."""
    if float(radius_px.max()) < 3.0:
        return radius_px
    h, w = radius_px.shape
    # "typical blur a little way out" from each pixel: a grey close bridges gaps up to the kernel,
    # so an enclosed sharp island takes its ring's (high) value while an open sharp field keeps its
    # own (low) value.
    k = int(np.clip(np.percentile(radius_px, 80), 12, min(h, w) // 6)) | 1
    surround = cv2.morphologyEx(radius_px, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # only raise where the surroundings are genuinely blurrier (an island); ramp in softly so we
    # don't introduce a new step at the fill boundary.
    fill = np.maximum(radius_px, surround - 0.10 * float(radius_px.max()))
    out = np.maximum(radius_px, np.minimum(fill, surround))
    return cv2.GaussianBlur(out, (0, 0), sigmaX=max(3.0, k / 3.5)).astype(np.float32)


def _dilate_coc(radius_px: np.ndarray) -> np.ndarray:
    """Spread the circle-of-confusion ACROSS depth edges so a differently-blurred neighbour can't
    leave a hard 'field-border' line, and a thin sharp feature sitting in a blurred region inherits
    its neighbours' blur instead of staying at radius 0 (GPU Gems 3 ch.28 / Hammon: take the larger
    of a pixel's own CoC and its dilated CoC). A morphological max over a modest window bridges the
    hard step; a light blur of the result keeps the ramp smooth. Local — does not globally over-blur."""
    if float(radius_px.max()) < 1.5:
        return radius_px
    h, w = radius_px.shape[:2]
    win = max(3, (min(h, w) // 90) | 1)  # ~edge-bridging window, not the whole frame
    dil = cv2.dilate(radius_px, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
    # raise each pixel toward the local max, but only partway, then smooth the RADIUS map firmly:
    # noisy depth makes the blur amount patchy (blotchy sharp/blur boundaries that read as lines —
    # a shadow band on the ground, an awning edge). Smoothing the radius only varies the blur
    # amount smoothly — it does NOT shift geometry like smoothing the depth would.
    spread = 0.6 * dil + 0.4 * radius_px
    out = np.maximum(radius_px, spread)
    return cv2.GaussianBlur(out, (0, 0), sigmaX=max(3.0, min(h, w) / 70.0)).astype(np.float32)


def _near_foreground_alpha(
    signal: np.ndarray, focus: float, metric: bool, a_sub: np.ndarray
) -> np.ndarray:
    """Soft coverage of NEAR-foreground occluders: non-subject pixels meaningfully closer than the
    focal plane (a pole by the lens, a near railing, etc.). These need to be blurred as their own
    layer that *spreads* over what's behind — otherwise they stay hard-edged in the flat plate."""
    h, w = signal.shape[:2]
    if metric:  # metres: nearer = smaller
        near = (signal < float(focus) * 0.7).astype(np.float32)
    else:  # normalized disparity: nearer = larger — require STRONGLY near, not a local depth spike
        near = (signal > float(focus) + 0.22).astype(np.float32)
    near = near * (1.0 - a_sub)  # never the subject
    if float(near.sum()) < 1.0:
        return near
    k = max(3, (min(h, w) // 200) | 1)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # drop small blobs: a mono-depth spike on a dark object is not a real occluder (a pole/railing
    # by the lens is large). Keeps the layer from smearing tan blobs over background people/objects.
    nb, lab, stats, _ = cv2.connectedComponentsWithStats((near > 0.5).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(near)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= 0.005 * h * w:
            keep[lab == i] = 1.0
    near = near * keep
    if float(near.sum()) < 1.0:
        return near
    near = cv2.GaussianBlur(near, (0, 0), sigmaX=max(1.5, min(h, w) / 400.0))
    return np.clip(near, 0.0, 1.0)


def _inpaint_behind(bg_u8: np.ndarray, near_a: np.ndarray) -> np.ndarray:
    """Fill the far plate where a near occluder sits, so revealing behind its blurred edge shows
    clean background — not a copy of the occluder. It's only ever seen blurred behind a soft edge, so
    a smooth push-pull (normalized-blur) fill is ideal: it spreads the SURROUNDING colours inward with
    no hard fill boundary and no colour bleeding from a distant object (which cv2.Telea produced)."""
    h, w = bg_u8.shape[:2]
    hole = near_a > 0.12
    grow = max(3, min(h, w) // 120)
    hole = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) > 0
    if not hole.any():
        return bg_u8
    known = (~hole).astype(np.float32)  # 2D — cv2.GaussianBlur drops a (H,W,1) singleton channel
    src = bg_u8.astype(np.float32)
    acc = src * known[..., None]
    wsum = known.copy()
    diag = float(np.hypot(h, w))
    for sig in (diag * 0.006, diag * 0.02, diag * 0.06):
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=sig)
        wsum = cv2.GaussianBlur(wsum, (0, 0), sigmaX=sig)
    fill = acc / np.clip(wsum, 1e-3, None)[..., None]
    out = np.where(hole[..., None], fill, src)
    return np.clip(out, 0, 255).astype(np.uint8)


def _has_character(p: BlurParams) -> bool:
    """Any non-DoF character effect active? (bloom / halation / chromatic aberration / distortion /
    grain). When none are, the blur-off path can return the source pixels untouched — no sRGB
    round-trip, so 'blur off' is byte-for-byte the original photo."""
    return p.highlight_boost > 0 or p.halation > 0 or p.ca > 0 or p.distortion > 0 or p.grain > 0


def _finish(out_lin: np.ndarray, p: BlurParams) -> np.ndarray:
    """Glows (linear light, before tonemap, so the spill stays bright) → tonemap → encode to uint8
    sRGB. The output-medium passes (chromatic aberration, distortion, grain) are applied separately
    by `_output_optics` at the FINAL export resolution — see there for why."""
    if p.highlight_boost > 0:  # neutral bloom across the frame
        out_lin = _apply_bloom(out_lin, p.highlight_boost)
    if p.halation > 0:  # warm red-orange glow out of the highlights only
        out_lin = _apply_halation(out_lin, p.halation, p.halation_size)
    out_lin = tonemap_highlights(out_lin)
    return (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _defocus_mask(bg_radius: np.ndarray, a_sub: np.ndarray) -> np.ndarray:
    """0 where the frame is sharp (the subject, and in-focus background) → 1 where it is clearly out
    of focus. Drives the optional 'grain only in the blurred parts' blend. The most-blurred half of
    the background saturates to 1; the subject (matte) is forced to 0 so it never grains in that mode."""
    ref = max(0.5 * float(bg_radius.max()), 2.0)
    blur_norm = np.clip(bg_radius / ref, 0.0, 1.0)
    return np.clip((1.0 - np.clip(a_sub, 0.0, 1.0)) * blur_norm, 0.0, 1.0).astype(np.float32)


def _output_optics(out_u8: np.ndarray, p: BlurParams, defocus: np.ndarray | None = None) -> np.ndarray:
    """Geometric lens optics + film grain — the very last passes, applied at the FINAL export
    resolution. Grain is modeled per output pixel (crystalline at any size), and CA/distortion are
    radial remaps; running them here keeps grain crisp and the remaps precise even when a working-res
    render was upscaled for a full-res export (otherwise the grain gets stretched and blurred).

    `defocus` is the working-res out-of-focus mask; apply_grain resizes it and uses it (with
    p.grain_blend) to optionally confine grain to the blurred regions."""
    if p.ca > 0:  # lateral chromatic aberration — per-channel magnification
        out_u8 = _apply_ca(out_u8, p.ca)
    if p.distortion > 0:  # barrel lens distortion
        out_u8 = _apply_distortion(out_u8, p.distortion)
    if p.grain > 0:  # modeled film grain
        from .grain import apply_grain

        out_u8 = apply_grain(out_u8, p.grain, p.grain_size, p.grain_seed, p.grain_blend, defocus)
    return out_u8


def _composite_full_res(
    orig_u8: np.ndarray,          # full-resolution source photo (sRGB uint8)
    work_u8: np.ndarray | None,   # working-res source photo (for the decontamination correction)
    fg_srgb_work: np.ndarray,     # decontaminated foreground F at working res
    alpha_work: np.ndarray,       # soft matte at working res (already edge-snapped by refine)
    bg_rendered_lin: np.ndarray,  # working-res blurred background, linear light, un-premultiplied
    bg_radius: np.ndarray,        # working-res per-pixel CoC radius (post dilate + island fill)
    p: BlurParams,
) -> np.ndarray:
    """Final composite at the ORIGINAL resolution (§6: "composite the matte back at full res for
    output"). The heavy DoF ran at working res; here the sharp subject is composited at full res
    over the blurred background upsampled — so the subject keeps native detail, and in-focus
    background regions are restored from the full-res original rather than softened by the upscale.
    Premultiplied, linear light — the §7.2 ordering (blur clean bg → composite sharp fg) is kept."""
    H, W = orig_u8.shape[:2]
    h, w = alpha_work.shape[:2]
    scale = max(H / float(h), W / float(w))

    # 1) full-res subject colour = full-res original + the low-frequency edge decontamination
    #    correction (F − observed, ~0 except in the edge band), so the subject keeps full detail
    #    AND the anti-halo edge fix from decontaminate().
    if work_u8 is not None:
        corr = fg_srgb_work - work_u8.astype(np.float32) / 255.0
        corr = cv2.resize(corr, (W, H), interpolation=cv2.INTER_LINEAR)
        fg_full = np.clip(orig_u8.astype(np.float32) / 255.0 + corr, 0.0, 1.0)
    else:
        fg_full = orig_u8.astype(np.float32) / 255.0
    fg_full_lin = srgb_to_linear(fg_full)

    # 2) full-res matte: upscale the (already edge-snapped) working matte, then match the working
    #    path's sharp-subject edge treatment — pull in ~1px-equiv + feather — at full-res scale, so
    #    the blurred background covers the contaminated edge pixels (no light outline rim).
    a_full = cv2.resize(np.clip(alpha_work, 0.0, 1.0), (W, H), interpolation=cv2.INTER_LINEAR)
    er = max(3, 2 * int(round(scale)) + 1)  # erode ~scale px → matches the working path's 1px pull-in
    a_full = cv2.erode(a_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
    a_full = cv2.GaussianBlur(a_full, (0, 0), sigmaX=max(1.0, scale))[..., None]

    # 3) full-res background: upsample the blurred plate; where it is IN FOCUS (CoC≈0) blend back
    #    the sharp full-res original so focused background isn't softened by the upscale. Subject
    #    pixels there carry a_full≈1 and are overwritten in step 4, so no subject leaks in.
    bg_full = cv2.resize(bg_rendered_lin, (W, H), interpolation=cv2.INTER_CUBIC)
    sharp = cv2.GaussianBlur((bg_radius < 0.75).astype(np.float32), (0, 0), sigmaX=1.0)
    sharp_full = cv2.resize(sharp, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    # NEVER sharp-restore from the original inside the subject's silhouette band: there the original
    # still holds the contaminated edge C = αF+(1−α)B, so pulling it into the background would spill a
    # subject-coloured rim (the §7.1 halo). Keep the clean, subject-removed blurred plate there — this
    # matches the working-res composite, which only ever composites over the inpainted plate.
    subj = cv2.resize(
        (np.clip(alpha_work, 0.0, 1.0) > 0.02).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
    )
    grow = max(3, (int(round(3.0 * scale)) | 1))
    subj = cv2.dilate(subj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    subj = cv2.GaussianBlur(subj, (0, 0), sigmaX=max(1.5, scale))[..., None]
    sharp_full = sharp_full * (1.0 - subj)
    orig_lin = srgb_to_linear(orig_u8.astype(np.float32) / 255.0)
    bg_full = bg_full * (1.0 - sharp_full) + orig_lin * sharp_full

    # 4) premultiplied subject OVER background (both optics + grain applied at this full res)
    out_lin = fg_full_lin * a_full + bg_full * (1.0 - a_full)
    defocus = _defocus_mask(bg_radius, alpha_work)  # for the optional grain-in-defocus blend
    return _output_optics(_finish(out_lin, p), p, defocus)


def render_layered_dof(
    fg_srgb: np.ndarray,
    alpha: np.ndarray,
    clean_bg_u8: np.ndarray,
    fg_signal: np.ndarray,
    bg_signal: np.ndarray,
    focus: float,
    metric: bool,
    p: BlurParams,
    orig_srgb: np.ndarray | None = None,   # full-res source photo → composite the output at full res
    work_srgb: np.ndarray | None = None,   # working-res source photo (blur-off base + decontam ref)
) -> np.ndarray:
    """Full occlusion-aware render → final uint8 sRGB image.

    When `orig_srgb` is supplied and larger than the working image, the final composite is done at
    the original resolution so export keeps full quality (§6). If the blur is fully off (K≈0, no
    swirl/Lensbaby), the sharp subject over the sharp background is just the original photo, so it
    is returned pristine at full res with only the non-DoF character effects applied."""
    h, w = alpha.shape[:2]
    a_sub = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    # full-res output target (W, H) when the caller passes a larger original; None → work res == out
    full_wh = None
    if orig_srgb is not None and orig_srgb.shape[:2] != (h, w):
        full_wh = (int(orig_srgb.shape[1]), int(orig_srgb.shape[0]))

    # --- lens blur fully OFF (K≈0, no swirl/Lensbaby) -------------------------------------------
    # The circle of confusion is zero everywhere, so the depth-of-field composite reduces to the
    # untouched photo. Return it pristine (at full res when we have it) with only the non-DoF
    # character effects (bloom / halation / chroma / grain). The depth signal is untouched, so the
    # depth-driven tools still work — this is "blur off, depth kept".
    bg_radius = focal_radius(bg_signal, focus, metric, p)
    subject_sharp = not p.subject_dof
    fg_radius = None if subject_sharp else focal_radius(fg_signal, focus, metric, p)
    blur_off = (
        float(bg_radius.max()) < 0.5
        and (fg_radius is None or float(fg_radius.max()) < 0.5)
        and p.swirl <= 0
        and p.sweet <= 0
    )
    if blur_off and (full_wh is not None or work_srgb is not None):
        base = orig_srgb if full_wh is not None else work_srgb
        if not _has_character(p):
            return base.copy()  # pristine: untouched source pixels, no sRGB round-trip loss
        # nothing is out of focus → a zero defocus mask, so "grain in blurred parts only" yields none
        no_defocus = np.zeros((h, w), np.float32)
        return _output_optics(_finish(srgb_to_linear(base.astype(np.float32) / 255.0), p), p, no_defocus)
    # (no source image handed in — fall through: at zero CoC the normal composite below reconstructs
    #  the sharp subject over the clean background, which is the correct sharp image anyway.)
    # Near-foreground occluders (a pole/railing right by the lens): the unified back-to-front sheet
    # below ALREADY renders these occlusion-correctly — the near depth-slice scatters OVER the far
    # slices, so a near object's blurred edge softly reveals the background behind it. Pulling the
    # occluder into a SEPARATE spreading layer (with inpaint-behind) was strictly worse: its coverage
    # edge composited as a hard vertical "layer line" against whatever sat behind it, and the inpaint
    # occasionally bled a distant colour inward. So it's off by default; LENSY_NEAR_LAYER=1 re-enables
    # it for experimentation.
    if os.environ.get("LENSY_NEAR_LAYER") == "1":
        near_a = _near_foreground_alpha(bg_signal, focus, metric, a_sub)
        has_near = float(near_a.sum()) > (0.0006 * h * w)
    else:
        near_a = np.zeros((h, w), np.float32)
        has_near = False

    # --- background sheet (opaque, complete via inpaint) ---
    # if there are near occluders, remove them from the FAR plate (+ fill behind) so the sheet is
    # clean background; the occluders themselves are rendered as a separate spreading layer below.
    far_bg_u8 = _inpaint_behind(clean_bg_u8, near_a) if has_near else clean_bg_u8
    bg_lin = srgb_to_linear(far_bg_u8.astype(np.float32) / 255.0)
    bg_radius = _dilate_coc(bg_radius)  # thin features inherit neighbour blur (bg_radius from above)
    bg_radius = _fill_sharp_islands(bg_radius, a_sub)  # kill sharp background patches floating in blur
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

    # --- full-res composite (common case: sharp subject, no Lensbaby, no near-occluder layer) ---
    # Do the subject-over-background composite at the ORIGINAL resolution so export keeps full
    # quality (§6). The blurred background is upsampled (it's low-frequency — lossless), the subject
    # is composited from the full-res original. Other cases fall through to the working-res composite
    # below and are upscaled at the end, so the output is always at least full resolution.
    # (swirl/sweet modify the background globally — including in-focus regions the full-res path
    #  restores from the sharp original — so they take the working-res path below and are upscaled.)
    if full_wh is not None and subject_sharp and not has_near and p.sweet <= 0 and p.swirl <= 0:
        return _composite_full_res(orig_srgb, work_srgb, fg_srgb, alpha, bg_rendered, bg_radius, p)

    # --- foreground sheet (subject) ---
    fg_lin = srgb_to_linear(np.clip(fg_srgb, 0.0, 1.0))
    if p.subject_dof:
        # cinematic: the subject gets the same layered DoF (off-focal parts soften, occlusion-correct)
        fg_radius = focal_radius(fg_signal, focus, metric, p)
        fg_coord = _layer_coord(fg_signal, metric)
        fgc, fga, _ = render_sheet(fg_lin, np.clip(alpha, 0.0, 1.0), fg_coord, fg_radius, p, None)
    else:
        # keep the subject perfectly sharp, but pull the matte in ~1px and feather it a touch so the
        # blurred background covers the contaminated edge pixels (no light outline rim) and the sharp
        # cutout doesn't read as a hard sticker — most visible at extreme blur.
        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=1.0)[..., None]
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
        # quartered alongside the DoF recalibration (0.05 → 0.0125) so the Lensbaby smear keeps its
        # original balance vs the depth-of-field ceiling instead of dwarfing it (§3 goal).
        max_sig = max(1.0, (p.k / 100.0) * float(np.hypot(h, w)) * 0.0125)
        fgc = _var_radial_blur(fgc, amt, max_sig)
        fga = _var_radial_blur(fga[..., 0], amt, max_sig)[..., None]
        bg_rendered = _var_radial_blur(bg_rendered, amt, max_sig)

    # --- compose subject OVER background (premultiplied) ---
    out_lin = fgc + bg_rendered * (1.0 - fga)

    # --- near-foreground occluder layer: scatter the occluders with their soft alpha and composite
    # OVER everything, so their edges bloom outward (true foreground bokeh, revealing the clean far
    # plate + subject behind) instead of sitting as a hard line in the flat background sheet. ---
    if has_near:
        near_lin = srgb_to_linear(clean_bg_u8.astype(np.float32) / 255.0)  # original: still has them
        near_radius = _dilate_coc(focal_radius(bg_signal, focus, metric, p))
        near_coord = _layer_coord(bg_signal, metric)
        nc, na, _ = render_sheet(near_lin, near_a, near_coord, near_radius, p, None)
        out_lin = nc + out_lin * (1.0 - np.clip(na, 0.0, 1.0))

    # glows + tonemap + encode (bloom/halation are low-frequency, so upscaling them is fine)
    out = _finish(out_lin, p)
    # non-simple cases (Lensbaby / subject DoF / near-occluder layer) render at working res; upscale
    # to the original resolution so the export is full-size...
    if full_wh is not None and (full_wh[0] != w or full_wh[1] != h):
        out = cv2.resize(out, full_wh, interpolation=cv2.INTER_CUBIC)
    # ...then apply grain + geometric optics at that FINAL resolution so grain stays crisp (not a
    # stretched upscale) and the remaps are precise. Grain can confine itself to the defocus.
    return _output_optics(out, p, _defocus_mask(bg_radius, a_sub))

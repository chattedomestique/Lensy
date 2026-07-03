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
    a_sub = np.clip(alpha, 0.0, 1.0).astype(np.float32)
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
    bg_radius = _dilate_coc(focal_radius(bg_signal, focus, metric, p))  # thin features inherit neighbour blur
    # Portrait blur FLOOR. The subject is defined by the MATTE, not by depth — so a *background*
    # object that happens to sit at the subject's focal distance (another person a step behind, a
    # railing) gets CoC≈0 and stays tack-sharp at ANY blur strength, reading as a hard cutout/line
    # the blur slider can't remove. Give the background a minimum blur that ramps up with distance
    # from the subject: the ground right at the feet stays naturally sharp, but anything set apart
    # from the subject always softens. Scales with the blur slider so low K stays subtle.
    if p.k > 0:
        sub_m = (a_sub > 0.5).astype(np.uint8)
        if sub_m.any() and int(sub_m.sum()) < int(0.97 * h * w):
            dist = cv2.distanceTransform(1 - sub_m, cv2.DIST_L2, 5)
            ramp = np.clip((dist / (0.09 * float(np.hypot(h, w))) - 0.15) / 0.85, 0.0, 1.0)
            floor = (p.k / 100.0) * (min(h, w) / 60.0) * ramp  # ~21px far away at k=100 on a 1290px frame
            bg_radius = np.maximum(bg_radius, floor.astype(np.float32))
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
        max_sig = max(1.0, (p.k / 100.0) * float(np.hypot(h, w)) * 0.05)
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

    # glows in linear light, before tonemap, so the spill stays bright
    if p.highlight_boost > 0:  # neutral bloom across the frame
        out_lin = _apply_bloom(out_lin, p.highlight_boost)
    if p.halation > 0:  # warm red-orange glow out of the highlights only
        out_lin = _apply_halation(out_lin, p.halation, p.halation_size)
    out_lin = tonemap_highlights(out_lin)
    out = (np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    # geometric lens optics, last
    if p.ca > 0:  # lateral chromatic aberration — per-channel magnification
        out = _apply_ca(out, p.ca)
    if p.distortion > 0:  # barrel lens distortion
        out = _apply_distortion(out, p.distortion)
    if p.grain > 0:  # modeled film grain — the very last pass, on the final frame
        from .grain import apply_grain

        out = apply_grain(out, p.grain, p.grain_size, p.grain_seed)
    return out

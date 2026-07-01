"""Lensy render pipeline — the product (§7).

The one inviolable ordering (§7.1):

    matte → decontaminate F → remove + inpaint the hole → blur the *clean* background
          → recomposite the sharp subject (premultiplied alpha).

"Blur the whole image then paste the subject back" is FORBIDDEN. This module hard-codes the
correct order; the stages live in sibling modules and each degrades gracefully.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import compose as _compose
from . import depth as _depth
from . import inpaint as _inpaint
from . import matte as _matte
from . import refine as _refine
from .blur import BlurParams, focal_radius, render_lens_blur
from .runtime import ModelBundle

log = logging.getLogger("lensy.pipeline")

# progress callback: (stage_key, human_label, fraction_0_to_1)
Progress = Callable[[str, str, float], None]


@dataclass
class RenderParams:
    k: float = 60.0
    disp_focus: float = 0.7      # focal plane in disparity space; ignored when autofocus is on
    autofocus: bool = True       # lock focus to the subject (median disparity under the matte)
    subject_dof: bool = False    # blur the subject by depth too (cinematic) vs keep it sharp
    blades: int = 0
    rotation: float = 0.0
    highlight_boost: float = 0.18
    cat_eye: float = 0.2
    working_res: int = 2048      # long-edge px the pipeline runs at

    def blur_params(self, disp_focus: float | None = None) -> BlurParams:
        return BlurParams(
            k=self.k,
            disp_focus=self.disp_focus if disp_focus is None else disp_focus,
            subject_dof=self.subject_dof,
            blades=self.blades,
            rotation=self.rotation,
            highlight_boost=self.highlight_boost,
            cat_eye=self.cat_eye,
        )


_STAGES = [
    ("matte", "Separating subject"),
    ("refine", "Snapping the edge"),
    ("decontaminate", "Cleaning edge color"),
    ("depth", "Reading depth"),
    ("inpaint", "Filling the background"),
    ("blur", "Rendering the lens"),
    ("compose", "Compositing"),
]


def _downscale(rgb_u8: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return rgb_u8
    s = long_edge / float(m)
    return cv2.resize(rgb_u8, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def run_pipeline(
    rgb_u8: np.ndarray,
    params: RenderParams,
    bundle: ModelBundle,
    progress: Progress | None = None,
) -> np.ndarray:
    """Run the full pipeline. Input/return: uint8 RGB (HxWx3)."""

    def emit(key: str, label: str, frac: float) -> None:
        if progress:
            try:
                progress(key, label, frac)
            except Exception:  # progress must never break a render
                log.debug("progress callback raised", exc_info=True)

    t0 = time.time()
    work = _downscale(rgb_u8, params.working_res)
    n = len(_STAGES)

    # optional diagnostic capture: set LENSY_DEBUG_DIR to dump input + every stage as images
    dbg = os.environ.get("LENSY_DEBUG_DIR")
    if dbg:
        os.makedirs(dbg, exist_ok=True)

    def dump(name: str, arr: np.ndarray, gray: bool = False) -> None:
        if not dbg:
            return
        try:
            if gray:
                cv2.imwrite(os.path.join(dbg, name), (np.clip(arr, 0, 1) * 255).astype(np.uint8))
            else:
                cv2.imwrite(os.path.join(dbg, name), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        except Exception:
            log.debug("debug dump failed for %s", name, exc_info=True)

    dump("00_input.jpg", work)

    # 1 — matte (soft alpha)
    emit(*_STAGES[0], 0 / n)
    alpha = _matte.estimate_alpha(work, bundle)

    # 2 — refine alpha to true edges
    emit(*_STAGES[1], 1 / n)
    alpha = _refine.refine_alpha(work, alpha)
    dump("01_alpha.png", alpha, gray=True)

    # 3 — decontaminate foreground color (THE anti-halo step)
    emit(*_STAGES[2], 2 / n)
    fg = _refine.decontaminate(work, alpha, bundle)
    dump("02_fg.jpg", (np.clip(fg, 0, 1) * 255).astype(np.uint8))

    # 4 — depth. Keep it in real units (metres for Depth Pro) and build two smoothed fields that
    # DON'T bleed across the silhouette: one for the subject, one for the background (subject hole
    # filled with background depth). The blur radius is the true optical CoC from these distances.
    emit(*_STAGES[3], 3 / n)
    metric = bundle.depth_metric
    signal = _depth.estimate_disparity(work, bundle)
    fg_signal = _depth.smooth_depth(signal)
    bg_signal = _depth.background_depth(signal, (alpha > 0.5).astype(np.uint8))

    # focal distance (in the depth's own units): auto = median depth under the matte.
    if params.autofocus and int((alpha > 0.5).sum()) > 64:
        focus = float(np.median(fg_signal[alpha > 0.5]))
    else:
        lo, hi = float(np.percentile(bg_signal, 2)), float(np.percentile(bg_signal, 98))
        # slider 1.0 = nearest: for metric that's the smallest depth, else the largest disparity
        focus = (hi - params.disp_focus * (hi - lo)) if metric else (lo + params.disp_focus * (hi - lo))
    log.info("focus=%.3f (%s)", focus, "m" if metric else "disp")
    blur_p = params.blur_params(disp_focus=focus)

    radius_fg = focal_radius(fg_signal, focus, metric, blur_p)
    radius_bg = focal_radius(bg_signal, focus, metric, blur_p)
    if dbg:
        rmax = max(float(radius_bg.max()), 1e-3)
        dump("03_radius_bg.png", radius_bg / rmax, gray=True)
        dump("03b_radius_fg.png", radius_fg / rmax, gray=True)

    # 5 — remove subject + inpaint the hole → clean background plate
    emit(*_STAGES[4], 4 / n)
    clean_bg = _inpaint.fill_background(work, alpha, bundle)
    dump("04_clean_bg.jpg", clean_bg)

    # 6 — blur the CLEAN background by its depth-driven CoC (linear-light scatter)
    emit(*_STAGES[5], 5 / n)
    blurred_bg = render_lens_blur(clean_bg, radius_bg, blur_p)
    dump("05_blurred_bg.jpg", blurred_bg)

    # 7 — composite the subject over the blurred background. For cinematic subject-DoF, blur the
    # ORIGINAL subject color (not the decontaminated F, whose bright extrapolated edge would be
    # amplified into a white halo by the blur); the sharp path keeps F for its clean anti-halo edge.
    emit(*_STAGES[6], 6 / n)
    fg_color = (work.astype(np.float32) / 255.0) if params.subject_dof else fg
    out = _compose.compose(fg_color, alpha, blurred_bg, radius_fg, blur_p)
    dump("06_output.jpg", out)

    emit("done", "Done", 1.0)
    log.info("pipeline done in %.2fs (work res %s)", time.time() - t0, work.shape[:2])
    return out


__all__ = ["RenderParams", "run_pipeline", "ModelBundle"]

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
from .blur import BlurParams, render_lens_blur
from .runtime import ModelBundle

log = logging.getLogger("lensy.pipeline")

# progress callback: (stage_key, human_label, fraction_0_to_1)
Progress = Callable[[str, str, float], None]


@dataclass
class RenderParams:
    k: float = 60.0
    disp_focus: float = 0.7      # focal plane in disparity space; ignored when autofocus is on
    autofocus: bool = True       # lock focus to the subject (median disparity under the matte)
    blades: int = 0
    rotation: float = 0.0
    highlight_boost: float = 0.18
    cat_eye: float = 0.2
    working_res: int = 2048      # long-edge px the pipeline runs at

    def blur_params(self, disp_focus: float | None = None) -> BlurParams:
        return BlurParams(
            k=self.k,
            disp_focus=self.disp_focus if disp_focus is None else disp_focus,
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

    # 4 — depth/disparity. Build two smoothed depth fields that DON'T bleed across the silhouette:
    # one for the subject, one for the background (subject hole filled with background depth). This
    # is what lets the whole scene grade continuously without a bright ring around the person.
    emit(*_STAGES[3], 3 / n)
    disparity = _depth.estimate_disparity(work, bundle)
    disp_fg = _depth.smooth_depth(disparity)
    disp_bg = _depth.background_depth(disparity, (alpha > 0.5).astype(np.uint8))
    dump("03_disparity.png", disparity, gray=True)
    dump("03b_disp_bg.png", disp_bg, gray=True)

    # focal plane: lock to the subject (median depth under the matte) so the person is sharp
    # and everything grades away from there — far more accurate than a fixed guess.
    focus = params.disp_focus
    if params.autofocus:
        subj = disp_fg[alpha > 0.5]
        if subj.size > 64:
            focus = float(np.median(subj))
            log.info("autofocus → disp_focus=%.3f (subject)", focus)
    blur_p = params.blur_params(disp_focus=focus)

    # 5 — remove subject + inpaint the hole → clean background plate
    emit(*_STAGES[4], 4 / n)
    clean_bg = _inpaint.fill_background(work, alpha, bundle)
    dump("04_clean_bg.jpg", clean_bg)

    # 6 — blur the CLEAN background by BACKGROUND depth (depth-graded, linear-light scatter)
    emit(*_STAGES[5], 5 / n)
    blurred_bg = render_lens_blur(clean_bg, disp_bg, blur_p)
    dump("05_blurred_bg.jpg", blurred_bg)

    # 7 — composite the subject with its OWN depth-of-field over the blurred background
    emit(*_STAGES[6], 6 / n)
    out = _compose.compose(fg, alpha, blurred_bg, disp_fg, blur_p)
    dump("06_output.jpg", out)

    emit("done", "Done", 1.0)
    log.info("pipeline done in %.2fs (work res %s)", time.time() - t0, work.shape[:2])
    return out


__all__ = ["RenderParams", "run_pipeline", "ModelBundle"]

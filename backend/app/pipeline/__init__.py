"""Lensy render pipeline — the product (§7).

The one inviolable ordering (§7.1):

    matte → decontaminate F → remove + inpaint the hole → blur the *clean* background
          → recomposite the sharp subject (premultiplied alpha).

"Blur the whole image then paste the subject back" is FORBIDDEN. This module hard-codes the
correct order; the stages live in sibling modules and each degrades gracefully.
"""

from __future__ import annotations

import logging
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
    disp_focus: float = 0.7
    blades: int = 0
    rotation: float = 0.0
    highlight_boost: float = 0.6
    cat_eye: float = 0.35
    working_res: int = 2048  # long-edge px the pipeline runs at

    def blur_params(self) -> BlurParams:
        return BlurParams(
            k=self.k,
            disp_focus=self.disp_focus,
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

    # 1 — matte (soft alpha)
    emit(*_STAGES[0], 0 / n)
    alpha = _matte.estimate_alpha(work, bundle)

    # 2 — refine alpha to true edges
    emit(*_STAGES[1], 1 / n)
    alpha = _refine.refine_alpha(work, alpha)

    # 3 — decontaminate foreground color (THE anti-halo step)
    emit(*_STAGES[2], 2 / n)
    fg = _refine.decontaminate(work, alpha, bundle)

    # 4 — depth/disparity
    emit(*_STAGES[3], 3 / n)
    disparity = _depth.estimate_disparity(work, bundle)

    # 5 — remove subject + inpaint the hole → clean background plate
    emit(*_STAGES[4], 4 / n)
    clean_bg = _inpaint.fill_background(work, alpha, bundle)

    # 6 — blur the CLEAN background (depth-graded, linear-light scatter)
    emit(*_STAGES[5], 5 / n)
    blurred_bg = render_lens_blur(clean_bg, disparity, params.blur_params())

    # 7 — recomposite the sharp subject, premultiplied alpha
    emit(*_STAGES[6], 6 / n)
    out = _compose.compose(fg, alpha, blurred_bg, disparity, params.blur_params())

    emit("done", "Done", 1.0)
    log.info("pipeline done in %.2fs (work res %s)", time.time() - t0, work.shape[:2])
    return out


__all__ = ["RenderParams", "run_pipeline", "ModelBundle"]

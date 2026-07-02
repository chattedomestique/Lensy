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

from . import depth as _depth
from . import inpaint as _inpaint
from . import layered as _layered
from . import matte as _matte
from . import refine as _refine
from .blur import BlurParams
from .runtime import ModelBundle

log = logging.getLogger("lensy.pipeline")

# progress callback: (stage_key, human_label, fraction_0_to_1)
Progress = Callable[[str, str, float], None]


@dataclass
class RenderParams:
    k: float = 60.0
    disp_focus: float = 0.7      # focal plane in disparity space; ignored when autofocus is on
    autofocus: bool = True       # lock focus to the subject (median disparity under the matte)
    subject_dof: bool = False    # (cinematic removed) subject is always composited sharp
    blades: int = 0
    rotation: float = 0.0
    highlight_boost: float = 0.18
    cat_eye: float = 0.2
    swirl: float = 0.0           # Petzval swirly bokeh
    sweet: float = 0.0           # Lensbaby sweet-spot blur intensity
    sweet_size: float = 0.35     # sharp sweet-spot radius
    halation: float = 0.0        # film halation glow
    halation_size: float = 0.4
    ca: float = 0.0              # lateral chromatic aberration
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
            swirl=self.swirl,
            sweet=self.sweet,
            sweet_size=self.sweet_size,
            halation=self.halation,
            halation_size=self.halation_size,
            ca=self.ca,
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


def downscale_to_working(rgb_u8: np.ndarray, working_res: int) -> np.ndarray:
    return _downscale(rgb_u8, working_res)


def analyze(rgb_u8: np.ndarray, params: RenderParams, bundle: ModelBundle):
    """Fast first pass for the editor: matte + depth only (~a few seconds), so the depth map shows
    up quickly to edit. Returns (work, alpha, depth_norm); depth_norm is [0,1], NEAR = 1."""
    work = _downscale(rgb_u8, params.working_res)
    alpha = _refine.refine_alpha(work, _matte.estimate_alpha(work, bundle))
    signal = _depth.estimate_disparity(work, bundle)  # near = large (metric metres or [0,1])
    if bundle.depth_metric:  # collapse metric metres → normalized disparity for the editor
        signal = 1.0 / np.clip(signal, 0.08, None)
    depth_norm = _depth.normalize01(signal)  # [0,1], near = 1
    return work, alpha.astype(np.float32), depth_norm.astype(np.float32)


def erase(work: np.ndarray, params: RenderParams, bundle: ModelBundle, mask_u8: np.ndarray):
    """Object removal: fill the masked region (white = erase) plausibly, then re-derive matte +
    depth on the cleaned image (removing something changes the scene). Returns the same tuple as
    analyze(): (cleaned_work, alpha, depth_norm). Caller must drop any cached fg/clean_bg."""
    cleaned = _inpaint.erase_region(work, mask_u8, bundle)
    return analyze(cleaned, params, bundle)


def precompose(work: np.ndarray, alpha: np.ndarray, bundle: ModelBundle):
    """The slow, depth-independent stages: decontaminate F + inpaint the background. Cached after
    the first render so every later slider edit reuses them (render_from is then ~2-3s)."""
    fg = _refine.decontaminate(work, alpha, bundle)
    clean_bg = _inpaint.fill_background(work, alpha, bundle)
    return fg, clean_bg


def render_from(
    work: np.ndarray,
    alpha: np.ndarray,
    fg: np.ndarray,              # cached decontaminated foreground F (from analyze)
    clean_bg: np.ndarray,        # cached inpainted background (from analyze)
    depth_norm: np.ndarray,      # [0,1], near = 1 — the (edited) depth map
    params: RenderParams,
    progress: Progress | None = None,
) -> np.ndarray:
    """Fast render from the cached precompose + a (hand-edited) depth map: just the layered
    occlusion-aware DoF and the lens character. ~2-3s — no matte/depth/decontaminate/inpaint."""

    def emit(key: str, label: str, frac: float) -> None:
        if progress:
            try:
                progress(key, label, frac)
            except Exception:
                log.debug("progress callback raised", exc_info=True)

    t0 = time.time()
    n = len(_STAGES)
    metric = False  # edited depth is normalized disparity, not metric

    emit(*_STAGES[3], 3 / n)
    fg_signal = _depth.smooth_depth(depth_norm)
    bg_signal = _depth.background_depth(depth_norm, (alpha > 0.5).astype(np.uint8))
    if params.autofocus and int((alpha > 0.5).sum()) > 64:
        focus = float(np.median(fg_signal[alpha > 0.5]))
    else:
        # editor path: edited depth is already centered on the subject → use disp_focus directly
        focus = float(np.clip(params.disp_focus, 0.0, 1.0))
    blur_p = params.blur_params(disp_focus=focus)

    emit(*_STAGES[5], 5 / n)
    emit(*_STAGES[6], 6 / n)
    out = _layered.render_layered_dof(fg, alpha, clean_bg, fg_signal, bg_signal, focus, metric, blur_p)

    emit("done", "Done", 1.0)
    log.info("render_from done in %.2fs (work res %s)", time.time() - t0, work.shape[:2])
    return out


def run_pipeline(
    rgb_u8: np.ndarray,
    params: RenderParams,
    bundle: ModelBundle,
    progress: Progress | None = None,
) -> np.ndarray:
    """One-shot: analyze → precompose → render (automatic depth). Return: uint8 RGB."""
    work, alpha, depth_norm = analyze(rgb_u8, params, bundle)
    fg, clean_bg = precompose(work, alpha, bundle)
    return render_from(work, alpha, fg, clean_bg, depth_norm, params, progress)


__all__ = [
    "RenderParams", "run_pipeline", "analyze", "precompose", "render_from",
    "downscale_to_working", "ModelBundle",
]

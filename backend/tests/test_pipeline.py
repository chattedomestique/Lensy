"""Smoke test: the full pipeline must round-trip a synthetic image using only fallbacks
(no model weights required). This is the 'it runs on a fresh machine' guarantee."""

from __future__ import annotations

import numpy as np

from app.pipeline import RenderParams, run_pipeline
from app.pipeline.runtime import ModelBundle


def _synthetic_photo(h: int = 256, w: int = 256) -> np.ndarray:
    """A bright subject blob on a textured background with a few highlights (for bokeh)."""
    rng = np.random.default_rng(0)
    img = (rng.integers(40, 120, size=(h, w, 3))).astype(np.uint8)  # background texture
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - w // 2) ** 2 + (yy - h // 2) ** 2) < (min(h, w) // 4) ** 2
    img[blob] = [220, 180, 150]  # subject
    img[20:26, 30:36] = 255  # a couple of highlights
    img[40:46, 200:206] = 255
    return img


def test_pipeline_runs_with_fallbacks():
    photo = _synthetic_photo()
    bundle = ModelBundle()  # everything None → all classic fallbacks
    stages = []

    def prog(key, label, frac):
        stages.append((key, frac))

    out = run_pipeline(photo, RenderParams(k=70, disp_focus=0.7, blades=6), bundle, prog)

    assert out.shape == photo.shape
    assert out.dtype == np.uint8
    assert {"matte", "blur", "compose", "done"} <= {s for s, _ in stages}
    # output should differ from input (we actually rendered something)
    assert not np.array_equal(out, photo)


def test_disp_focus_changes_output():
    photo = _synthetic_photo()
    bundle = ModelBundle()
    # autofocus would override disp_focus, so disable it to test the manual focal plane
    a = run_pipeline(photo, RenderParams(k=80, disp_focus=0.2, autofocus=False), bundle)
    b = run_pipeline(photo, RenderParams(k=80, disp_focus=0.9, autofocus=False), bundle)
    assert not np.array_equal(a, b)

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
    # render_from emits from decontaminate onward (matte/depth happen in analyze())
    assert {"blur", "compose", "done"} <= {s for s, _ in stages}
    # output should differ from input (we actually rendered something)
    assert not np.array_equal(out, photo)


def test_disp_focus_changes_output():
    photo = _synthetic_photo()
    bundle = ModelBundle()
    # autofocus would override disp_focus, so disable it to test the manual focal plane
    a = run_pipeline(photo, RenderParams(k=80, disp_focus=0.2, autofocus=False), bundle)
    b = run_pipeline(photo, RenderParams(k=80, disp_focus=0.9, autofocus=False), bundle)
    assert not np.array_equal(a, b)


def test_full_resolution_output():
    """A photo larger than the working resolution must round-trip at its NATIVE size — the DoF
    runs at working res but the final composite (and export) is done full-res (§6)."""
    photo = _synthetic_photo(1950, 2600)  # long edge 2600 > working_res 2048
    bundle = ModelBundle()
    out = run_pipeline(photo, RenderParams(k=70, working_res=2048), bundle)
    assert out.shape == photo.shape  # native resolution preserved, not downscaled to 2048
    assert out.dtype == np.uint8


def test_blur_off_is_pristine_full_res():
    """K=0 turns the lens blur fully OFF: with no character effects the output is the untouched
    photo at native resolution (full quality retained — not even an sRGB round-trip)."""
    photo = _synthetic_photo(1500, 2200)
    bundle = ModelBundle()
    # highlight_boost defaults to 0.18; the app sends 0 for "no effects", so set it explicitly
    params = RenderParams(k=0, highlight_boost=0.0, working_res=2048)
    out = run_pipeline(photo, params, bundle)
    assert out.shape == photo.shape
    assert np.array_equal(out, photo)  # byte-identical to the source


def test_blur_off_keeps_depth_effects():
    """With blur OFF, the non-DoF character effects still apply (depth stays available for them)."""
    photo = _synthetic_photo(1500, 2200)
    bundle = ModelBundle()
    plain = run_pipeline(photo, RenderParams(k=0, highlight_boost=0.0, working_res=2048), bundle)
    grainy = run_pipeline(
        photo, RenderParams(k=0, highlight_boost=0.0, grain=0.8, working_res=2048), bundle
    )
    assert grainy.shape == photo.shape
    assert not np.array_equal(grainy, plain)  # grain applied on the sharp full-res frame


def test_max_blur_recalibrated_to_quarter():
    """UI K=100 now reaches a CoC ceiling of 2.75% of the diagonal — a quarter of the old 11%."""
    from app.pipeline.blur import BlurParams, focal_radius

    sig = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    diag = float(np.hypot(256, 256))
    r100 = focal_radius(sig, 0.5, False, BlurParams(k=100))
    assert r100.max() <= 0.0275 * diag + 1e-3          # ceiling is the new (quartered) scale
    assert abs(0.0275 * diag - 0.25 * (0.11 * diag)) < 0.5  # ceiling == 25% of the old ceiling
    # K=0 is exactly zero everywhere — no residual blur floor
    assert float(focal_radius(sig, 0.5, False, BlurParams(k=0)).max()) == 0.0

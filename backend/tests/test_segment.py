"""Subject-selection matte restriction: must work for ordinary objects, not just salient people."""

from __future__ import annotations

import numpy as np

from app.pipeline.segment import restrict_matte


def _sel(h, w, box):
    m = np.zeros((h, w), np.uint8)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = 255
    return m


def test_object_can_become_subject():
    """BiRefNet returns ~0 over a non-salient object; tapping it must still yield a real subject
    matte built from the SAM2 selection (this is the people-only bug)."""
    h, w = 200, 200
    matte = np.zeros((h, w), np.float32)  # salient matte ignores the object
    sel = _sel(h, w, (60, 140, 60, 140))
    out = restrict_matte(matte, sel)
    assert out.max() > 0.9  # the object is covered
    assert (out[70:130, 70:130] > 0.5).mean() > 0.8  # solidly, in its interior
    assert out[10, 10] < 0.05  # and nowhere else


def test_person_matte_unchanged():
    """When the salient matte covers the tapped region (a person), keep BiRefNet's fine alpha —
    the object branch must not alter the people path."""
    h, w = 200, 200
    matte = np.zeros((h, w), np.float32)
    matte[50:150, 70:130] = 1.0  # person body
    sel = _sel(h, w, (50, 150, 70, 130))
    out = restrict_matte(matte, sel)
    # identical to gating BiRefNet's matte to the tapped region (the pre-existing behavior)
    assert out[100, 100] > 0.95
    assert out[10, 10] < 0.05

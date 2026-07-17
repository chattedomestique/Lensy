"""API-level unit tests that don't need model weights: the erase-undo snapshot/restore logic."""

from __future__ import annotations

import numpy as np

from app.api import Analysis, _MAX_UNDO, _pop_undo, _push_undo


def _analysis() -> Analysis:
    z = np.zeros((8, 8), np.float32)
    return Analysis(id="t", work=np.zeros((8, 8, 3), np.uint8), orig=np.ones((16, 16, 3), np.uint8),
                    alpha=z.copy(), depth=z.copy(), matte_full=z.copy())


def test_erase_undo_restores_pre_erase_state():
    a = _analysis()
    work0, orig0, depth0, matte0, alpha0 = a.work, a.orig, a.depth, a.matte_full, a.alpha
    a.fg, a.clean_bg = np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 3), np.uint8)

    _push_undo(a)  # simulate the snapshot the erase handler takes
    # simulate the erase mutating the scene (reassigns, as the real handler does)
    a.work = np.full((8, 8, 3), 5, np.uint8)
    a.orig = None
    a.depth = np.full((8, 8), 0.5, np.float32)
    a.matte_full = np.full((8, 8), 0.5, np.float32)
    a.alpha = np.full((8, 8), 0.5, np.float32)

    _pop_undo(a)
    assert a.work is work0 and a.orig is orig0 and a.depth is depth0
    assert a.matte_full is matte0 and a.alpha is alpha0
    assert a.fg is None and a.clean_bg is None  # caches dropped → recomputed on next render
    assert not a.undo_stack  # stack emptied


def test_undo_stack_is_capped():
    a = _analysis()
    for _ in range(_MAX_UNDO + 5):
        _push_undo(a)
    assert len(a.undo_stack) == _MAX_UNDO

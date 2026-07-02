"""Interactive object selection for the erase / object-removal tool. A tap becomes a precise
object mask via SAM2 (Segment Anything 2.1); without SAM2 we fall back to a GrabCut around a box
or a plain disc around the point (the user's brush can then refine it)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .runtime import ModelBundle

log = logging.getLogger("lensy.segment")


def segment_at(
    rgb_u8: np.ndarray,
    points_xy: list[tuple[float, float]],
    labels: list[int],
    box: tuple[float, float, float, float] | None,
    bundle: ModelBundle,
) -> np.ndarray:
    """Return a uint8 mask (255 = the selected object) for the tap point(s) and/or box.
    points_xy are pixel coords; labels are 1 (include) / 0 (exclude)."""
    if bundle.sam2_model is not None and (points_xy or box is not None):
        try:
            return _sam2_mask(rgb_u8, points_xy, labels, box, bundle)
        except Exception as e:  # noqa: BLE001
            log.warning("SAM2 segment failed (%s); GrabCut fallback", e.__class__.__name__)
    return _grabcut_mask(rgb_u8, points_xy, box)


def _sam2_mask(rgb_u8, points_xy, labels, box, bundle) -> np.ndarray:
    import torch

    model, processor = bundle.sam2_model, bundle.sam2_processor
    kwargs: dict = {"images": rgb_u8, "return_tensors": "pt"}
    if points_xy:
        # nesting: [image][object][point][x,y] ; labels [image][object][label]
        kwargs["input_points"] = [[[[float(x), float(y)] for (x, y) in points_xy]]]
        kwargs["input_labels"] = [[[int(v) for v in labels]]]
    if box is not None:
        kwargs["input_boxes"] = [[[float(v) for v in box]]]  # [image][box][x1,y1,x2,y2]

    inputs = processor(**kwargs).to(bundle.device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)  # 3 candidates + IoU scores
    # post_process_masks → [image] of shape (num_objects, num_masks, H, W); take the first object.
    arr = np.asarray(processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0])
    while arr.ndim > 3:
        arr = arr[0]  # → (num_masks, H, W)
    scores = np.asarray(outputs.iou_scores[0, 0].cpu()).ravel()
    cands = [arr[k].astype(bool) for k in range(arr.shape[0])]

    # For object *removal* a tap should grab the whole object, not a confident sub-part. Prefer the
    # largest candidate that isn't basically the entire frame; fall back to the highest IoU score.
    total = float(cands[0].size)
    frac = [c.sum() / total for c in cands]
    usable = [i for i, f in enumerate(frac) if 0.002 < f < 0.85]
    best = max(usable, key=lambda i: frac[i]) if usable else int(np.argmax(scores))
    return (cands[best].astype(np.uint8) * 255)


def _grabcut_mask(rgb_u8, points_xy, box) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        rect = (max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1))
        try:
            gc = np.zeros((h, w), np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), gc, rect, bgd, fgd, 3,
                        cv2.GC_INIT_WITH_RECT)
            return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:  # noqa: BLE001
            pass
    r = max(8, int(min(h, w) * 0.06))
    for (x, y) in points_xy or []:
        cv2.circle(mask, (int(x), int(y)), r, 255, -1)
    return mask

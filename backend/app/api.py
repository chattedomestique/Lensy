"""HTTP surface for Lensy.

    POST /analyze              → upload a photo → runs matte + depth → {analyze_id, size}
    GET  /analyze/{id}/depth.png   → the (editable) depth map, grayscale (near = white)
    GET  /analyze/{id}/matte.png   → the subject matte, grayscale
    POST /render               → render: either a fresh photo, OR analyze_id + an edited depth map
    GET  /render/{id}/events   → Server-Sent Events: live stage progress
    GET  /render/{id}/result   → the finished image (image/jpeg)
    GET  /healthz              → model-loaded state

Render runs in a worker thread (CPU/MPS heavy → off the event loop). Errors become friendly JSON
envelopes, never a bare 500 with a stack (§4)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image, ImageOps

from .pipeline import (
    RenderParams, analyze, downscale_to_working, erase, precompose, render_from, run_pipeline,
)
from .pipeline.segment import restrict_matte, segment_at

log = logging.getLogger("lensy.api")
router = APIRouter()

_JOBS: dict[str, "Job"] = {}
_ANALYSES: dict[str, "Analysis"] = {}
_MAX_JOBS = 16
_MAX_ANALYSES = 12
_MAX_UNDO = 10  # how many processed erases can be undone per analysis
# The full-res original kept for native-resolution export is bounded on the long edge so a single
# huge upload (a 100 MB JPEG can decode to >1 GB) can't blow up memory (§1: downscale huge inputs).
# 6144 px keeps typical phone/mirrorless shots (12–24 MP) native; override with LENSY_MAX_EXPORT_EDGE.
_MAX_EXPORT_EDGE = int(os.environ.get("LENSY_MAX_EXPORT_EDGE", "6144"))


@dataclass
class Job:
    id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result_jpeg: bytes | None = None
    error: str | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Analysis:
    id: str
    work: np.ndarray       # working-res RGB uint8
    orig: np.ndarray | None  # full-res source RGB uint8 (pre-downscale) → full-quality export (§6)
    alpha: np.ndarray      # matte [0,1]
    depth: np.ndarray      # depth [0,1], near = 1
    fg: np.ndarray | None = None        # decontaminated F — computed lazily on the first render
    clean_bg: np.ndarray | None = None  # inpainted background — the slow step, cached after
    icc: bytes | None = None            # source ICC profile (e.g. Display P3), carried to output
    matte_full: np.ndarray | None = None   # the auto (BiRefNet) matte before subject restriction
    subject_sel: np.ndarray | None = None  # union of tapped SAM2 subject masks (None = auto matte)
    undo_stack: list = field(default_factory=list)  # pre-erase snapshots, for undoing a processed erase
    created: float = field(default_factory=time.time)


def _friendly(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _decode_image(raw: bytes) -> tuple[np.ndarray, bytes | None]:
    """Decode to RGB uint8 and return the source ICC profile (Display P3 for most iPhone photos)
    so we can carry it to the output — otherwise wide-gamut colour is reinterpreted as sRGB and
    the render looks washed-out next to the original. P3 shares sRGB's transfer curve, so the
    linear-light blur math is unaffected; only the profile tag needs to survive."""
    try:
        im = Image.open(io.BytesIO(raw))
        icc = im.info.get("icc_profile")
        img = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(img, dtype=np.uint8), icc
    except Exception as e:
        raise ValueError(f"could not decode image: {e}") from e


def _encode_jpeg(rgb_u8: np.ndarray, quality: int = 95, icc: bytes | None = None) -> bytes:
    """Encode near-lossless: quality 95 with 4:4:4 chroma (subsampling=0) so fine edge/hair detail
    survives, and carry the source ICC profile so wide-gamut colour round-trips (§7 boundary)."""
    buf = io.BytesIO()
    kw = {"icc_profile": icc} if icc else {}
    Image.fromarray(rgb_u8).save(
        buf, format="JPEG", quality=quality, subsampling=0, optimize=True, **kw
    )
    return buf.getvalue()


def _encode_png_gray(f01: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.clip(f01, 0, 1) * 255 + 0.5).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _decode_gray(raw: bytes, size_wh: tuple[int, int]) -> np.ndarray:
    """Decode an uploaded grayscale depth PNG to float [0,1] at the target (w,h)."""
    img = Image.open(io.BytesIO(raw)).convert("L").resize(size_wh, Image.BILINEAR)
    return np.asarray(img, np.float32) / 255.0


def _evict(store: dict, cap: int) -> None:
    while len(store) >= cap:
        store.pop(next(iter(store)), None)


def _push_undo(a: "Analysis") -> None:
    """Snapshot the fields an erase replaces, so the processed erase can be undone. Erase reassigns
    (never mutates in place) these arrays, so storing references — not copies — is safe and cheap.
    fg/clean_bg are caches and are recomputed on the next render, so they're left out of the snapshot."""
    a.undo_stack.append(
        {"work": a.work, "orig": a.orig, "depth": a.depth,
         "matte_full": a.matte_full, "alpha": a.alpha, "subject_sel": a.subject_sel}
    )
    if len(a.undo_stack) > _MAX_UNDO:
        a.undo_stack.pop(0)


def _pop_undo(a: "Analysis") -> None:
    """Restore the most recent pre-erase snapshot and drop the precompose cache (it recomputes)."""
    snap = a.undo_stack.pop()
    a.work, a.orig, a.depth = snap["work"], snap["orig"], snap["depth"]
    a.matte_full, a.alpha, a.subject_sel = snap["matte_full"], snap["alpha"], snap["subject_sel"]
    a.fg = a.clean_bg = None


@router.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading.")
    return JSONResponse({"status": "ok", "models": bundle.status()})


# ----------------------------- analyze (matte + depth) -----------------------------


@router.post("/analyze")
async def analyze_photo(
    request: Request,
    photo: UploadFile = File(...),
    working_res: int = Form(2048),
) -> JSONResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading — try again in a moment.")
    raw = await photo.read()
    if not raw:
        return _friendly(400, "empty_upload", "No image data received.")
    if len(raw) > 100 * 1024 * 1024:
        return _friendly(413, "too_large", "Image exceeds the 100 MB limit.")
    try:
        rgb, icc = _decode_image(raw)
    except ValueError as e:
        return _friendly(400, "bad_image", str(e))

    params = RenderParams(working_res=int(np.clip(working_res, 512, 4096)))
    loop = asyncio.get_running_loop()
    try:
        work, alpha, depth = await loop.run_in_executor(None, analyze, rgb, params, bundle)
    except Exception as e:  # noqa: BLE001
        log.exception("analyze failed")
        return _friendly(500, "analyze_failed", f"{e.__class__.__name__}: {e}")

    _evict(_ANALYSES, _MAX_ANALYSES)
    aid = uuid.uuid4().hex
    # keep the (bounded) full-res original so the final composite (and export) runs at native
    # resolution; None when the photo is already within the working-res budget (nothing to gain).
    orig = downscale_to_working(rgb, _MAX_EXPORT_EDGE)
    orig = orig if orig.shape[:2] != work.shape[:2] else None
    _ANALYSES[aid] = Analysis(
        id=aid, work=work, orig=orig, alpha=alpha, depth=depth, icc=icc, matte_full=alpha.copy()
    )
    h, w = work.shape[:2]
    return JSONResponse({"analyze_id": aid, "width": w, "height": h})


@router.get("/analyze/{aid}/depth.png")
async def analyze_depth(aid: str) -> Response:
    a = _ANALYSES.get(aid)
    if a is None:
        raise HTTPException(status_code=404, detail="unknown analysis")
    return Response(content=_encode_png_gray(a.depth), media_type="image/png")


@router.get("/analyze/{aid}/matte.png")
async def analyze_matte(aid: str) -> Response:
    a = _ANALYSES.get(aid)
    if a is None:
        raise HTTPException(status_code=404, detail="unknown analysis")
    return Response(content=_encode_png_gray(a.alpha), media_type="image/png")


@router.get("/analyze/{aid}/photo.jpg")
async def analyze_photo_img(aid: str) -> Response:
    a = _ANALYSES.get(aid)
    if a is None:
        raise HTTPException(status_code=404, detail="unknown analysis")
    return Response(content=_encode_jpeg(a.work, icc=a.icc), media_type="image/jpeg")


# ------------------------------ subject (tap to select) ----------------------------


@router.post("/subject")
async def select_subject(
    request: Request,
    analyze_id: str = Form(...),
    points: str = Form("[]"),   # JSON [[nx, ny, label], …]; each tap adds a person to the subject
    reset: bool = Form(False),  # clear back to the automatic (whole-scene) matte
) -> JSONResponse:
    """Restrict the subject to who you tap: SAM2-select that person and keep the soft matte only
    there, so other salient people (at a different depth) fall back to the blurred background and
    the focal plane locks to the tapped subject. Tap again to add same-plane subjects."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading — try again in a moment.")
    a = _ANALYSES.get(analyze_id)
    if a is None:
        return _friendly(404, "unknown_analysis", "That analysis expired — re-add the photo.")
    h, w = a.work.shape[:2]
    if a.matte_full is None:
        a.matte_full = a.alpha.copy()

    if reset:
        a.subject_sel = None
        a.alpha = a.matte_full.copy()
        a.fg = a.clean_bg = None
        return JSONResponse({"ok": True, "width": w, "height": h})

    try:
        pts = json.loads(points) or []
        points_xy = [(float(p[0]) * w, float(p[1]) * h) for p in pts]
        labels = [int(p[2]) if len(p) > 2 else 1 for p in pts]
    except Exception as e:  # noqa: BLE001
        return _friendly(400, "bad_prompt", f"could not read the tap: {e}")
    if not points_xy:
        return _friendly(400, "no_point", "Tap a subject to select.")

    loop = asyncio.get_running_loop()
    sel = await loop.run_in_executor(None, segment_at, a.work, points_xy, labels, None, bundle)
    a.subject_sel = sel if a.subject_sel is None else np.maximum(a.subject_sel, sel)
    a.alpha = restrict_matte(a.matte_full, a.subject_sel)
    a.fg = a.clean_bg = None  # matte changed → decontaminate + inpaint must recompute
    return JSONResponse({"ok": True, "width": w, "height": h})


# ------------------------------ erase (object removal) -----------------------------


@router.post("/segment")
async def segment(
    request: Request,
    analyze_id: str = Form(...),
    points: str = Form("[]"),   # JSON [[nx, ny, label], …] normalized 0..1; label 1=include 0=exclude
    box: str = Form(""),        # JSON [nx1, ny1, nx2, ny2] normalized (optional)
) -> Response:
    """Tap-to-select: return a mask (PNG, white = object) for the tapped point via SAM2."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading — try again in a moment.")
    a = _ANALYSES.get(analyze_id)
    if a is None:
        return _friendly(404, "unknown_analysis", "That analysis expired — re-add the photo.")
    h, w = a.work.shape[:2]
    try:
        pts = json.loads(points) or []
        points_xy = [(float(p[0]) * w, float(p[1]) * h) for p in pts]
        labels = [int(p[2]) if len(p) > 2 else 1 for p in pts]
        bx = None
        if box:
            b = json.loads(box)
            bx = (float(b[0]) * w, float(b[1]) * h, float(b[2]) * w, float(b[3]) * h)
    except Exception as e:  # noqa: BLE001
        return _friendly(400, "bad_prompt", f"could not read selection: {e}")

    loop = asyncio.get_running_loop()
    mask = await loop.run_in_executor(None, segment_at, a.work, points_xy, labels, bx, bundle)
    return Response(content=_encode_png_gray(mask.astype(np.float32) / 255.0), media_type="image/png")


@router.post("/erase")
async def erase_object(
    request: Request,
    analyze_id: str = Form(...),
    mask: UploadFile = File(...),  # grayscale PNG, white = erase
    layer: str = Form("auto"),     # "auto" | "subject" | "background" — where the erase may act
) -> JSONResponse:
    """Fill the masked region plausibly (LaMa) and re-derive matte + depth on the cleaned image.
    The analysis is updated in place; the client then reloads depth/matte/photo and re-renders."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading — try again in a moment.")
    a = _ANALYSES.get(analyze_id)
    if a is None:
        return _friendly(404, "unknown_analysis", "That analysis expired — re-add the photo.")
    raw = await mask.read()
    if not raw:
        return _friendly(400, "empty_mask", "No mask received.")
    h, w = a.work.shape[:2]
    try:
        m01 = _decode_gray(raw, (w, h))
    except Exception as e:  # noqa: BLE001
        return _friendly(400, "bad_mask", f"could not read mask: {e}")
    if float(m01.max()) < 0.5:
        return _friendly(400, "empty_mask", "Nothing was selected to erase.")

    mask_u8 = (np.clip(m01, 0, 1) * 255).astype(np.uint8)
    # layer toggle: keep the erase on one side of the matte so it can't bleed across — e.g. erase a
    # wire behind a head (background) without touching the head, or a blemish on skin (subject)
    # without pulling in background.
    if layer in ("subject", "background") and a.alpha is not None:
        sub = a.alpha > 0.5
        keep = sub if layer == "subject" else ~sub
        mask_u8 = np.where(keep, mask_u8, np.uint8(0))
        if int((mask_u8 > 127).sum()) == 0:
            return _friendly(400, "empty_mask", f"Nothing on the {layer} to erase there.")

    params = RenderParams(working_res=max(h, w))  # already at working res — don't downscale
    loop = asyncio.get_running_loop()
    try:
        cleaned, alpha, depth = await loop.run_in_executor(None, erase, a.work, params, bundle, mask_u8)
    except Exception as e:  # noqa: BLE001
        log.exception("erase failed")
        return _friendly(500, "erase_failed", f"{e.__class__.__name__}: {e}")
    _push_undo(a)  # snapshot the pre-erase scene so this processed erase can be undone
    a.work, a.depth = cleaned, depth
    a.matte_full = alpha
    a.alpha = restrict_matte(alpha, a.subject_sel) if a.subject_sel is not None else alpha
    a.fg = None  # invalidate the precompose cache — the scene changed
    a.clean_bg = None
    a.orig = None  # the full-res original no longer matches the erased working image → render at work res
    return JSONResponse({"ok": True, "width": w, "height": h, "can_undo": True})


@router.post("/undo")
async def undo_erase(
    request: Request,
    analyze_id: str = Form(...),
) -> JSONResponse:
    """Undo the most recent processed erase: restore the pre-erase scene (pixels, depth, matte, and
    the full-res original). The client then reloads depth/matte/photo and re-renders."""
    a = _ANALYSES.get(analyze_id)
    if a is None:
        return _friendly(404, "unknown_analysis", "That analysis expired — re-add the photo.")
    if not a.undo_stack:
        return _friendly(400, "nothing_to_undo", "Nothing to undo.")
    _pop_undo(a)
    h, w = a.work.shape[:2]
    return JSONResponse({"ok": True, "width": w, "height": h, "can_undo": bool(a.undo_stack)})


# ----------------------------------- render ----------------------------------------


def _params_from_form(
    k, disp_focus, autofocus, subject_dof, blades, rotation, highlight_boost, cat_eye,
    swirl, sweet, sweet_size, halation, halation_size, ca, distortion, grain, grain_size,
    grain_blend, working_res,
) -> RenderParams:
    return RenderParams(
        k=float(np.clip(k, 0, 100)),
        disp_focus=float(np.clip(disp_focus, 0, 1)),
        autofocus=bool(autofocus),
        subject_dof=bool(subject_dof),
        blades=int(blades),
        rotation=float(rotation),
        highlight_boost=float(np.clip(highlight_boost, 0, 2)),
        cat_eye=float(np.clip(cat_eye, 0, 1)),
        swirl=float(np.clip(swirl, 0, 1)),
        sweet=float(np.clip(sweet, 0, 1)),
        sweet_size=float(np.clip(sweet_size, 0.05, 1)),
        halation=float(np.clip(halation, 0, 1)),
        halation_size=float(np.clip(halation_size, 0.05, 1)),
        ca=float(np.clip(ca, 0, 1)),
        distortion=float(np.clip(distortion, 0, 1)),
        grain=float(np.clip(grain, 0, 1)),
        grain_size=float(np.clip(grain_size, 0, 1)),
        grain_blend=float(np.clip(grain_blend, 0, 1)),
        working_res=int(np.clip(working_res, 512, 4096)),
    )


def _spawn_render(request: Request, fn, *args, icc: bytes | None = None) -> JSONResponse:
    """Run a render callable (fn(*args, progress) -> rgb) as a background job with SSE progress.
    The output JPEG is tagged with `icc` so wide-gamut colour survives the round-trip."""
    if len(_JOBS) >= _MAX_JOBS:
        for jid in list(_JOBS)[: len(_JOBS) - _MAX_JOBS + 1]:
            _JOBS.pop(jid, None)
    job = Job(id=uuid.uuid4().hex)
    _JOBS[job.id] = job
    loop = asyncio.get_running_loop()

    def progress_cb(key: str, label: str, frac: float) -> None:
        loop.call_soon_threadsafe(
            job.queue.put_nowait, {"stage": key, "label": label, "progress": round(frac, 3)}
        )

    async def worker() -> None:
        try:
            out = await loop.run_in_executor(None, fn, *args, progress_cb)
            job.result_jpeg = await loop.run_in_executor(None, lambda: _encode_jpeg(out, icc=icc))
        except Exception as e:  # noqa: BLE001
            log.exception("render failed")
            job.error = f"{e.__class__.__name__}: {e}"
            loop.call_soon_threadsafe(
                job.queue.put_nowait, {"stage": "error", "label": "Render failed", "error": job.error}
            )
        finally:
            loop.call_soon_threadsafe(job.done.set)

    asyncio.create_task(worker())
    return JSONResponse({"job_id": job.id})


@router.post("/render")
async def start_render(
    request: Request,
    photo: UploadFile | None = File(None),
    depth: UploadFile | None = File(None),       # edited depth map (grayscale, near = white)
    analyze_id: str = Form(""),
    k: float = Form(60.0),
    disp_focus: float = Form(0.7),
    autofocus: bool = Form(True),
    subject_dof: bool = Form(False),
    blades: int = Form(0),
    rotation: float = Form(0.0),
    highlight_boost: float = Form(0.18),
    cat_eye: float = Form(0.2),
    swirl: float = Form(0.0),
    sweet: float = Form(0.0),
    sweet_size: float = Form(0.35),
    halation: float = Form(0.0),
    halation_size: float = Form(0.4),
    ca: float = Form(0.0),
    distortion: float = Form(0.0),
    grain: float = Form(0.0),
    grain_size: float = Form(0.4),
    grain_blend: float = Form(0.0),
    working_res: int = Form(2048),
) -> JSONResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading — try again in a moment.")
    params = _params_from_form(
        k, disp_focus, autofocus, subject_dof, blades, rotation, highlight_boost, cat_eye,
        swirl, sweet, sweet_size, halation, halation_size, ca, distortion, grain, grain_size,
        grain_blend, working_res,
    )
    if analyze_id:  # grain is static per image: seed from the analysis id, not per render
        params.grain_seed = (int(analyze_id[:8], 16) % 9973) / 9973.0

    # Path A — render from a prior analysis + (optionally) a hand-edited depth map
    if analyze_id:
        a = _ANALYSES.get(analyze_id)
        if a is None:
            return _friendly(404, "unknown_analysis", "That depth analysis expired — regenerate it.")
        depth_map = a.depth
        if depth is not None:
            raw = await depth.read()
            if raw:
                try:
                    h, w = a.work.shape[:2]
                    depth_map = _decode_gray(raw, (w, h))
                except Exception as e:  # noqa: BLE001
                    return _friendly(400, "bad_depth", f"could not read edited depth: {e}")

        def do_render(progress):
            if a.fg is None:  # lazy precompose on the first render, cached for later edits
                a.fg, a.clean_bg = precompose(a.work, a.alpha, bundle)
            return render_from(
                a.work, a.alpha, a.fg, a.clean_bg, depth_map, params, progress, orig=a.orig
            )

        return _spawn_render(request, do_render, icc=a.icc)

    # Path B — one-shot from a photo (automatic depth)
    if photo is None:
        return _friendly(400, "no_input", "Provide a photo, or an analyze_id.")
    raw = await photo.read()
    if not raw:
        return _friendly(400, "empty_upload", "No image data received.")
    if len(raw) > 100 * 1024 * 1024:
        return _friendly(413, "too_large", "Image exceeds the 100 MB limit.")
    try:
        rgb, icc = _decode_image(raw)
    except ValueError as e:
        return _friendly(400, "bad_image", str(e))
    return _spawn_render(request, run_pipeline, rgb, params, bundle, icc=icc)


@router.get("/render/{job_id}/events")
async def render_events(job_id: str) -> StreamingResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    async def event_stream():
        while True:
            try:
                msg = await asyncio.wait_for(job.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if job.done.is_set() and job.queue.empty():
                    break
                yield ": keep-alive\n\n"
                continue
            if msg.get("stage") == "error":
                yield f"event: error\ndata: {json.dumps(msg)}\n\n"
                return
            yield f"event: progress\ndata: {json.dumps(msg)}\n\n"
            if msg.get("stage") == "done":
                break
        payload = {"result_url": f"/render/{job_id}/result"} if job.error is None else {"error": job.error}
        ev = "done" if job.error is None else "error"
        yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/render/{job_id}/result")
async def render_result(job_id: str) -> Response:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.error:
        return _friendly(500, "render_failed", job.error)
    if job.result_jpeg is None:
        return _friendly(409, "not_ready", "Render still in progress.")
    return Response(content=job.result_jpeg, media_type="image/jpeg")

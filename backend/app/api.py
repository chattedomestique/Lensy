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
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image, ImageOps

from .pipeline import RenderParams, analyze, precompose, render_from, run_pipeline

log = logging.getLogger("lensy.api")
router = APIRouter()

_JOBS: dict[str, "Job"] = {}
_ANALYSES: dict[str, "Analysis"] = {}
_MAX_JOBS = 16
_MAX_ANALYSES = 12


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
    alpha: np.ndarray      # matte [0,1]
    depth: np.ndarray      # depth [0,1], near = 1
    fg: np.ndarray | None = None        # decontaminated F — computed lazily on the first render
    clean_bg: np.ndarray | None = None  # inpainted background — the slow step, cached after
    created: float = field(default_factory=time.time)


def _friendly(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _decode_image(raw: bytes) -> np.ndarray:
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)).convert("RGB"))
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"could not decode image: {e}") from e


def _encode_jpeg(rgb_u8: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="JPEG", quality=quality, subsampling=0)
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
        rgb = _decode_image(raw)
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
    _ANALYSES[aid] = Analysis(id=aid, work=work, alpha=alpha, depth=depth)
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
    return Response(content=_encode_jpeg(a.work), media_type="image/jpeg")


# ----------------------------------- render ----------------------------------------


def _params_from_form(
    k, disp_focus, autofocus, subject_dof, blades, rotation, highlight_boost, cat_eye,
    swirl, sweet, sweet_size, halation, halation_size, ca, working_res,
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
        working_res=int(np.clip(working_res, 512, 4096)),
    )


def _spawn_render(request: Request, fn, *args) -> JSONResponse:
    """Run a render callable (fn(*args, progress) -> rgb) as a background job with SSE progress."""
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
            job.result_jpeg = await loop.run_in_executor(None, _encode_jpeg, out)
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
    working_res: int = Form(2048),
) -> JSONResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading — try again in a moment.")
    params = _params_from_form(
        k, disp_focus, autofocus, subject_dof, blades, rotation, highlight_boost, cat_eye,
        swirl, sweet, sweet_size, halation, halation_size, ca, working_res,
    )

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
            return render_from(a.work, a.alpha, a.fg, a.clean_bg, depth_map, params, progress)

        return _spawn_render(request, do_render)

    # Path B — one-shot from a photo (automatic depth)
    if photo is None:
        return _friendly(400, "no_input", "Provide a photo, or an analyze_id.")
    raw = await photo.read()
    if not raw:
        return _friendly(400, "empty_upload", "No image data received.")
    if len(raw) > 100 * 1024 * 1024:
        return _friendly(413, "too_large", "Image exceeds the 100 MB limit.")
    try:
        rgb = _decode_image(raw)
    except ValueError as e:
        return _friendly(400, "bad_image", str(e))
    return _spawn_render(request, run_pipeline, rgb, params, bundle)


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

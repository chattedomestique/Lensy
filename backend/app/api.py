"""HTTP surface for Lensy.

    POST /render               → upload a photo + lens params, start a render job → {job_id}
    GET  /render/{id}/events   → Server-Sent Events: live stage progress (no polling)
    GET  /render/{id}/result   → the finished image (image/jpeg)
    GET  /healthz              → model-loaded state

Render runs in a worker thread (CPU/MPS heavy → off the event loop). Progress is pushed from
that thread into an asyncio queue and streamed over SSE. Errors become friendly JSON envelopes,
never a bare 500 with a stack (§4)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from dataclasses import dataclass, field

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image

from .pipeline import RenderParams, run_pipeline

log = logging.getLogger("lensy.api")
router = APIRouter()

# in-memory job store (single-user personal tool; no DB needed)
_JOBS: dict[str, "Job"] = {}
_MAX_JOBS = 16


@dataclass
class Job:
    id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result_jpeg: bytes | None = None
    error: str | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


def _friendly(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _decode_image(raw: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        # respect EXIF orientation so portraits aren't sideways
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"could not decode image: {e}") from e


def _encode_jpeg(rgb_u8: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


@router.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return _friendly(503, "warming", "Models are still loading.")
    return JSONResponse({"status": "ok", "models": bundle.status()})


@router.post("/render")
async def start_render(
    request: Request,
    photo: UploadFile = File(...),
    k: float = Form(60.0),
    disp_focus: float = Form(0.7),
    blades: int = Form(0),
    rotation: float = Form(0.0),
    highlight_boost: float = Form(0.6),
    cat_eye: float = Form(0.35),
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

    params = RenderParams(
        k=float(np.clip(k, 0, 100)),
        disp_focus=float(np.clip(disp_focus, 0, 1)),
        blades=int(blades),
        rotation=float(rotation),
        highlight_boost=float(np.clip(highlight_boost, 0, 2)),
        cat_eye=float(np.clip(cat_eye, 0, 1)),
        working_res=int(np.clip(working_res, 512, 4096)),
    )

    # evict old jobs if needed
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
            out = await loop.run_in_executor(
                None, run_pipeline, rgb, params, bundle, progress_cb
            )
            job.result_jpeg = await loop.run_in_executor(None, _encode_jpeg, out)
        except Exception as e:  # surface friendly, log detail
            log.exception("render failed")
            job.error = f"{e.__class__.__name__}: {e}"
            loop.call_soon_threadsafe(
                job.queue.put_nowait, {"stage": "error", "label": "Render failed", "error": job.error}
            )
        finally:
            loop.call_soon_threadsafe(job.done.set)

    asyncio.create_task(worker())
    return JSONResponse({"job_id": job.id})


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
                yield ": keep-alive\n\n"  # comment ping holds the connection
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

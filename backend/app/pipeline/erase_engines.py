"""Object-removal engines for the erase tool — ported from the Vanish `studio_engine.py`, adapted
to Lensy's numpy pipeline with real progress reporting.

  lama        : big-lama (Lensy's bundle) — instant, never hallucinates. The default.
  objectclear : ObjectClear pipeline (diffusers) — removes the object **and its shadow/reflection**.
  flux        : Flux.1 Fill via a ComfyUI child process — max-fidelity fill on hard backgrounds.

The two heavy engines lazy-load on first use and reuse an existing on-disk install (the Vanish
`~/object-removal-studio/` layout, all paths overridable via env), so the 7–23 GB of weights are
never re-downloaded. GPU work is serialized (`_infer_lock`). Every engine reports progress in
[0,1] so the UI shows a real percentage during the long runs (ObjectClear ~minutes, Flux 10–15 min
on 16 GB with `--lowvram`). A missing dependency / model / path degrades gracefully: the engine
raises `EngineUnavailable` and the caller falls back to LaMa.

Ported faithfully from the user's working engine — the model calls (ObjectClear's
`from_pretrained_with_custom_modules` + `resize_by_short_side`, the ComfyUI Fill workflow packing)
are unchanged; only the I/O boundary (numpy ↔ PIL) and the progress hooks are new.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from .runtime import ModelBundle

log = logging.getLogger("lensy.erase_engines")

# progress callback: (fraction_0_to_1 | None, human_label). None fraction = indeterminate.
Progress = Callable[[float | None, str], None]

ENGINE_NAMES = ("lama", "objectclear", "flux")


class EngineUnavailable(RuntimeError):
    """Raised when an engine's deps / models / paths aren't present, so the caller can fall back."""


# ---- on-disk install layout (Vanish's, all overridable) ------------------------------------------
_STUDIO = Path(os.environ.get("VANISH_STUDIO", "~/object-removal-studio")).expanduser()
_OC_REPO = Path(os.environ.get("VANISH_OC_REPO", str(_STUDIO / "ObjectClear")))
_COMFY_DIR = Path(os.environ.get("VANISH_COMFY_DIR", str(_STUDIO / "ComfyUI")))
_COMFY_PY = os.environ.get("VANISH_COMFY_PY", str(_COMFY_DIR / "venv" / "bin" / "python"))
_COMFY_URL = os.environ.get("VANISH_COMFY_URL", "http://127.0.0.1:8188")
_WF_PATH = Path(os.environ.get("VANISH_FLUX_WORKFLOW", str(_STUDIO / "workflows" / "Flux-Fill-Object-Removal.api.json")))
# low-mem by default on 16 GB (the user's target) — offloads to CPU RAM and streams.
_COMFY_ARGS = os.environ.get("VANISH_COMFY_ARGS", "--lowvram").split()
_COMFY_START_TIMEOUT = int(os.environ.get("VANISH_COMFY_START_TIMEOUT", "300"))
_FLUX_TIMEOUT = int(os.environ.get("VANISH_FLUX_TIMEOUT", "1800"))  # 30 min ceiling for a lowvram run
# VANISH_LOW_MEM: on a memory-tight machine (the 16 GB mini) the two heavy engines can't be resident
# together, so loading one first unloads the other and frees the MPS cache — only one big model at a
# time. Off by default (roomier machines keep both warm for instant switching); set =1 on the mini.
_LOW_MEM = os.environ.get("VANISH_LOW_MEM", "").strip().lower() in ("1", "true", "yes", "on")

# VANISH_ENGINES: which removal engines this install offers, comma-separated. Lets a machine WITHOUT
# ComfyUI drop Flux ("lama,objectclear") so the picker never even shows Reconstruct. LaMa is always
# available (it's the fallback). Default: all three.
_ENABLED = [e.strip() for e in os.environ.get("VANISH_ENGINES", "lama,objectclear,flux").split(",")
            if e.strip() in ENGINE_NAMES]
if "lama" not in _ENABLED:
    _ENABLED = ["lama", *_ENABLED]

# lazy singletons + serialization
_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_engines: dict[str, object] = {}
_status: dict[str, str] = {"lama": "idle", "objectclear": "idle", "flux": "idle"}
_errors: dict[str, str] = {}  # last load/run error per engine (surfaced by /engines for diagnosis)
_comfy_proc: subprocess.Popen | None = None


def engine_status() -> dict[str, str]:
    """Per-engine load state: idle | loading | ready | error (LaMa reflects the bundle)."""
    return dict(_status)


def enabled_engines() -> list[str]:
    """Which engines this install offers (VANISH_ENGINES); the picker shows only these."""
    return list(_ENABLED)


def engine_errors() -> dict[str, str]:
    """Last error per engine, for diagnosing a failed heavy engine from /engines."""
    return dict(_errors)


def set_lama_status(ready: bool) -> None:
    _status["lama"] = "ready" if ready else "idle"


# ---- numpy ↔ PIL helpers (mirror studio_engine) --------------------------------------------------


def _to_pil(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    return Image.fromarray(arr).convert(mode)


def _fit(img: Image.Image, max_side: int, mult: int = 1) -> Image.Image:
    """Downscale so the longer side <= max_side; optionally snap dims to a multiple (Flux/Metal)."""
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    nw, nh = int(round(w * s)), int(round(h * s))
    if mult > 1:
        nw = max(mult, nw - nw % mult)
        nh = max(mult, nh - nh % mult)
    return img.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else img


def _composite_back(result: Image.Image, orig: Image.Image, mask: Image.Image,
                    feather: int = 6, dilate: int = 9) -> np.ndarray:
    """Paste the model result into the original ONLY within the (feathered) mask, so unedited areas
    keep full resolution and the edge blends. Returns uint8 RGB ndarray."""
    ow, oh = orig.size
    r = result.resize((ow, oh), Image.LANCZOS) if result.size != (ow, oh) else result
    m = mask.convert("L").resize((ow, oh), Image.NEAREST)
    if dilate > 1:
        m = m.filter(ImageFilter.MaxFilter(dilate if dilate % 2 else dilate + 1))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(Image.composite(r.convert("RGB"), orig.convert("RGB"), m), dtype=np.uint8)


# ---- LaMa (routes to Lensy's existing, already-improved erase_region) ----------------------------


def _run_lama(rgb_u8: np.ndarray, mask_u8: np.ndarray, params: dict, progress: Progress,
              bundle: ModelBundle) -> np.ndarray:
    from . import inpaint as _inpaint

    progress(0.15, "Quick Erase")
    out = _inpaint.erase_region(rgb_u8, mask_u8, bundle)  # 2048-cap LaMa + feathered composite
    progress(1.0, "Quick Erase")
    return out


# ---- low-memory engine juggling (VANISH_LOW_MEM) -------------------------------------------------


def _free_mps() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _unload_engine(name: str) -> None:
    """Drop a heavy engine and free its memory. Called under `_load_lock`."""
    if name == "objectclear":
        pipe = _engines.pop("objectclear", None)
        if pipe is not None:
            try:
                pipe.to("cpu")  # release MPS allocations before dropping the reference
            except Exception:  # noqa: BLE001
                pass
            del pipe
            _status["objectclear"] = "idle"
            import gc

            gc.collect()
            _free_mps()
            log.info("VANISH_LOW_MEM: unloaded ObjectClear")
    elif name == "flux" and _comfy_proc is not None:
        shutdown()  # terminate the ComfyUI child (frees its process + weights)
        _status["flux"] = "idle"
        log.info("VANISH_LOW_MEM: stopped ComfyUI (Flux)")


def _free_for(target: str) -> None:
    """VANISH_LOW_MEM: before loading `target`, unload every OTHER heavy engine so only one big
    model is resident at once. Also frees the MPS cache. No-op unless VANISH_LOW_MEM is set."""
    if not _LOW_MEM:
        return
    for name in ("objectclear", "flux"):
        if name != target:
            _unload_engine(name)
    _free_mps()


# ---- ObjectClear (diffusers; removes object + shadow/reflection) ---------------------------------


def _get_objectclear():
    with _load_lock:
        if "objectclear" in _engines:
            return _engines["objectclear"]
        _free_for("objectclear")  # low-mem: free Flux (and the MPS cache) before loading ObjectClear
        _status["objectclear"] = "loading"
        try:
            import sys

            if str(_OC_REPO) not in sys.path:
                sys.path.insert(0, str(_OC_REPO))
            import torch
            from objectclear.pipelines import ObjectClearPipeline
        except Exception as e:  # noqa: BLE001
            msg = (f"ObjectClear deps not available ({e.__class__.__name__}: {e}). Expected the "
                   f"`objectclear` package under {_OC_REPO} (set VANISH_OC_REPO) and `diffusers` installed.")
            _status["objectclear"] = "error"
            _errors["objectclear"] = msg
            raise EngineUnavailable(msg) from e
        try:
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
                "jixin0101/ObjectClear", torch_dtype=torch.float16,
                variant="fp16", apply_attention_guided_fusion=True,
            )
            pipe.to(dev)
        except Exception as e:  # noqa: BLE001
            msg = f"ObjectClear failed to load ({e.__class__.__name__}: {e})"
            _status["objectclear"] = "error"
            _errors["objectclear"] = msg
            raise EngineUnavailable(msg) from e
        _engines["objectclear"] = pipe
        _status["objectclear"] = "ready"
        return pipe


def _run_objectclear(rgb_u8: np.ndarray, mask_u8: np.ndarray, params: dict, progress: Progress,
                     bundle: ModelBundle) -> np.ndarray:
    import torch

    pipe = _get_objectclear()
    try:
        from objectclear.utils import resize_by_short_side
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"objectclear.utils missing ({e.__class__.__name__})") from e

    steps = int(params.get("steps", 20))
    gs = float(params.get("guidance_scale", 2.5))
    seed = int(params.get("seed", 42))
    orig = _to_pil(rgb_u8, "RGB")
    osize = orig.size
    img = resize_by_short_side(orig, 512, resample=Image.BICUBIC)
    msk = resize_by_short_side(_to_pil(mask_u8, "L"), 512, resample=Image.NEAREST)
    w, h = img.size
    gen = torch.Generator(device="cpu").manual_seed(seed)  # MPS has no Generator

    # per-step progress via a diffusers step callback (falls back gracefully if the custom pipeline
    # doesn't accept one). Reserve 0..0.95 for sampling; the resize tail finishes it.
    def _cb(_pipe, step, _t, cbk):
        progress(0.05 + 0.9 * (step + 1) / max(steps, 1), "Deep Clean")
        return cbk

    progress(0.02, "Deep Clean")
    kw = dict(prompt="remove the instance of object", image=img, mask_image=msk, generator=gen,
              num_inference_steps=steps, guidance_scale=gs, height=h, width=w, return_attn_map=False)
    with _infer_lock:
        try:
            out = pipe(**kw, callback_on_step_end=_cb).images[0]
        except TypeError:  # pipeline predates callback_on_step_end → run without per-step progress
            progress(None, "Deep Clean")
            out = pipe(**kw).images[0]
    progress(0.98, "Deep Clean")
    return np.asarray(out.resize(osize).convert("RGB"), dtype=np.uint8)


# ---- Flux Fill (via a ComfyUI child process) -----------------------------------------------------


def _comfy_up() -> bool:
    try:
        urllib.request.urlopen(_COMFY_URL + "/system_stats", timeout=2)
        return True
    except Exception:
        return False


def _ensure_comfy() -> None:
    global _comfy_proc
    if _comfy_up():
        _status["flux"] = "ready"
        return
    with _load_lock:
        if _comfy_up():
            _status["flux"] = "ready"
            return
        if not Path(_COMFY_PY).exists() or not (_COMFY_DIR / "main.py").exists():
            msg = (f"ComfyUI not found (expected {_COMFY_DIR}/main.py and {_COMFY_PY}). Set "
                   "VANISH_COMFY_DIR / VANISH_COMFY_PY, or drop Flux via VANISH_ENGINES=lama,objectclear.")
            _status["flux"] = "error"
            _errors["flux"] = msg
            raise EngineUnavailable(msg)
        _free_for("flux")  # low-mem: free ObjectClear (and the MPS cache) before launching ComfyUI
        _status["flux"] = "loading"
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        _comfy_proc = subprocess.Popen(
            [_COMFY_PY, "main.py", "--port", "8188", "--listen", "127.0.0.1", *_COMFY_ARGS],
            cwd=str(_COMFY_DIR), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        t0 = time.monotonic()
        while time.monotonic() - t0 < _COMFY_START_TIMEOUT:
            if _comfy_up():
                _status["flux"] = "ready"
                return
            time.sleep(1.0)
        _status["flux"] = "error"
        _errors["flux"] = "ComfyUI failed to start within the timeout"
        raise EngineUnavailable("ComfyUI failed to start within the timeout")


def _comfy_upload(img_rgba: Image.Image) -> str:
    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG")
    data = buf.getvalue()
    boundary = "----vanish" + uuid.uuid4().hex
    name = f"vanish_{uuid.uuid4().hex}.png"
    b = boundary.encode()
    body = (b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="image"; filename="' + name.encode() + b'"\r\n'
            b"Content-Type: image/png\r\n\r\n" + data + b"\r\n"
            b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            b"--" + b + b"--\r\n")
    req = urllib.request.Request(_COMFY_URL + "/upload/image", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    return json.load(urllib.request.urlopen(req, timeout=60))["name"]


def _flux_progress_thread(client_id: str, prompt_id: str, steps: int, progress: Progress,
                          stop: threading.Event) -> None:
    """Best-effort real sampling progress from ComfyUI's websocket. If `websocket-client` isn't
    installed, we silently skip (the caller still shows an indeterminate 'Reconstructing…')."""
    try:
        from websocket import create_connection  # websocket-client
    except Exception:
        return
    url = _COMFY_URL.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
    try:
        ws = create_connection(url, timeout=10)
    except Exception:
        return
    try:
        while not stop.is_set():
            try:
                raw = ws.recv()
            except Exception:
                break
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "progress":
                d = msg.get("data", {})
                val, mx = float(d.get("value", 0)), float(d.get("max", steps) or steps)
                progress(0.05 + 0.9 * min(val / max(mx, 1.0), 1.0), "Reconstruct")
            elif msg.get("type") == "executing" and msg.get("data", {}).get("node") is None:
                break
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _run_flux(rgb_u8: np.ndarray, mask_u8: np.ndarray, params: dict, progress: Progress,
              bundle: ModelBundle) -> np.ndarray:
    if not _WF_PATH.exists():
        raise EngineUnavailable(f"Flux workflow not found at {_WF_PATH} (set VANISH_FLUX_WORKFLOW)")
    progress(0.01, "Reconstruct")
    _ensure_comfy()

    steps = int(params.get("steps", 20))
    seed = int(params.get("seed", 0))
    orig = _to_pil(rgb_u8, "RGB")
    work = _fit(orig, 1536, mult=16)  # keep MPS tensors within INT_MAX
    wmask = _to_pil(mask_u8, "L").resize(work.size, Image.NEAREST)
    rgb = np.asarray(work)
    alpha = np.where(np.asarray(wmask) > 127, 0, 255).astype(np.uint8)  # masked(white) → alpha 0 → inpaint
    up = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

    with _infer_lock:
        name = _comfy_upload(up)
        wf = json.load(open(_WF_PATH))
        wf["17"]["inputs"]["image"] = name
        wf["3"]["inputs"]["steps"] = steps
        wf["3"]["inputs"]["seed"] = seed
        client_id = uuid.uuid4().hex
        req = urllib.request.Request(
            _COMFY_URL + "/prompt",
            data=json.dumps({"prompt": wf, "client_id": client_id}).encode(),
            headers={"Content-Type": "application/json"},
        )
        pid = json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]

        stop = threading.Event()
        pth = threading.Thread(target=_flux_progress_thread, args=(client_id, pid, steps, progress, stop), daemon=True)
        pth.start()
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < _FLUX_TIMEOUT:
                time.sleep(1.5)
                try:
                    hist = json.load(urllib.request.urlopen(_COMFY_URL + f"/history/{pid}", timeout=15))
                except Exception:
                    continue
                if pid not in hist:
                    continue
                entry = hist[pid]
                if entry.get("outputs"):
                    info = next((o["images"][0] for o in entry["outputs"].values() if "images" in o), None)
                    if info is None:
                        raise EngineUnavailable("Flux produced no image output")
                    q = urllib.parse.urlencode({"filename": info["filename"],
                                                "subfolder": info.get("subfolder", ""),
                                                "type": info.get("type", "output")})
                    raw = urllib.request.urlopen(_COMFY_URL + "/view?" + q, timeout=60).read()
                    res = Image.open(io.BytesIO(raw)).convert("RGB")
                    progress(0.99, "Reconstruct")
                    return _composite_back(res, orig, _to_pil(mask_u8, "L"))
                for m in entry.get("status", {}).get("messages", []):
                    if m and m[0] == "execution_error":
                        raise EngineUnavailable("ComfyUI: " + json.dumps(m[1])[:300])
            raise EngineUnavailable("Flux timed out")
        finally:
            stop.set()


_RUNNERS = {"lama": _run_lama, "objectclear": _run_objectclear, "flux": _run_flux}


def run_engine(engine: str, rgb_u8: np.ndarray, mask_u8: np.ndarray, params: dict | None,
               progress: Progress, bundle: ModelBundle) -> tuple[np.ndarray, str]:
    """Run the chosen removal engine. Returns (cleaned_rgb_u8, engine_actually_used). A heavy engine
    that can't load (missing deps/models) falls back to LaMa so the erase never hard-fails."""
    # only offer enabled engines (VANISH_ENGINES); anything else routes to LaMa
    engine = engine if (engine in _RUNNERS and engine in _ENABLED) else "lama"
    fn = _RUNNERS[engine]
    try:
        return fn(rgb_u8, mask_u8, params or {}, progress, bundle), engine
    except Exception as e:  # noqa: BLE001 — ANY failure in a heavy engine falls back to LaMa
        if engine == "lama":
            raise  # LaMa is the last resort; let it surface
        _status[engine] = "error"
        _errors[engine] = f"{e.__class__.__name__}: {e}"
        log.warning("%s failed (%s: %s); falling back to LaMa", engine, e.__class__.__name__, e)
        progress(0.1, "Quick Erase (fallback)")
        return _run_lama(rgb_u8, mask_u8, params or {}, progress, bundle), "lama"


def warm_engine(engine: str) -> None:
    """Preload a heavy engine in a background thread (best-effort)."""
    def _w():
        try:
            if engine == "objectclear":
                _get_objectclear()
            elif engine == "flux":
                _ensure_comfy()
        except Exception as e:  # noqa: BLE001
            _status[engine] = "error"
            _errors.setdefault(engine, f"{e.__class__.__name__}: {e}")

    threading.Thread(target=_w, daemon=True).start()


def shutdown() -> None:
    """Terminate the ComfyUI child, if we launched one (called from the app lifespan)."""
    global _comfy_proc
    if _comfy_proc is not None:
        try:
            _comfy_proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        _comfy_proc = None

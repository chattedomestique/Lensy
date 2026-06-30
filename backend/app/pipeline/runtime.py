"""Runtime model bundle. Models are warm-loaded **once** at app startup (FastAPI lifespan)
and held here. Every loader is wrapped so a missing weight / missing dep degrades to a
classic fallback instead of crashing — Lensy must render on a fresh machine before
`setup.sh` has finished caching the big weights (§ README graceful degradation)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("lensy.runtime")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# keep every HF / torch download inside backend/models (git-ignored)
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))


def pick_device() -> str:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch not installed yet — fine, fallbacks are NumPy/OpenCV
        pass
    return "cpu"


@dataclass
class ModelBundle:
    """Holds warm models. A `None` field means "use the classic fallback for that stage"."""

    device: str = "cpu"
    matte_model: object | None = None       # BiRefNet
    matte_processor: object | None = None
    depth_model: object | None = None       # Apple Depth Pro
    depth_transform: object | None = None
    inpaint_model: object | None = None      # LaMa
    has_pymatting: bool = False              # estimate_foreground_ml available
    notes: list[str] = field(default_factory=list)

    def status(self) -> dict:
        return {
            "device": self.device,
            "matte": "birefnet" if self.matte_model else "fallback(grabcut)",
            "depth": "depth-anything-v2" if self.depth_model else "fallback(radial)",
            "inpaint": "lama" if self.inpaint_model else "fallback(cv2)",
            "decontaminate": "pymatting" if self.has_pymatting else "fallback(passthrough)",
            "notes": self.notes,
        }


def load_bundle() -> ModelBundle:
    """Best-effort warm load. Never raises; records what fell back in `notes`."""
    b = ModelBundle(device=pick_device())
    log.info("Lensy device = %s", b.device)

    # --- pymatting (foreground decontamination — the key anti-halo step) ---
    try:
        import pymatting  # noqa: F401

        b.has_pymatting = True
    except Exception as e:
        b.notes.append(f"pymatting unavailable ({e.__class__.__name__}); decontam = passthrough")

    # --- BiRefNet matte (HF transformers) ---
    try:
        b.matte_model, b.matte_processor = _load_birefnet(b.device)
    except Exception as e:
        b.notes.append(f"BiRefNet unavailable ({e.__class__.__name__}); matte = GrabCut fallback")
        log.info("BiRefNet not loaded: %s", e)

    # --- Depth (Depth Anything V2) ---
    try:
        b.depth_model, b.depth_transform = _load_depth(b.device)
    except Exception as e:
        b.notes.append(f"depth model unavailable ({e.__class__.__name__}); depth = radial fallback")
        log.info("Depth model not loaded: %s", e)

    # --- LaMa inpaint (optional; cv2.inpaint is the cheap fallback) ---
    try:
        b.inpaint_model = _load_lama(b.device)
    except Exception as e:
        b.notes.append(f"LaMa unavailable ({e.__class__.__name__}); inpaint = cv2.inpaint fallback")
        log.info("LaMa not loaded: %s", e)

    log.info("Model bundle ready: %s", b.status())
    return b


# ---- loaders (kept thin; each raises on any failure so load_bundle() can fall back) ----


# Matte model. Default to BiRefNet's dedicated **matting** variant (true soft alpha for hair),
# not the general segmentation checkpoint — the matting/HR-matting weights are what protect the
# edge gate (§7.2.1). Override with LENSY_MATTE_MODEL:
#   ZhengPeng7/BiRefNet_HR-matting  — 2048px, best edges (default)
#   ZhengPeng7/BiRefNet-matting     — 1024px matting, lighter
#   ZhengPeng7/BiRefNet-portrait    — tuned for people
_MATTE_MODEL_ID = os.environ.get("LENSY_MATTE_MODEL", "ZhengPeng7/BiRefNet_HR-matting")


def _load_birefnet(device: str):
    """BiRefNet matting variant via transformers AutoModel (trust_remote_code)."""
    import torch
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(_MATTE_MODEL_ID, trust_remote_code=True)
    # weights ship as half; MPS is happiest in float32 — force it to match our float inputs
    model.to(device).float().eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model, None  # BiRefNet preprocessing is simple; done inline in matte.py


# Depth Anything V2 tier — Base is the accuracy/speed sweet spot on Apple Silicon. Override
# with LENSY_DEPTH_MODEL (e.g. depth-anything/Depth-Anything-V2-Large-hf for max accuracy).
# NB: Apple Depth Pro (the brief's first pick) was dropped after benchmarking — on a 16GB M4
# it took 60-130s/render and leaked MPS memory. Depth Anything V2 runs in ~0.2-1s and is
# stable; in this pipeline depth only grades the blur falloff (the clean edge comes from
# matte→decontaminate→inpaint), so boundary accuracy of depth is not the edge gate.
_DEPTH_MODEL_ID = os.environ.get("LENSY_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Base-hf")


def _load_depth(device: str):
    """Depth Anything V2 via transformers. Returns (model, image_processor). The model's
    `predicted_depth` is disparity-like (near = large) — used directly, not inverted."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(_DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_MODEL_ID)
    model.to(device).float().eval()
    return model, processor


def _load_lama(device: str):
    """LaMa big-lama via simple-lama-inpainting. The shipped `big-lama.pt` is a CUDA-traced
    TorchScript module, so it fails to deserialize on a CUDA-less Mac and LaMa's Fourier
    convolutions are unreliable on MPS — we force a CPU load + CPU inference. Inpaint runs
    once per render on a masked region, so the CPU latency (~few seconds) is acceptable."""
    import torch
    from simple_lama_inpainting import SimpleLama

    orig = torch.jit.load
    torch.jit.load = lambda f, *a, **k: orig(f, map_location="cpu")  # noqa: ARG005
    try:
        return SimpleLama(device=torch.device("cpu"))
    finally:
        torch.jit.load = orig

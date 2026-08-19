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
# SAM2 has a few ops without Metal kernels — let them fall back to CPU rather than crash (must be
# set before torch is imported).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


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
    depth_model: object | None = None       # Depth Anything V2/V3, or Depth Pro
    depth_transform: object | None = None
    depth_metric: bool = False               # True = model outputs metric depth (invert for disparity)
    depth_backend: str = "hf"                # "hf" (transformers) or "da3" (Depth Anything 3 pkg)
    depth_name: str = "depth"
    inpaint_model: object | None = None      # LaMa
    sam2_model: object | None = None         # Segment Anything 2 (interactive object select)
    sam2_processor: object | None = None
    has_pymatting: bool = False              # estimate_foreground_ml available
    notes: list[str] = field(default_factory=list)

    def status(self) -> dict:
        return {
            "device": self.device,
            "matte": "birefnet" if self.matte_model else "fallback(grabcut)",
            "depth": self.depth_name if self.depth_model else "fallback(radial)",
            "inpaint": "lama" if self.inpaint_model else "fallback(cv2)",
            "segment": "sam2" if self.sam2_model else "fallback(grabcut)",
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

    # --- Depth ---
    try:
        if "da3" in _DEPTH_MODEL_ID.lower():
            b.depth_model = _load_da3(b.device)
            b.depth_backend = "da3"
            b.depth_metric = True  # DA3 mono outputs (metric) depth → inverted to disparity
            b.depth_name = "depth-anything-v3-mono"
        else:
            b.depth_model, b.depth_transform = _load_depth(b.device)
            b.depth_metric = "depthpro" in _DEPTH_MODEL_ID.lower()
            b.depth_name = "depth-pro" if b.depth_metric else "depth-anything-v2"
    except Exception as e:
        log.info("Depth model %s not loaded: %s", _DEPTH_MODEL_ID, e)
        # DA3's packaging is finicky (xformers/3D extras) — fall back to the reliable V2-Large
        # before dropping all the way to the radial guess.
        try:
            b.depth_model, b.depth_transform = _load_depth(b.device, _DEPTH_FALLBACK_ID)
            b.depth_backend = "hf"
            b.depth_metric = False
            b.depth_name = "depth-anything-v2"
            b.notes.append(f"{_DEPTH_MODEL_ID} unavailable ({e.__class__.__name__}); using V2-Large")
        except Exception as e2:
            b.notes.append(f"depth unavailable ({e2.__class__.__name__}); depth = radial fallback")

    # --- LaMa inpaint (optional; cv2.inpaint is the cheap fallback) ---
    try:
        b.inpaint_model = _load_lama(b.device)
    except Exception as e:
        b.notes.append(f"LaMa unavailable ({e.__class__.__name__}); inpaint = cv2.inpaint fallback")
        log.info("LaMa not loaded: %s", e)

    # --- SAM2 interactive segmentation (optional; GrabCut-around-point is the fallback) ---
    try:
        b.sam2_model, b.sam2_processor = _load_sam2(b.device)
    except Exception as e:
        b.notes.append(f"SAM2 unavailable ({e.__class__.__name__}); segment = grabcut fallback")
        log.info("SAM2 not loaded: %s", e)

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


# Depth model. Default **Apple Depth Pro** (apple/DepthPro-hf) — the brief's first pick (§7.2/§7.4),
# chosen for best-in-class boundary accuracy + thin-structure (hair/fur) recall, which is exactly
# what protects the edge (§7.1). It outputs *metric* depth and is heavy (~1.9GB weights, 1536²
# inference), so on Apple Silicon it's loaded in **fp16** (LENSY_DEPTH_FP16, on by default for MPS)
# to fit a 16GB Mac, and it runs **once per import** — the result is cached for every later slider
# edit, so the ~40-130s cost is paid on import, not per render. If it can't load, Lensy falls back
# to Depth-Anything-V2-Large automatically. Options via LENSY_DEPTH_MODEL:
#   apple/DepthPro-hf                           (metric, richest edges/hair — default)
#   da3mono                                     (Depth Anything V3 mono-Large, ~1.5s — fast + light)
#   depth-anything/Depth-Anything-V2-Large-hf   (~1.5s, transformers-native; the reliable fallback)
#   depth-anything/Depth-Anything-V2-Base-hf    (~0.4s, fastest)
_DEPTH_MODEL_ID = os.environ.get("LENSY_DEPTH_MODEL", "apple/DepthPro-hf")
_DEPTH_FALLBACK_ID = "depth-anything/Depth-Anything-V2-Large-hf"
_DA3_MODEL_ID = os.environ.get("LENSY_DA3_MODEL", "depth-anything/da3mono-large")
# Load Depth Pro in half precision on MPS: ~halves resident + peak memory so it fits a 16GB Mac.
# Set LENSY_DEPTH_FP16=0 to force fp32 (more memory, marginally more precise).
_DEPTH_FP16 = os.environ.get("LENSY_DEPTH_FP16", "1").lower() not in ("0", "false", "no", "")


def _load_depth(device: str, model_id: str | None = None):
    """Load a transformers depth model + processor. Depth Pro needs its own classes and outputs
    *metric* depth (meters → invert for disparity); Depth Anything V2 outputs disparity-like values
    directly. load_bundle() sets `depth_metric` from the model id.

    Depth Pro is loaded in **fp16 on MPS** (LENSY_DEPTH_FP16): it's a ~1.9GB model with a large
    1536² activation peak, and half precision ~halves both resident and peak memory so it fits a
    16GB Mac alongside BiRefNet/SAM2. Apple's own Depth Pro runs fp16, and the precision is plenty
    here (we normalize to [0,1] disparity for the editor). Everything else stays fp32; depth.py
    casts the inputs to match the model's dtype."""
    import torch  # noqa: F401  (imported so a torch-less env fails here → graceful fallback)

    model_id = model_id or _DEPTH_MODEL_ID
    is_depthpro = "depthpro" in model_id.lower()
    if is_depthpro:
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        processor = DepthProImageProcessor.from_pretrained(model_id)
        model = DepthProForDepthEstimation.from_pretrained(model_id)
    else:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
    if is_depthpro and device == "mps" and _DEPTH_FP16:
        model.half().to(device).eval()  # halve on CPU first → no fp32 spike on the device
    else:
        model.to(device).float().eval()
    return model, processor


def _install_da3_stubs() -> None:
    """DA3's official package hard-depends on multi-view / 3D / CUDA extras (xformers, evo, trimesh,
    moviepy) that single-image depth never touches — and its api.py imports two of them at module
    load. We install the package `--no-deps` and stub those two submodules so the import succeeds."""
    import sys
    import types

    for name, attrs in (
        ("depth_anything_3.utils.export", {"export": lambda *a, **k: None}),
        ("depth_anything_3.utils.pose_align", {"align_poses_umeyama": lambda *a, **k: None}),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod


def _load_da3(device: str):
    """Depth Anything V3 (mono). Loads via the `depth_anything_3` package (not transformers).
    Inference is `model.inference([rgb], process_res=…) -> Prediction.depth` (metric depth)."""
    _install_da3_stubs()
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(_DA3_MODEL_ID)
    model.to(device).eval()
    return model


# SAM2 (Segment Anything 2.1) for tap-to-select object removal. `-large` is only ~900MB and the
# expensive image-embed runs once per photo, so per-click latency stays low. Override with
# LENSY_SAM2_MODEL (…-base-plus / -small / -tiny to trim RAM).
_SAM2_MODEL_ID = os.environ.get("LENSY_SAM2_MODEL", "facebook/sam2.1-hiera-large")


def _load_sam2(device: str):
    """SAM2 image predictor via transformers Sam2Model + Sam2Processor. Load then .to(device)
    explicitly (device_map='auto' is a CUDA idiom); float32 on MPS."""
    from transformers import Sam2Model, Sam2Processor

    model = Sam2Model.from_pretrained(_SAM2_MODEL_ID)
    model.to(device).float().eval()
    processor = Sam2Processor.from_pretrained(_SAM2_MODEL_ID)
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

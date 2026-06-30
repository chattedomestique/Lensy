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
            "depth": "depth-pro" if self.depth_model else "fallback(radial)",
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

    # --- Apple Depth Pro ---
    try:
        b.depth_model, b.depth_transform = _load_depth_pro(b.device)
    except Exception as e:
        b.notes.append(f"Depth Pro unavailable ({e.__class__.__name__}); depth = radial fallback")
        log.info("Depth Pro not loaded: %s", e)

    # --- LaMa inpaint (optional; cv2.inpaint is the cheap fallback) ---
    try:
        b.inpaint_model = _load_lama(b.device)
    except Exception as e:
        b.notes.append(f"LaMa unavailable ({e.__class__.__name__}); inpaint = cv2.inpaint fallback")
        log.info("LaMa not loaded: %s", e)

    log.info("Model bundle ready: %s", b.status())
    return b


# ---- loaders (kept thin; each raises on any failure so load_bundle() can fall back) ----


def _load_birefnet(device: str):
    """BiRefNet HR matting via transformers AutoModel (trust_remote_code)."""
    import torch
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    model.to(device).eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model, None  # BiRefNet preprocessing is simple; done inline in matte.py


def _load_depth_pro(device: str):
    """Apple Depth Pro. Uses the `depth_pro` package if installed (from Apple's repo)."""
    import depth_pro  # apple/ml-depth-pro

    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    return model, transform


def _load_lama(device: str):
    """LaMa big-lama via simple-lama-inpainting if available."""
    from simple_lama_inpainting import SimpleLama

    return SimpleLama(device=device)

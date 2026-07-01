"""Best-effort weight pre-cache. Each download is independent and non-fatal — a failure just
means that stage uses its classic fallback until the weight is present. Idempotent (HF/torch
cache skip re-downloads). Run by setup.sh; safe to run directly."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))


def _try(name: str, fn) -> bool:
    print(f"  • {name} …", end=" ", flush=True)
    try:
        fn()
        print("ok")
        return True
    except Exception as e:  # noqa: BLE001 — best effort
        print(f"skip ({e.__class__.__name__}: {e})")
        return False


DEPTH_MODEL_ID = os.environ.get("LENSY_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")


MATTE_MODEL_ID = os.environ.get("LENSY_MATTE_MODEL", "ZhengPeng7/BiRefNet_HR-matting")


def fetch_birefnet() -> None:
    from transformers import AutoModelForImageSegmentation

    AutoModelForImageSegmentation.from_pretrained(MATTE_MODEL_ID, trust_remote_code=True)


def fetch_depth() -> None:
    if "depthpro" in DEPTH_MODEL_ID.lower():
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        DepthProImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        DepthProForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    else:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)


def fetch_lama() -> None:
    import torch
    from simple_lama_inpainting import SimpleLama

    orig = torch.jit.load
    torch.jit.load = lambda f, *a, **k: orig(f, map_location="cpu")  # noqa: ARG005
    try:
        SimpleLama(device=torch.device("cpu"))
    finally:
        torch.jit.load = orig


def main() -> int:
    print("Pre-caching model weights into backend/models/ …")
    results = {
        "BiRefNet (matte)": _try("BiRefNet", fetch_birefnet),
        "Depth Anything V2 (depth)": _try("Depth Anything V2", fetch_depth),
        "LaMa (inpaint)": _try("LaMa", fetch_lama),
    }
    have = sum(results.values())
    print(f"\nCached {have}/{len(results)} model(s). Missing ones fall back automatically.")
    return 0  # never fail the build


if __name__ == "__main__":
    sys.exit(main())

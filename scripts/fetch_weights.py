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


SAM2_MODEL_ID = os.environ.get("LENSY_SAM2_MODEL", "facebook/sam2.1-hiera-large")


def fetch_sam2() -> None:
    from transformers import Sam2Model, Sam2Processor

    Sam2Model.from_pretrained(SAM2_MODEL_ID)
    Sam2Processor.from_pretrained(SAM2_MODEL_ID)


DA3_MODEL_ID = os.environ.get("LENSY_DA3_MODEL", "depth-anything/da3mono-large")


def fetch_da3() -> None:
    # mirror runtime's stubs so the import doesn't drag the 3D/video export deps
    import sys
    import types

    for name, attrs in (
        ("depth_anything_3.utils.export", {"export": lambda *a, **k: None}),
        ("depth_anything_3.utils.pose_align", {"align_poses_umeyama": lambda *a, **k: None}),
    ):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules.setdefault(name, mod)
    from depth_anything_3.api import DepthAnything3

    DepthAnything3.from_pretrained(DA3_MODEL_ID)


def main() -> int:
    print("Pre-caching model weights into backend/models/ …")
    results = {
        "BiRefNet (matte)": _try("BiRefNet", fetch_birefnet),
        "Depth Anything V3 (depth)": _try("Depth Anything V3", fetch_da3),
        "Depth Anything V2 (depth fallback)": _try("Depth Anything V2", fetch_depth),
        "LaMa (inpaint)": _try("LaMa", fetch_lama),
        "SAM2 (object select)": _try("SAM2", fetch_sam2),
    }
    have = sum(results.values())
    print(f"\nCached {have}/{len(results)} model(s). Missing ones fall back automatically.")
    return 0  # never fail the build


if __name__ == "__main__":
    sys.exit(main())

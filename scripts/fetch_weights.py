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


def fetch_birefnet() -> None:
    from transformers import AutoModelForImageSegmentation

    AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)


def fetch_depth_pro() -> None:
    import depth_pro  # apple/ml-depth-pro (pip install git+https://github.com/apple/ml-depth-pro)

    depth_pro.create_model_and_transforms()


def fetch_lama() -> None:
    from simple_lama_inpainting import SimpleLama

    SimpleLama()


def main() -> int:
    print("Pre-caching model weights into backend/models/ …")
    results = {
        "BiRefNet (matte)": _try("BiRefNet", fetch_birefnet),
        "Depth Pro (depth)": _try("Depth Pro", fetch_depth_pro),
        "LaMa (inpaint)": _try("LaMa", fetch_lama),
    }
    have = sum(results.values())
    print(f"\nCached {have}/{len(results)} model(s). Missing ones fall back automatically.")
    return 0  # never fail the build


if __name__ == "__main__":
    sys.exit(main())

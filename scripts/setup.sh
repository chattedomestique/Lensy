#!/usr/bin/env bash
# Lensy setup — idempotent. Creates the Python venv, installs deps (core + optional models),
# installs frontend deps, and best-effort pre-caches model weights. Never asks you to fetch
# weights by hand. Safe to re-run.
#
#   ./scripts/setup.sh            # core + frontend, and attempt model deps/weights
#   LENSY_SKIP_MODELS=1 ...        # skip the heavy model deps/weights (fast, fallback-only)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BOLD=$'\033[1m'; DIM=$'\033[2m'; OK=$'\033[32m'; WARN=$'\033[33m'; OFF=$'\033[0m'
say() { echo "${BOLD}▸ $*${OFF}"; }
ok()  { echo "  ${OK}✓${OFF} $*"; }
warn(){ echo "  ${WARN}!${OFF} $*"; }

# --- prerequisites -----------------------------------------------------------
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
command -v node    >/dev/null || { echo "node not found (needed for the PWA)"; exit 1; }
say "Toolchain"
ok "python $(python3 --version 2>&1 | awk '{print $2}'), node $(node --version)"

# --- backend venv + deps -----------------------------------------------------
say "Backend venv + core deps"
cd "$ROOT/backend"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -e .            # core deps from pyproject [project.dependencies]
ok "core deps installed"

# --- optional model deps + weights (best-effort, never fatal) ----------------
if [ "${LENSY_SKIP_MODELS:-0}" = "1" ]; then
  warn "LENSY_SKIP_MODELS=1 — skipping model deps; Lensy runs in fallback-only mode"
else
  say "Model deps (torch/transformers/pymatting …) — best effort"
  if pip install -q -e ".[models]"; then
    ok "model deps installed"
    # simple-lama-inpainting pulls in plain opencv-python, which clobbers the contrib build
    # we need for cv2.ximgproc.guidedFilter (§7.2.2). Normalize to contrib-only.
    if pip show opencv-python >/dev/null 2>&1; then
      pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
      pip install -q --force-reinstall "opencv-contrib-python>=4.9"
      ok "normalized OpenCV to contrib-only (guidedFilter available)"
    fi
    say "Pre-caching weights (BiRefNet, Depth Pro, LaMa) — may take a while, never fatal"
    python "$ROOT/scripts/fetch_weights.py" || warn "some weights not cached; fallbacks will be used until they are"
  else
    warn "model deps failed to install — Lensy will use classic fallbacks (still works)"
  fi
fi
deactivate

# --- frontend deps -----------------------------------------------------------
say "Frontend deps"
cd "$ROOT/frontend"
npm install --silent
ok "frontend deps installed"

echo
echo "${BOLD}Lensy is set up.${OFF} Next: ${BOLD}./scripts/dev.sh${OFF}"

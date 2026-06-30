#!/usr/bin/env bash
# Lensy dev loop — runs FastAPI (uvicorn --reload) + the Vite dev server together. Ctrl-C
# stops both. Pass --tunnel to also bring up a cloudflared quick tunnel to the backend.
#
#   ./scripts/dev.sh
#   ./scripts/dev.sh --tunnel
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOLD=$'\033[1m'; OFF=$'\033[0m'

WANT_TUNNEL=0
[ "${1:-}" = "--tunnel" ] && WANT_TUNNEL=1

if [ ! -d "$ROOT/backend/.venv" ]; then
  echo "No backend venv — run ./scripts/setup.sh first."; exit 1
fi

pids=()
cleanup() {
  echo
  echo "${BOLD}Stopping Lensy…${OFF}"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- backend: uvicorn --reload ----------------------------------------------
echo "${BOLD}▸ backend${OFF}  http://localhost:8000   (docs: /docs)"
( cd "$ROOT/backend" && source .venv/bin/activate \
    && exec uvicorn app.main:app --reload --port 8000 ) &
pids+=("$!")

# --- frontend: vite ----------------------------------------------------------
echo "${BOLD}▸ frontend${OFF} http://localhost:5173"
( cd "$ROOT/frontend" && exec npm run dev -- --host ) &
pids+=("$!")

# --- optional: cloudflared quick tunnel -------------------------------------
if [ "$WANT_TUNNEL" = "1" ]; then
  if command -v cloudflared >/dev/null; then
    echo "${BOLD}▸ tunnel${OFF}   cloudflared → backend :8000"
    ( exec cloudflared tunnel --url http://localhost:8000 ) &
    pids+=("$!")
  else
    echo "  ! cloudflared not installed — skipping tunnel (brew install cloudflared)"
  fi
fi

echo
echo "${BOLD}Lensy is running.${OFF} Open http://localhost:5173 — Ctrl-C to stop."
wait

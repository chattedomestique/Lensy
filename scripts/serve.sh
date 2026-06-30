#!/usr/bin/env bash
# Lensy production serve — runs the render backend so your hosted PWA can reach it.
#
#   ./scripts/serve.sh            # run the backend on :8842 for your NAMED Cloudflare Tunnel
#   ./scripts/serve.sh --quick    # also open a throwaway *.trycloudflare.com tunnel (no account)
#
# Normal setup: a Cloudflare *named tunnel* maps  lensy.sunhouse.media → http://localhost:8842
# (you manage cloudflared yourself, like your other tunnels). The hosted app defaults to that
# URL, so it just connects. Add this ingress rule to your cloudflared config.yml:
#
#   ingress:
#     - hostname: lensy.sunhouse.media
#       service: http://localhost:8842
#     - service: http_status:404
#
# then:  cloudflared tunnel route dns <tunnel-name> lensy.sunhouse.media
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOLD=$'\033[1m'; ORANGE=$'\033[38;5;209m'; DIM=$'\033[2m'; OFF=$'\033[0m'

PORT="${LENSY_PORT:-8842}"
WANT_QUICK=0
[ "${1:-}" = "--quick" ] && WANT_QUICK=1

if [ ! -d "$ROOT/backend/.venv" ]; then
  echo "No backend venv — run ./scripts/setup.sh first."; exit 1
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Set LENSY_PORT to a free port and update your tunnel ingress."; exit 1
fi

TUNLOG="$(mktemp -t lensy-tunnel)"
pids=()
cleanup() {
  echo; echo "${BOLD}Stopping Lensy…${OFF}"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  rm -f "$TUNLOG"; wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- backend (production: no --reload) ---
echo "${BOLD}▸ backend${OFF}  http://localhost:${PORT}"
( cd "$ROOT/backend" && source .venv/bin/activate \
    && exec uvicorn app.main:app --host 127.0.0.1 --port "$PORT" ) &
pids+=("$!")

for _ in $(seq 1 60); do
  curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1 && break; sleep 1
done

echo
echo "${BOLD}Lensy backend is live on :${PORT}.${OFF}"
echo "  ${BOLD}${ORANGE}https://lensy.sunhouse.media${OFF}  ${DIM}(via your named tunnel → localhost:${PORT})${OFF}"
echo "  ${DIM}App:  https://chattedomestique.github.io/Lensy/  — auto-connects to that URL${OFF}"

# --- optional throwaway tunnel (if the named one isn't up yet) ---
if [ "$WANT_QUICK" = "1" ]; then
  if command -v cloudflared >/dev/null; then
    echo "${BOLD}▸ quick tunnel${OFF}  opening a throwaway *.trycloudflare.com …"
    ( exec cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" ) >"$TUNLOG" 2>&1 &
    pids+=("$!")
    URL=""
    for _ in $(seq 1 30); do
      URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNLOG" | head -1 || true)"
      [ -n "$URL" ] && break; sleep 1
    done
    [ -n "$URL" ] && echo "  ${ORANGE}${URL}${OFF}  ${DIM}(paste into the app's Server pill)${OFF}"
  else
    echo "  ! cloudflared not installed — skipping quick tunnel (brew install cloudflared)"
  fi
fi

echo "  ${DIM}Ctrl-C to stop.${OFF}"
wait

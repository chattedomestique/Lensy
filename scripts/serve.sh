#!/usr/bin/env bash
# Lensy production serve — runs the render backend and exposes it to the hosted PWA over a
# Cloudflare Tunnel. Prints the public HTTPS URL to paste into the app's "Connect to your
# render server" panel. Ctrl-C stops both.
#
#   ./scripts/serve.sh
#
# No Cloudflare account needed: this uses a *quick tunnel* (ephemeral *.trycloudflare.com URL,
# new each run). For a stable URL, set up a named tunnel + your own domain and run cloudflared
# with that config instead — then you only paste the URL once.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOLD=$'\033[1m'; ORANGE=$'\033[38;5;209m'; DIM=$'\033[2m'; OFF=$'\033[0m'

if [ ! -d "$ROOT/backend/.venv" ]; then
  echo "No backend venv — run ./scripts/setup.sh first."; exit 1
fi
if ! command -v cloudflared >/dev/null; then
  echo "cloudflared not installed. Install it with:  brew install cloudflared"; exit 1
fi

PORT="${LENSY_PORT:-8000}"
TUNLOG="$(mktemp -t lensy-tunnel)"
pids=()
cleanup() {
  echo; echo "${BOLD}Stopping Lensy…${OFF}"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  rm -f "$TUNLOG"; wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- backend (no --reload in production) ---
echo "${BOLD}▸ backend${OFF}  http://localhost:${PORT}"
( cd "$ROOT/backend" && source .venv/bin/activate \
    && exec uvicorn app.main:app --host 127.0.0.1 --port "$PORT" ) &
pids+=("$!")

# wait for the backend to answer before opening the tunnel
for _ in $(seq 1 60); do
  curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1 && break; sleep 1
done

# --- cloudflared quick tunnel ---
echo "${BOLD}▸ tunnel${OFF}   opening Cloudflare quick tunnel…"
( exec cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" ) >"$TUNLOG" 2>&1 &
pids+=("$!")

# scrape the public URL cloudflared prints, then show it prominently
URL=""
for _ in $(seq 1 30); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNLOG" | head -1 || true)"
  [ -n "$URL" ] && break; sleep 1
done

echo
if [ -n "$URL" ]; then
  echo "${BOLD}Lensy backend is live.${OFF}"
  echo "  ${BOLD}${ORANGE}${URL}${OFF}"
  echo "  ${DIM}↑ paste this into the app → \"Connect to your render server\"${OFF}"
  echo "  ${DIM}App:  https://chattedomestique.github.io/Lensy/${OFF}"
else
  echo "Tunnel didn't report a URL yet — check below:"; tail -20 "$TUNLOG"
fi
echo "  ${DIM}Ctrl-C to stop.${OFF}"
wait

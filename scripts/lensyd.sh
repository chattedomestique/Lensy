#!/usr/bin/env bash
# Lensy production daemon — the entrypoint launchd runs. Not for interactive use:
# use ./scripts/lensyctl.sh (install/status/restart/logs) or ./scripts/serve.sh for a foreground run.
#
# launchd runs an INSTALLED COPY of this file (LENSY_PREFIX, default /opt/homebrew/lib/lensy),
# never the one in the git checkout: on 2026-08-19 a `git reset --hard` deleted it out from under
# the running agent and launchd crash-looped ~2,700 times — about a day down. The checkout is still
# where the code, .venv and weights live; LENSY_ROOT points there.
#
# Deliberately has NO respawn loop of its own: launchd's KeepAlive is the one supervisor.
# Two supervisors fight (launchd restarts the wrapper while the wrapper restarts uvicorn),
# which is how you get an orphan holding the port and a tunnel stuck at 502.
set -euo pipefail

ROOT="${LENSY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PORT="${LENSY_PORT:-8842}"
LOGDIR="$HOME/Library/Logs/lensy"
FAULT="$LOGDIR/config-fault"
mkdir -p "$LOGDIR"

# SIGTERM is always an intentional stop (lensyctl stop, bootout, kickstart -k). Exit 0 so KeepAlive
# leaves it down — including mid-backoff below, where bash would otherwise die with 143 and be
# respawned. Once uvicorn is running, the forwarding trap at the bottom takes over.
trap 'exit 0' TERM

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "── lensyd start $(stamp)  pid=$$  port=${PORT}  root=${ROOT}"

# A setup fault (no venv, port owned by another service) won't clear itself in 30 s, and launchd
# can't be told "don't restart on exit 78" — KeepAlive only knows success vs failure. So record the
# reason (the watchdog reads it and stands down instead of kickstarting into the same wall) and
# hold before exiting: one retry every ~5 min that self-heals when the cause clears, instead of a
# respawn — and a full model load — every 30 s burying the one line that matters.
fault() {
  local wait_s="${LENSY_FAULT_BACKOFF:-300}"
  echo "!! $* — retrying in ${wait_s}s"
  echo "$*" >"$FAULT"
  sleep "$wait_s" & wait $!  # `wait` is interruptible, so a stop lands now, not in 5 min
  exit 78  # EX_CONFIG
}

[ -x "$ROOT/backend/.venv/bin/uvicorn" ] \
  || fault "no backend venv at $ROOT/backend/.venv (run ./scripts/setup.sh)"

# A previous instance SIGKILLed mid-render can leave the port bound. Reclaim it — but only from
# our own uvicorn, never from whatever else is listening (this Mac runs a lot of services).
# -sTCP:LISTEN matters: a bare `-i tcp:PORT` also matches the *client* end of every connection to
# the port — cloudflared — and killing that drops every tunnel on the Mac, not just Lensy's.
for pid in $(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true); do
  if ps -o command= -p "$pid" 2>/dev/null | grep -q 'app.main:app'; then
    echo "   reclaiming port ${PORT} from stale lensy pid ${pid}"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
  else
    fault "port ${PORT} is held by pid ${pid} ($(ps -o comm= -p "$pid" 2>/dev/null)), not Lensy — refusing to kill it"
  fi
done
rm -f "$FAULT"

# Serve the built PWA from this origin. Only if missing — scripts/build.sh forces a refresh.
# A failed build must not stop the backend: the API still works, the UI is just stale/absent.
if [ ! -f "$ROOT/frontend/dist/index.html" ]; then
  echo "   front-end not built — building once…"
  ( cd "$ROOT/frontend" && npm run build ) >>"$LOGDIR/build.log" 2>&1 \
    && echo "   front-end built → frontend/dist" \
    || echo "!! front-end build failed (see ${LOGDIR}/build.log) — serving API only"
fi

cd "$ROOT/backend"

# uvicorn runs as a child, not via `exec`. It re-raises SIGTERM after its graceful shutdown
# (capture_signals, 0.29+), so on its own it dies *by signal* — which launchd counts as a failure
# and respawns: a `lensyctl stop` came back 31 s later (measured: exit 143). So stay the parent,
# forward the stop, and exit 0 for it. Any exit nobody asked for — crash, OOM kill, exception —
# passes through as a failure so KeepAlive restarts it. Still no restart loop here: launchd
# remains the one supervisor.
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --timeout-graceful-shutdown 20 &
child=$!
stop_requested=0
trap 'stop_requested=1; kill -TERM "$child" 2>/dev/null || true' TERM INT
ec=0
while :; do
  wait "$child" && ec=0 || ec=$?
  kill -0 "$child" 2>/dev/null || break  # a trapped signal cuts `wait` short — wait again
done
if [ "$stop_requested" = 1 ]; then
  echo "── lensyd stop $(stamp)  (requested)"
  exit 0
fi
echo "!! uvicorn exited unexpectedly (status ${ec}) — launchd will restart it"
exit $(( ec == 0 ? 1 : ec ))

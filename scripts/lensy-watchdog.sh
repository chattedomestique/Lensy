#!/usr/bin/env bash
# Liveness probe for the Lensy backend. launchd's KeepAlive only knows whether the *process*
# exists — it cannot see a wedged one (MPS stall, deadlocked worker) that still holds the port
# and answers nothing. That's the failure that leaves the tunnel at 502 with everything
# "running". This probes /healthz and kickstarts the service when it stops answering.
#
# Run every 120 s by com.sunhouse.lensy-watchdog, from the same installed copy as lensyd.sh.
set -euo pipefail

PORT="${LENSY_PORT:-8842}"
LABEL="com.sunhouse.lensy"
URL="http://127.0.0.1:${PORT}/healthz"
LOGDIR="$HOME/Library/Logs/lensy"
STATE="$LOGDIR/watchdog.state"
FAULT="$LOGDIR/config-fault"
mkdir -p "$LOGDIR"

# Consecutive-failure thresholds, in ticks of the probe interval (120 s): dead → ~4 min of
# silence; warming → ~20 min, since a cold load of all four models on a 16GB Mac is legitimately
# slow and must never be interrupted halfway.
DEAD_TICKS="${LENSY_WATCHDOG_DEAD_TICKS:-2}"
WARM_TICKS="${LENSY_WATCHDOG_WARM_TICKS:-10}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }
notify() { osascript -e "display notification \"$1\" with title \"Lensy\"" >/dev/null 2>&1 || true; }

# launchd never rotates its StandardOutPath, and uvicorn logs every request — left alone these
# grow without bound. Copy-truncate keeps the fd launchd already holds valid.
rotate_logs() {
  local f max=20971520  # 20 MB
  for f in "$LOGDIR/lensy.log" "$LOGDIR/watchdog.log"; do
    [ -f "$f" ] || continue
    [ "$(stat -f%z "$f" 2>/dev/null || echo 0)" -gt "$max" ] || continue
    cp "$f" "${f}.1" && : >"$f"
    log "rotated $(basename "$f") (>20MB) → $(basename "$f").1"
  done
}
rotate_logs

# `lensyctl stop` means stopped. Without this, a deliberate stop looks exactly like a crash
# and gets "rescued" four minutes later.
[ -f "$LOGDIR/stopped" ] && exit 0

read_state() { [ -f "$STATE" ] && cat "$STATE" || echo "0 0"; }
write_state() { echo "$1 $2" >"$STATE"; }
read -r dead_n warm_n <<<"$(read_state)"

# %{http_code} is 000 for connection-refused and for timeout — both mean "not answering".
# Capture the code and curl's exit status separately: `... || echo 000` would *append* to the
# code curl already printed, turning a real status into an unmatchable string.
code="$(curl -s -o /dev/null -w '%{http_code}' -m 15 "$URL" 2>/dev/null)" || true
[ -z "$code" ] && code="000"

case "$code" in
  200)
    { [ "$dead_n" != "0" ] || [ "$warm_n" != "0" ]; } && log "healthy again (was dead=${dead_n} warm=${warm_n})"
    write_state 0 0
    rm -f "$LOGDIR/fault.flag"
    # Up is not the same as good. load_bundle() never raises — a model that failed to load just
    # becomes "fallback(grabcut)" / "fallback(radial)", so Lensy answers 200 while quietly
    # rendering at a quality that fails the §7 edge bar. Surface it instead of shipping it.
    body="$(curl -s -m 10 "$URL" 2>/dev/null || true)"
    if printf '%s' "$body" | grep -q 'fallback('; then
      degraded="$(printf '%s' "$body" | tr ',' '\n' | grep 'fallback(' | tr -d '"{}' | paste -sd' ' -)"
      if [ ! -f "$LOGDIR/degraded.flag" ]; then
        log "!! DEGRADED — models fell back to classic paths: ${degraded}"
        log "   renders will look worse than they should. Check weights, then: lensyctl restart"
        notify "Some models fell back — renders are degraded."
        : >"$LOGDIR/degraded.flag"
      fi
    elif [ -f "$LOGDIR/degraded.flag" ]; then
      log "all models loaded properly again"; rm -f "$LOGDIR/degraded.flag"
    fi
    exit 0
    ;;
  503)
    # Models still loading. Alive, just not ready — only escalate if it never finishes.
    warm_n=$((warm_n + 1)); dead_n=0
    write_state "$dead_n" "$warm_n"
    log "warming (${warm_n}/${WARM_TICKS})"
    [ "$warm_n" -lt "$WARM_TICKS" ] && exit 0
    reason="stuck warming for ${warm_n} ticks"
    ;;
  *)
    # lensyd.sh hit a setup fault and is retrying on its own slow cadence (it rewrites the marker
    # every ~5 min, so a stale one is ignored). Kickstarting would only run it into the same wall
    # — and notify every four minutes. Say so once instead.
    if [ -n "$(find "$FAULT" -mmin -15 2>/dev/null)" ]; then
      if [ ! -f "$LOGDIR/fault.flag" ]; then
        log "!! not answering — lensyd reports a setup fault: $(cat "$FAULT")"
        log "   standing down (lensyd retries every ~5 min). Fix the cause, then: lensyctl restart"
        notify "Can't start — $(cut -c1-90 "$FAULT")"
        : >"$LOGDIR/fault.flag"
      fi
      write_state 0 0
      exit 0
    fi
    dead_n=$((dead_n + 1)); warm_n=0
    write_state "$dead_n" "$warm_n"
    log "no answer from ${URL} (http=${code}) (${dead_n}/${DEAD_TICKS})"
    [ "$dead_n" -lt "$DEAD_TICKS" ] && exit 0
    reason="unresponsive for ${dead_n} ticks (http=${code})"
    ;;
esac

log "!! ${reason} — kickstarting ${LABEL}"
launchctl kickstart -k "gui/$(id -u)/${LABEL}" 2>&1 | sed 's/^/   /' || log "   kickstart failed (is the agent loaded? lensyctl install)"
write_state 0 0
notify "Backend was unresponsive — restarted it."

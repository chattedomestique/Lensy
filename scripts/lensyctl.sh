#!/usr/bin/env bash
# Lensy service control — install once, then it survives reboots, crashes, wedges, and whatever
# happens to the git checkout.
#
#   ./scripts/lensyctl.sh install     # install + start both launchd agents (idempotent — re-run
#                                     #   after pulling changes to lensyd.sh, the watchdog, templates)
#   ./scripts/lensyctl.sh status      # is it actually up? end-to-end, through the tunnel
#   ./scripts/lensyctl.sh restart     # bounce the backend
#   ./scripts/lensyctl.sh stop|start  # stop also pauses the watchdog, so it stays stopped
#   ./scripts/lensyctl.sh logs [-f]   # backend log (-f to follow)
#   ./scripts/lensyctl.sh uninstall   # remove both agents and the installed scripts
#
#   LENSY_PREFIX  where the supervisor scripts are installed (default /opt/homebrew/lib/lensy, beside
#                 this Mac's other launchd-run tools). launchd runs them from there, not the checkout.
#   LENSY_ROOT    the checkout the service runs — code, .venv, weights (default: this repo).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${LENSY_ROOT:-$SRC}"
PREFIX="${LENSY_PREFIX:-/opt/homebrew/lib/lensy}"
PORT="${LENSY_PORT:-8842}"
UID_N="$(id -u)"
AGENTS="$HOME/Library/LaunchAgents"
LOGDIR="$HOME/Library/Logs/lensy"
LABEL="com.sunhouse.lensy"
WLABEL="com.sunhouse.lensy-watchdog"
PUBLIC="${LENSY_PUBLIC_URL:-https://lensy.sunhouse.media}"
SHIMS=(lensyd.sh lensy-watchdog.sh)

BOLD=$'\033[1m'; ORANGE=$'\033[38;5;209m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; OFF=$'\033[0m'
ok()   { echo "  ${GREEN}●${OFF} $*"; }
bad()  { echo "  ${RED}●${OFF} $*"; }
warn() { echo "  ${ORANGE}●${OFF} $*"; }

loaded() { launchctl print "gui/${UID_N}/$1" >/dev/null 2>&1; }

install_shims() {
  mkdir -p "$PREFIX"
  local s
  for s in "${SHIMS[@]}"; do install -m 755 "$SRC/scripts/$s" "$PREFIX/$s"; done
  echo "  copied ${SHIMS[*]} → ${PREFIX}"
}

install_agent() {
  local label="$1" tmpl="$SRC/scripts/launchd/$1.plist.template" dest="$AGENTS/$1.plist"
  sed -e "s|__BIN__|$PREFIX|g" -e "s|__ROOT__|$ROOT|g" -e "s|__HOME__|$HOME|g" -e "s|__PORT__|$PORT|g" \
    "$tmpl" >"$dest"
  plutil -lint "$dest" >/dev/null || { echo "!! generated plist is invalid: $dest"; exit 1; }
  # bootout is ASYNCHRONOUS — the service has to finish terminating before the label frees up.
  # Bootstrapping too early returns "Bootstrap failed: 5: Input/output error", which reads like
  # a broken plist but is really just a race. Wait for the label to actually disappear.
  if loaded "$label"; then
    launchctl bootout "gui/${UID_N}/${label}" 2>/dev/null || true
    for _ in $(seq 1 40); do loaded "$label" || break; sleep 0.5; done
  fi
  # A `launchctl disable` persists across reboots and makes bootstrap refuse the job; clear it.
  launchctl enable "gui/${UID_N}/${label}" 2>/dev/null || true
  local tries=0
  until launchctl bootstrap "gui/${UID_N}" "$dest" 2>/dev/null; do
    tries=$((tries + 1))
    [ "$tries" -ge 10 ] && { echo "!! could not bootstrap ${label}"; launchctl bootstrap "gui/${UID_N}" "$dest"; exit 1; }
    sleep 1
  done
  echo "  loaded ${label}"
}

cmd_install() {
  [ -x "$ROOT/backend/.venv/bin/uvicorn" ] || { echo "No backend venv in $ROOT — run ./scripts/setup.sh first."; exit 1; }
  mkdir -p "$AGENTS" "$LOGDIR"
  rm -f "$LOGDIR/stopped"
  echo "${BOLD}▸ installing${OFF}  ${DIM}(serving ${ROOT})${OFF}"
  install_shims
  install_agent "$LABEL"
  install_agent "$WLABEL"
  echo
  echo "${BOLD}Waiting for models to warm…${OFF}"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 && break; sleep 1
  done
  echo
  cmd_status
}

cmd_uninstall() {
  local l s
  for l in "$WLABEL" "$LABEL"; do
    loaded "$l" && launchctl bootout "gui/${UID_N}/${l}" 2>/dev/null || true
    rm -f "$AGENTS/$l.plist"
    echo "  removed $l"
  done
  for s in "${SHIMS[@]}"; do rm -f "$PREFIX/$s"; done
  rmdir "$PREFIX" 2>/dev/null || true
  echo "  removed ${PREFIX}"
}

# The watchdog treats a missing backend as a wedge and kickstarts it — so an intentional stop has
# to tell it to stand down, or `stop` would quietly undo itself four minutes later.
cmd_start()   { rm -f "$LOGDIR/stopped"; launchctl kickstart "gui/${UID_N}/${LABEL}" && echo "  started"; }
cmd_stop()    { mkdir -p "$LOGDIR"; : >"$LOGDIR/stopped"; launchctl kill SIGTERM "gui/${UID_N}/${LABEL}" 2>/dev/null || true; echo "  stopped — stays down (watchdog paused) until: lensyctl start"; }
cmd_restart() { rm -f "$LOGDIR/stopped"; launchctl kickstart -k "gui/${UID_N}/${LABEL}" && echo "  restarting…"; }

cmd_logs() {
  [ "${1:-}" = "-f" ] && exec tail -f "$LOGDIR/lensy.log"
  tail -n 60 "$LOGDIR/lensy.log" 2>/dev/null || echo "(no log yet)"
}

cmd_status() {
  echo "${BOLD}Lensy${OFF} ${DIM}— ${ROOT}${OFF}"

  if loaded "$LABEL"; then
    local pid; pid="$(launchctl print "gui/${UID_N}/${LABEL}" 2>/dev/null | awk -F'= *' '/^\tpid =/{print $2; exit}')"
    if [ -n "${pid:-}" ]; then ok "agent ${LABEL} running (pid ${pid})"
    elif [ -f "$LOGDIR/stopped" ]; then warn "agent ${LABEL} stopped on purpose — lensyctl start"
    else bad "agent ${LABEL} loaded but not running"; fi
  else
    bad "agent ${LABEL} NOT installed — run: ./scripts/lensyctl.sh install"
  fi

  # The agent must run the installed lensyd.sh. Anything else (say, re-pointed at serve.sh by
  # hand) means the watchdog, fault backoff and single-supervisor guarantees are all gone.
  local prog; prog="$(plutil -extract ProgramArguments.1 raw "$AGENTS/$LABEL.plist" 2>/dev/null || true)"
  if [ -n "$prog" ] && [ "$prog" != "$PREFIX/lensyd.sh" ]; then
    bad "agent runs ${prog}, not ${PREFIX}/lensyd.sh — run: lensyctl install"
  fi
  local s
  for s in "${SHIMS[@]}"; do
    if [ ! -f "$PREFIX/$s" ]; then bad "${PREFIX}/${s} missing — run: lensyctl install"
    elif [ -f "$SRC/scripts/$s" ] && ! cmp -s "$SRC/scripts/$s" "$PREFIX/$s"; then
      warn "installed ${s} is older than this checkout's — run: lensyctl install"
    fi
  done

  if loaded "$WLABEL"; then
    if [ -f "$LOGDIR/stopped" ]; then warn "watchdog paused (stopped on purpose)"; else ok "watchdog ${WLABEL} loaded"; fi
  else
    bad "watchdog NOT installed"
  fi
  if [ -n "$(find "$LOGDIR/config-fault" -mmin -15 2>/dev/null)" ]; then
    bad "setup fault: $(cat "$LOGDIR/config-fault")"
  fi

  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then ok "listening on :${PORT}"; else bad "nothing listening on :${PORT}"; fi

  local code body; body="$(curl -s -m 15 "http://127.0.0.1:${PORT}/healthz" 2>/dev/null || true)"
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 15 "http://127.0.0.1:${PORT}/healthz" 2>/dev/null)" || true
  [ -z "$code" ] && code="000"
  case "$code" in
    200)
      ok "local /healthz → 200"
      # Which models are real vs fallen back — 200 alone doesn't mean renders are any good.
      printf '%s' "$body" | tr ',' '\n' | grep -E '"(matte|depth|inpaint|segment|decontaminate)"' \
        | tr -d '"{}' | while read -r m; do
            case "$m" in
              *fallback*) bad "  ${m}  ${DIM}← degraded${OFF}" ;;
              *)          echo "      ${DIM}${m}${OFF}" ;;
            esac
          done
      ;;
    503) warn "local /healthz → 503 (models still warming)" ;;
    *)   bad "local /healthz → ${code} (no answer)" ;;
  esac

  if pgrep -f "cloudflared tunnel run" >/dev/null 2>&1; then ok "cloudflared tunnel running"; else bad "cloudflared tunnel NOT running"; fi

  local pcode; pcode="$(curl -s -o /dev/null -w '%{http_code}' -m 20 "${PUBLIC}/healthz" 2>/dev/null)" || true
  case "${pcode:-000}" in
    200) ok "${ORANGE}${PUBLIC}${OFF} → 200 ${DIM}(end-to-end OK)${OFF}" ;;
    502) bad "${PUBLIC} → 502 ${DIM}(tunnel up, backend not answering)${OFF}" ;;
    *)   bad "${PUBLIC} → ${pcode:-000}" ;;
  esac

  echo "${DIM}  logs: ${LOGDIR}/lensy.log · watchdog: ${LOGDIR}/watchdog.log${OFF}"
}

case "${1:-status}" in
  install) cmd_install ;; uninstall) cmd_uninstall ;;
  start) cmd_start ;; stop) cmd_stop ;; restart) cmd_restart ;;
  status) cmd_status ;; logs) shift; cmd_logs "$@" ;;
  *) echo "usage: $0 {install|uninstall|start|stop|restart|status|logs [-f]}"; exit 2 ;;
esac

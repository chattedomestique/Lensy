#!/usr/bin/env bash
# Lensy production build — produces the PWA bundle in frontend/dist/ for Cloudflare Pages and
# verifies the backend imports + loads its model bundle cleanly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOLD=$'\033[1m'; OK=$'\033[32m'; OFF=$'\033[0m'
say() { echo "${BOLD}▸ $*${OFF}"; }

# --- frontend bundle ---------------------------------------------------------
say "Building frontend → frontend/dist/"
( cd "$ROOT/frontend" && npm run build )
echo "  ${OK}✓${OFF} PWA bundle ready (frontend/dist/)"

# --- backend sanity ----------------------------------------------------------
say "Verifying backend imports + model bundle"
( cd "$ROOT/backend" && source .venv/bin/activate \
  && python -c "
from app.main import app
from app.pipeline.runtime import load_bundle
b = load_bundle()
print('  backend OK —', app.title, app.version)
print('  device:', b.device, '| stages:', b.status())
" )
echo
echo "${BOLD}Build complete.${OFF} Deploy frontend/dist/ to Cloudflare Pages; run the backend behind the tunnel."

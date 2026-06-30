"""FastAPI entrypoint. Models warm-load **once** in the lifespan handler and stay resident
(§4) — never per request. Run with: `uvicorn app.main:app --reload` (see scripts/dev.sh)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .pipeline.runtime import load_bundle

# The built PWA, if present. When it exists we serve the whole app from this same origin
# (e.g. https://lensy.sunhouse.media) — UI + API together, no CORS, no separate host.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-16s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lensy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Lensy warming up — loading models…")
    app.state.bundle = load_bundle()
    log.info("Lensy ready.")
    yield
    app.state.bundle = None
    log.info("Lensy shut down.")


app = FastAPI(title="Lensy", version="0.1.0", lifespan=lifespan)

# Dev: the PWA is served from a different origin (Vite on :5173). In production it rides the
# Cloudflare Tunnel same-origin, so this is permissive but harmless for a personal tool.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# Serve the built front-end at the root, so the backend's own domain *is* the app. API routes
# above (/render, /healthz, /docs) are registered first and take precedence; StaticFiles(html=True)
# serves index.html for "/" and the hashed assets for everything else. Mounted last, and only if
# the bundle has been built (scripts/serve.sh + build.sh build it) — otherwise we expose a hint.
if (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="app")
else:
    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse(
            {
                "app": "Lensy",
                "note": "front-end not built — run scripts/build.sh (or scripts/serve.sh)",
                "docs": "/docs",
                "health": "/healthz",
            }
        )

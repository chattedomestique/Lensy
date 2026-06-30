"""FastAPI entrypoint. Models warm-load **once** in the lifespan handler and stay resident
(§4) — never per request. Run with: `uvicorn app.main:app --reload` (see scripts/dev.sh)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .pipeline.runtime import load_bundle

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


@app.get("/")
async def root() -> dict:
    return {"app": "Lensy", "docs": "/docs", "health": "/healthz"}

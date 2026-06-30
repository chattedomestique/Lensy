# Lensy

A private, local **"Portrait Mode" for still photos** — separate subject from background and render a
*physically believable* lens blur with **clean edges** (no halo, no fringe, no chewed hair).

> Personal tooling. Not commercial, not distributed. See [`CLAUDE.md`](CLAUDE.md) for the full brief —
> the render pipeline (§7) and aesthetic (§5) are the load-bearing parts.

## Shape

```
PWA (Vite, vanilla TS)  ──HTTPS──▶  FastAPI + PyTorch (MPS)  ──▶  composited result
   frontend/                            backend/
```

- **frontend/** — installable PWA. Upload → tweak lens controls → render → before/after compare → download.
- **backend/** — FastAPI. Warm-loads models once, runs the edge-safe render pipeline, streams progress over SSE.
- **scripts/** — `setup.sh`, `dev.sh`, `build.sh`. Command-line only; no IDE required.

## Quickstart

```bash
./scripts/setup.sh    # venv + python deps + frontend deps + model weights (idempotent)
./scripts/dev.sh      # backend (uvicorn --reload) + frontend dev server, together
```

Then open the printed frontend URL (default http://localhost:5173). Backend lives at http://localhost:8000.

### Graceful degradation
The pipeline is built so it **runs before every heavy model is present**. Each stage falls back:

| Stage          | Primary                    | Fallback (auto)                 |
|----------------|----------------------------|---------------------------------|
| Matte          | BiRefNet (HR)              | classic GrabCut / luminance     |
| Refine         | guided filter (ximgproc)   | bilateral / box                 |
| Decontaminate  | pymatting `estimate_fg_ml` | premultiplied passthrough       |
| Depth          | Apple Depth Pro            | radial / luminance disparity    |
| Inpaint        | LaMa (big-lama)            | `cv2.inpaint` (Telea)           |
| Blur           | linear-light scatter (ours)| —  (always available, MPS-safe) |

So a first render works immediately; quality climbs as `setup.sh` finishes caching weights.

## The one inviolable rule (§7.1)

> matte → **decontaminate** foreground color → remove + inpaint the subject's hole → blur the *clean*
> background (depth-graded, linear light, scatter) → recomposite the sharp subject with premultiplied alpha.

"Blur the whole image then paste the subject back" is **forbidden** — it is the direct cause of edge halos
and chewed hair.

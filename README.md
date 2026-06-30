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
| Depth          | Depth Anything V2          | radial / luminance disparity    |
| Inpaint        | LaMa (big-lama)            | `cv2.inpaint` (Telea)           |
| Blur           | linear-light scatter (ours)| —  (always available, MPS-safe) |

So a first render works immediately; quality climbs as `setup.sh` finishes caching weights.

### Implementation notes (deviations from the brief)
- **Depth: Depth Anything V2, not Apple Depth Pro.** The brief's first pick was Depth Pro,
  but on a 16 GB M4 it took **60–130 s/render and leaked MPS memory** (degrading each run).
  Depth Anything V2 runs in **~0.2–1 s**, is stable, and — because depth here only grades the
  blur *falloff* (the clean edge comes from matte → decontaminate → inpaint) — costs nothing on
  the edge gate. Override the tier with `LENSY_DEPTH_MODEL`.
- **LaMa runs on CPU at reduced resolution.** `big-lama.pt` is CUDA-traced (won't load on a
  CUDA-less Mac) and uses FFT convs that are flaky on MPS, so it runs on CPU. Since the fill is
  only ever seen *blurred and behind* the sharp subject, it's computed at ≤768 px long edge and
  upscaled — invisible in the result, and the difference between a ~15 s and a ~4 s render.
- **End-to-end: ~7–8 s** for a 1000×1500 photo on an M4 (BiRefNet ~2 s, LaMa ~4 s, rest <1 s).

## Use it as a hosted app

The PWA is published to **GitHub Pages** at **https://chattedomestique.github.io/Lensy/** (a
GitHub Actions workflow builds `frontend/` and deploys on every push). Rendering still runs on
your Mac, so the hosted page reaches your backend over a **Cloudflare Tunnel**:

```bash
brew install cloudflared      # one time
./scripts/serve.sh            # runs the backend + opens a tunnel, prints a public HTTPS URL
```

Then open the app, click the **Server** pill (top-right) → **Connect to your render server**,
paste the `https://….trycloudflare.com` URL it printed, and hit **Connect**. The pill turns
green and you can upload → render → download from anywhere. Install it (browser “Add to Home
Screen / Install”) to get a standalone app icon.

> Quick-tunnel URLs change each run, so you re-paste after restarting `serve.sh`. For a URL that
> never changes, set up a Cloudflare **named tunnel** with your own domain and point the app at
> that once. The in-app setting is saved in `localStorage`, so it sticks between visits.

Local-only? Skip all of the above and just run `./scripts/dev.sh` → http://localhost:5173.

## The one inviolable rule (§7.1)

> matte → **decontaminate** foreground color → remove + inpaint the subject's hole → blur the *clean*
> background (depth-graded, linear light, scatter) → recomposite the sharp subject with premultiplied alpha.

"Blur the whole image then paste the subject back" is **forbidden** — it is the direct cause of edge halos
and chewed hair.

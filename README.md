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
GitHub Actions workflow builds `frontend/` and deploys on every push). Rendering runs on your
Mac, reached over a **Cloudflare named tunnel** at **https://lensy.sunhouse.media** — the hosted
app defaults to that URL, so it just connects.

**Server (your Mac), port `8842`:**
```bash
./scripts/serve.sh            # runs the render backend on :8842 (production)
```

**Tunnel (named, you manage cloudflared like your other tunnels).** Route
`lensy.sunhouse.media → http://localhost:8842`:

- *Dashboard / token-managed tunnel* (what `cloudflared tunnel run --token …` uses):
  Cloudflare **Zero Trust → Networks → Tunnels → your tunnel → Public Hostname → Add** —
  subdomain `lensy`, domain `sunhouse.media`, service **HTTP** `localhost:8842`. The running
  tunnel picks it up live; the DNS record is created for you.
- *Local config tunnel* (`config.yml`):
  ```yaml
  ingress:
    - hostname: lensy.sunhouse.media
      service: http://localhost:8842
    - service: http_status:404
  ```
  then `cloudflared tunnel route dns <tunnel-name> lensy.sunhouse.media`.

Then open the app — the **Server** pill (top-right) goes green and you can upload → render →
download from anywhere. Install it (browser “Add to Home Screen / Install”) for a standalone icon.

> The port is `LENSY_PORT` (default **8842**), used by the backend, `dev.sh`, and `serve.sh`.
> No named tunnel handy? `./scripts/serve.sh --quick` opens a throwaway `trycloudflare.com` URL —
> paste it into the Server pill (it's saved in `localStorage`). Local-only? `./scripts/dev.sh` →
> http://localhost:5173.

## The one inviolable rule (§7.1)

> matte → **decontaminate** foreground color → remove + inpaint the subject's hole → blur the *clean*
> background (depth-graded, linear light, scatter) → recomposite the sharp subject with premultiplied alpha.

"Blur the whole image then paste the subject back" is **forbidden** — it is the direct cause of edge halos
and chewed hair.

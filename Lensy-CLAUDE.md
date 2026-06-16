# CLAUDE.md — Lensy (standalone brief)
## Personal Tooling · PWA + Apple Silicon Server · 2026

> **This is a self-contained brief.** Drop it into a fresh repo as `CLAUDE.md` and it tells you
> everything needed to build Lensy — the app, the render pipeline, the models, the lens physics,
> and the look & feel. No other files required.

---

## 0. Philosophy

Lensy is a **private, local tool** — a personal "Portrait Mode" for still photos. Never publicly
distributed, never commercial. That buys pragmatism: ignore App Store / SaaS constraints, use whatever
model weights give the best result **regardless of license**, and spend all our care on making something
**powerful, beautiful, and delightful to use every day.**

What it does: take a photo → separate subject from background → render a **physically believable lens
blur** behind the subject — *without* the blur halo-ing, fringing, or chewing the subject's edges.
**That last part is the whole point. A clean edge is the product.**

Aesthetic north star: **Dieter Rams meets Teenage Engineering.** Purposeful, distinct, a little playful.
Never clinical. Tools that feel **handcrafted, not generated. Warm, not corporate.**

---

## 1. Shape of the System

A PWA front-end + a Python back-end that does the heavy lifting, joined over a Cloudflare Tunnel.
**Still photos only.**

```
┌──────────────┐    HTTPS     ┌─────────────────┐   CF Tunnel   ┌──────────────────────────┐
│  PWA (CF      │ ──────────▶ │  Cloudflare      │ ────────────▶ │  Apple Silicon server     │
│  Pages)       │  photo up   │  (CDN + Tunnel)  │  no open      │  FastAPI + PyTorch (MPS)  │
│  installable  │ ◀────────── │                  │  ports        │  render pipeline          │
└──────────────┘   result     └─────────────────┘ ◀──────────── └──────────────────────────┘
```

- **Front-end:** static, installable PWA served by **Cloudflare Pages**. Uploads a photo, shows live
  progress, presents the result with lens controls. All UI; zero ML.
- **Back-end:** **FastAPI** on an Apple Silicon Mac. Loads models once at startup (warm), runs the render
  pipeline, returns the composited image. PyTorch `device="mps"`; CoreML where it helps.
- **Transport:** `cloudflared` tunnel from the Mac — no public IP, no port-forwarding. Cloudflare's free
  proxy caps the request body at **100 MB** (fine for photos); still downscale huge inputs server-side.

---

## 2. Build System — Command Line Only

No IDE required to run, build, or modify anything. Everything is a script runnable from a terminal.

**Canonical scripts in `scripts/`:**
- **`setup.sh`** — create the Python venv, install deps, download + cache all model weights. Idempotent.
  Never ask the human to fetch weights by hand.
- **`dev.sh`** — the dev loop: start FastAPI (uvicorn `--reload`), the front-end dev server, and
  optionally the `cloudflared` tunnel, together. Ctrl-C stops all.
- **`build.sh`** — produce the production front-end bundle in `frontend/dist/` for Cloudflare Pages, and
  verify the backend imports + loads models clean.

`./scripts/setup.sh && ./scripts/dev.sh` from zero must yield a working local Lensy.

**Stacks:** Front-end = **Vite** (vanilla TS, no heavy framework unless needed) + `vite-plugin-pwa`.
Back-end = Python 3.11+, **FastAPI** + **uvicorn**, `pyproject.toml`, project-local `.venv/`, PyTorch MPS.

---

## 3. Project Structure

```
Lensy/
├── CLAUDE.md                  ← this file
├── frontend/                  ← PWA (Vite)
│   ├── index.html
│   ├── manifest.webmanifest
│   ├── src/{main.ts, controls.ts, api.ts, sw.ts, styles/}
│   └── public/icons/          ← maskable + any-purpose PWA icons
├── backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI app, lifespan = warm model load
│   │   ├── api.py             ← POST /render, GET /healthz, progress (SSE/WebSocket)
│   │   └── pipeline/
│   │       ├── matte.py       ← BiRefNet (+ optional ViTMatte refine)
│   │       ├── refine.py      ← guided filter + pymatting decontamination
│   │       ├── depth.py       ← Apple Depth Pro
│   │       ├── inpaint.py     ← LaMa (cv2.inpaint fallback)
│   │       ├── blur.py        ← linear-light scatter lens-blur renderer
│   │       └── compose.py     ← premultiplied OVER, edge feather
│   ├── pyproject.toml
│   └── models/                ← weight cache (git-ignored)
├── scripts/                   ← setup.sh, dev.sh, build.sh
└── build/ , .venv/ , dist/    ← git-ignored
```

---

## 4. App Architecture

**Front-end (PWA):** installable day one — valid `manifest.webmanifest`, maskable icons, a service
worker caching the app shell (opens instantly, offline shell; rendering needs the server). Flow:
drop/select photo → preview → tweak controls → `POST /render` → progress → result with a **before/after**
compare (draggable divider) and download/save. Controls mirror the lens model (§7): blur strength `K`,
focal plane (`disp_focus`, ideally tap-to-focus on the preview), aperture blade count, highlight boost.
Sensible defaults so a first render needs zero fiddling.

**Back-end (FastAPI):** models load **once** in the lifespan handler and stay warm — never per request.
`POST /render` runs the pipeline (§7); stream stage progress over **SSE or WebSocket** (don't poll).
`GET /healthz` reports model-loaded state. Friendly error envelopes, never a bare 500 with a stack.

**No polling:** front-end gets progress via a server-pushed stream; back-end uses `async`, not sleep
loops. If something must poll, back off when idle, never faster than ~1s.

---

## 5. Aesthetics & Design System  *(the north star)*

This part of the house style transcends platform. Express it in CSS custom properties + modern web idioms.

**Philosophy** — synthesize into every surface: **Rams** (purposeful, uncluttered, honest) ·
**Teenage Engineering** (warm, tactile, distinct, playful type & color) · **Neo-brutalism 2025–26**
(confident strokes, visible structure; not sterile) · **2026 platform polish** (glass/vibrancy, depth,
adaptive expressive color). Result: **handcrafted, not generated.**

**Typography — pair serif + sans (hard rule).**
| Role | Web choice |
|---|---|
| Display / hero | a display **serif** (Canela/Freight/New York feel; `Georgia` fallback) |
| Body / UI | a clean **sans** (Inter / Geist / `-apple-system`) |
| Mono / data | SF Mono / Geist Mono / Berkeley Mono |

Hierarchy ≥4 levels: **Display** 28–48px serif medium/semibold · **Title** 18–24px sans semibold ·
**Body** 14–16px sans regular · **Caption** 11–12px sans reduced opacity. Tighten display tracking
(−0.02 to −0.04em), loosen captions (+0.02em). Line-height 1.4–1.6 body, 1.1–1.2 display.

**Color — never pure black or pure white; always tint.** Lensy is a photography tool: a warm,
gallery-neutral base that lets the *photo* be the loud thing, with one confident accent. Light + dark
from day one (deliberate, not an inversion). Respect `prefers-color-scheme`.

```css
:root {
  --bg:#f6f4ef; --surface:#fffdf8; --text:#1e1b18; --subtle:#8c867d;
  --accent:#ec734a;     /* terracotta — the one loud color */
  --accent-2:#2f6f8f;   /* cool counterpoint for focus/links */
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#16140f; --surface:#1f1c16; --text:#f2ede3; --subtle:#9b958a; --accent:#ff8a5c; }
}
```

**Shape & layout** — radii generous (10–16px panels, 6–8px buttons, 4px inputs); spacing scale
4/8/12/16/24/32/48 (stick to it); shadows soft, warm-tinted, 2–3 layers, never pure black; borders
0.5–1px semi-transparent. A little visible structure (neo-brutalist confidence) on key panels is welcome.

**Motion** — default spring feel `cubic-bezier(.2,.8,.2,1)` ~320ms; a micro-interaction on **every**
interactive element (hover, press, focus-visible); prefer combined opacity+scale over slide-ins; never
exceed ~400ms for a UI response; honor `prefers-reduced-motion`.

**Iconography** — lean linear icons for chrome, filled for active/selected; minimal, characterful.
Ship the PWA app icon day one (playful, warm), maskable + any-purpose at 192 / 512.

---

## 6. Code Quality

**Python:** type hints everywhere; `async def` handlers; Pydantic request/response models. Pipeline
stages are pure-ish functions (array in / array out, premultiplied RGBA where relevant) so each is
testable alone. Work in **linear light** inside the pipeline; convert sRGB↔linear only at the boundaries.
Raise typed exceptions → friendly JSON; log detail with `logging`; never swallow errors.

**TypeScript:** strict; small named modules; rendering/state logic separate from DOM glue; no
`console.log` debris.

**Performance (Apple Silicon):** models warm-loaded once, never per request. Process at a working
resolution (long edge ~1536–2048), composite the matte back at full res for output. CPU-heavy stages off
the event loop (`run_in_executor`); MPS work batched where possible. Profile before optimizing.
⚠️ **Verify before relying on it:** some renderers (e.g. BokehMe) use custom **CUDA** kernels that won't
run on MPS — the hand-rolled OpenCV/linear-light scatter renderer in `blur.py` is the MPS-safe
**primary**; CUDA-bound options are opt-in upgrades only after confirming they run on Metal.

---

## 7. The Render Pipeline  *(this is the product — read carefully)*

### 7.1 The one inviolable rule
Both failure modes — the subject's color smearing outward as a **halo**, and the background blur **eating
hair/edges** — share one cause: **blurring across the occlusion boundary**, where each edge pixel is
already a mix `C = αF + (1−α)B`. A kernel straddling the silhouette averages the two layers. The fix is an
**ordering**, and it must never be violated:

> **matte → decontaminate foreground color → remove + inpaint the subject's hole → blur the *clean*
> background (depth-graded, linear light, scatter) → recomposite the sharp subject with premultiplied alpha.**

**"Blur the whole image then paste the subject back" is FORBIDDEN** — it is the direct cause of edge
halos and chewed hair (the blurred background already contains averaged-in subject color outside the
silhouette, and pulled background color inward over the hair).

### 7.2 Stages, in order
1. **Matte** — **BiRefNet** (HR matting variant) → soft alpha `α`. Optional **ViTMatte** trimap refine
   (build a trimap from `α`) to recover hair wisps. *Must be a soft alpha, not a hard mask.*
2. **Refine** — OpenCV `cv2.ximgproc.guidedFilter(guide=RGB, src=α)` snaps the matte to true image edges
   (matting-Laplacian link), no halos.
3. **Decontaminate** — pymatting `estimate_foreground_ml(image, α)` → true foreground `F`.
   **This is the single most important anti-halo step** — it removes background-tinted color from edge pixels.
4. **Depth** — **Apple Depth Pro** → depth/disparity. Chosen for **best-in-class boundary accuracy + thin
   structure (hair) recall** — exactly what protects the edge. Native to Apple Silicon.
5. **Inpaint** — **LaMa** (big-lama) fills the subject's hole so the background blur near the silhouette
   only averages *real* background. `cv2.inpaint` is the cheap fallback.
6. **Blur** — depth-graded **scatter** renderer (§7.3).
7. **Compose** — premultiplied `F·α` **over** the blurred background; apply a **mild** blur to slightly
   off-focal-plane subject regions to defeat the "sticker" cutout look.

Validate edge quality on **real hair photos** at every step — that's the acceptance bar.

### 7.3 Lens-blur physics (what makes it read as a *lens*, not a Gaussian smudge)
1. **Work in linear light, ideally HDR.** Convert sRGB→linear *before* blurring, re-encode after — without
   this you get no bokeh balls; bright highlights must stay bright as they spread.
2. **Scatter, not gather, for highlights.** An out-of-focus point light physically *spreads its energy*
   into an aperture-shaped disk. Gather averages highlights down and kills the bokeh balls. Trick: detect
   bright spots → splat aperture-shaped sprites; cheap gather for the rest.
3. **Signed Circle of Confusion from depth:** `CoC = K · (disparity − disp_focus)`. Sign = foreground vs
   background (drives occlusion order); `K` = blur-strength slider; `disp_focus` = focal plane.
4. **Energy-conserving accumulation:** each splat normalized by kernel area (`1/πr²`); accumulate
   `color·weight` and `weight` separately, then divide. Premultiplied. Prevents brightening/darkening as
   CoC changes.
5. **Layer by depth, composite back-to-front** so near/far never pre-contaminate each other.
6. **Aperture = the splat kernel:** disk (circular) or N-gon for N iris blades → polygonal bokeh.
7. **Lens character (optional polish):** optical vignetting → **cat's-eye/swirly** bokeh (vary kernel
   shape with image radius); **anamorphic** (oval kernel via squeeze factor); **bokeh fringing/CA**
   (per-channel CoC); **bloom/busy bokeh** (emerges from linear-HDR highlight handling).

### 7.4 Model & tool reference (personal use → license is informational only)
**Matting:** BiRefNet ⭐ (MIT, soft alpha, MPS) · ViTMatte (MIT, trimap refiner) ·
transparent-background/InSPyReNet (MIT, easy) · rembg (MIT wrapper) · BEN2 · MODNet · BRIA RMBG-2.0 ·
SAM/SAM2 (hard masks → trimap only). *(RVM/MatAnyone are for video — not needed, stills only.)*
**Depth:** Apple Depth Pro ⭐ (best edges) · Depth Anything V2/V3 · Marigold (Apache, diffusion) ·
Lotus (Apache, +normals) · Metric3D v2 (metric+normals) · MiDaS/ZoeDepth (legacy/fast).
**Bokeh renderers:** custom OpenCV linear-light scatter ⭐ (MPS-safe primary) · BokehMe (Apache; ⚠️ CUDA
kernel — verify on MPS) · BokehMe++ (cat-eye, alpha input) · Dr.Bokeh (occlusion-aware) ·
BokehDiff (ICCV 2025, diffusion, heavy) · three.js BokehShader (if blur ever moves client-side).
**Edge toolkit:** pymatting (MIT, foreground decontamination) · OpenCV ximgproc guidedFilter (Apache) ·
premultiplied-alpha compositing · LaMa inpaint · libcom (optional harmonization).

---

## 8. Decision Autonomy

Decide for yourself: file/module naming within this structure; front-end component breakdown; exact
palette values within §5's warm-neutral + one-accent direction; whether a stage is pure Python, NumPy, or
a model; SSE vs WebSocket for progress; bundler/plugin specifics. **Ask the human only when:** a choice
fundamentally changes scope/behavior; two genuinely valid approaches differ meaningfully (e.g. swapping a
core model); or a secret/system access (Cloudflare token) must be surfaced. Otherwise: make the call,
build it, validate it.

---

## 9. Checklist — Before Declaring a Build Done

- [ ] `./scripts/setup.sh` then `./scripts/dev.sh` runs clean from zero — backend warm, front-end up.
- [ ] PWA installs (manifest valid, service worker registers, opens offline shell, icon renders).
- [ ] A real photo round-trips: upload → render → before/after → download.
- [ ] **Edges are clean** on a hard test (flyaway hair, glasses, fur): no halo, no fringe, no chewed edge,
      no "sticker" cutout. **This is the gate — fail it and the build is not done.**
- [ ] Bokeh reads as a *lens*: highlights bloom into aperture-shaped balls; blur grades with depth.
- [ ] Light and dark modes both look intentional (not an inversion).
- [ ] Every interactive element has hover / press / focus-visible states; `prefers-reduced-motion` honored.
- [ ] Typography pairs serif + sans with a clear ≥4-level hierarchy.
- [ ] Friendly errors surfaced to the UI; technical detail only in logs; no `print`/`console.log` debris.
- [ ] Models load once and stay warm; a render at working res returns in a few seconds on Apple Silicon.
- [ ] First-run is smooth — no manual setup steps, no weight-fetching instructions.

---

*Self-contained baseline for Lensy. The edge-quality rule in §7 and the aesthetic in §5 do not bend.*

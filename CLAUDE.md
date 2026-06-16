# CLAUDE.md
## Lensy — Build Guidelines
### Personal Tooling · PWA + Apple Silicon Server · 2026

> Companion doc: **`docs/research.md`** holds the full pipeline research, model choices, and
> the lens-blur physics. *This* file tells you how to build the app and how it should feel.
> When the two overlap, `research.md` is authoritative on the *what* (algorithms/models),
> this file is authoritative on the *how* (structure, craft, aesthetics).

---

## 0. Philosophy

Lensy is a **private, local tool** — a personal "Portrait Mode" for still photos. It is never
publicly distributed and never commercial. That buys us pragmatism: ignore App Store / SaaS
constraints, use whatever model weights give the best result regardless of license, and spend
all our care on making something **powerful, beautiful, and delightful to use every day.**

What Lensy does: take a photo → separate subject from background → render a **physically
believable lens blur** behind the subject — *without* the blur halo-ing, fringing, or chewing
the subject's edges. That last part is the whole point. A clean edge is the product.

The aesthetic north star: **Dieter Rams meets Teenage Engineering.** Purposeful, distinct, a
little playful. Never clinical. Tools that feel **handcrafted, not generated.**

---

## 1. Shape of the System

A PWA front-end and a Python back-end that does the heavy lifting, joined over a Cloudflare
Tunnel. Stills only.

```
┌──────────────┐    HTTPS     ┌─────────────────┐   CF Tunnel   ┌──────────────────────────┐
│  PWA (CF      │ ──────────▶ │  Cloudflare      │ ────────────▶ │  Apple Silicon server     │
│  Pages)       │  photo up   │  (CDN + Tunnel)  │  no open      │  FastAPI + PyTorch (MPS)  │
│  installable  │ ◀────────── │                  │  ports        │  render pipeline          │
└──────────────┘   result     └─────────────────┘ ◀──────────── └──────────────────────────┘
```

- **Front-end:** static, installable PWA served by **Cloudflare Pages**. Uploads a photo, shows
  live progress, presents the result with the lens controls. All UI; zero ML.
- **Back-end:** **FastAPI** on an Apple Silicon Mac. Loads models once at startup (warm), runs the
  render pipeline, returns the composited image. PyTorch `device="mps"`; CoreML where it helps.
- **Transport:** `cloudflared` tunnel from the Mac — no public IP, no port-forwarding. CF's free
  proxy caps the request body at **100 MB** (fine for photos); still downscale huge inputs server-side.

---

## 2. Build System — Command Line Only

No IDE required to run, build, or modify anything. Everything is a script you can run from a terminal.

### 2.1 Canonical scripts (project root `scripts/`)
- **`scripts/setup.sh`** — creates the Python venv, installs deps, downloads + caches all model
  weights. Idempotent: safe to re-run. Never ask the human to fetch weights by hand.
- **`scripts/dev.sh`** — the dev loop. Starts the FastAPI backend (uvicorn `--reload`), the
  front-end dev server, and (optionally) the `cloudflared` tunnel, all together. Ctrl-C stops all.
- **`scripts/build.sh`** — produces the production front-end bundle in `frontend/dist/` ready for
  Cloudflare Pages, and verifies the backend imports + loads models clean.

Running `./scripts/setup.sh && ./scripts/dev.sh` from zero must give a working local Lensy.

### 2.2 Stacks
- **Front-end:** **Vite** (vanilla TS, no heavy framework unless a need appears) + `vite-plugin-pwa`
  for the service worker and manifest. Keep the bundle small and the code legible.
- **Back-end:** Python 3.11+, **FastAPI** + **uvicorn**, a `pyproject.toml` (or `requirements.txt`),
  all deps in a project-local `.venv/`. PyTorch with MPS.

### 2.3 No incremental cruft
Treat the dev servers as the loop. `setup.sh` re-run must converge; don't accumulate half-states.

---

## 3. Project Structure

```
Lensy/
├── CLAUDE.md                  ← this file
├── docs/
│   └── research.md            ← pipeline research, models, physics (the "what")
├── frontend/                  ← PWA (Vite)
│   ├── index.html
│   ├── manifest.webmanifest
│   ├── src/
│   │   ├── main.ts            ← app shell, upload, result view
│   │   ├── controls.ts        ← blur K, focal plane, aperture blades, highlight
│   │   ├── api.ts             ← talks to the backend
│   │   ├── styles/            ← design tokens + components (see §5)
│   │   └── sw.ts              ← service worker (offline shell)
│   └── public/icons/          ← PWA icons (maskable + any-purpose)
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

### 4.1 Front-end (PWA)
- **Installable from day one:** valid `manifest.webmanifest`, maskable icons, a service worker that
  caches the app shell so it opens instantly and works offline (the *shell*; rendering needs the server).
- **Flow:** drop/select a photo → preview → tweak controls → `POST /render` → progress → result with a
  **before/after** compare and a download/save. Make the compare gesture feel good (drag a divider).
- **Controls** mirror the lens model in `research.md`: blur strength `K`, focal plane (`disp_focus`,
  ideally tap-to-focus on the preview), aperture blade count, highlight boost. Sensible defaults so a
  first render needs zero fiddling.

### 4.2 Back-end (FastAPI)
- Models load **once** in the app lifespan handler and stay warm. Never load per-request.
- `POST /render` runs the pipeline (§7) and returns the image. Long enough that you want **progress**:
  stream stage updates over **SSE or WebSocket** — do not make the client poll.
- `GET /healthz` reports model-loaded state. Friendly error envelopes, never a bare 500 with a stack.

### 4.3 No polling
Same rule as the broader tooling ethos: don't busy-poll for state. Front-end gets progress via a
server-pushed stream; back-end uses `async`/awaitables, not sleep-loops. If something must poll, back
off when idle and never faster than ~1s.

---

## 5. Aesthetics & Design System  *(the north star — translated to web)*

This is the part of our house style that transcends platform. Carry it over faithfully; express it in
CSS custom properties and modern web idioms.

### 5.1 Design philosophy
Synthesize into every surface:
- **Dieter Rams** — purposeful, uncluttered, honest.
- **Teenage Engineering** — warm, tactile, distinct personality, playful type & color.
- **Neo-brutalism (2025–26 inflection)** — confident strokes, visible structure; not sterile.
- **2026 platform polish** — glass/vibrancy, depth, spatial awareness; adaptive expressive color.

Result: **handcrafted, not generated. Warm, not corporate.**

### 5.2 Typography — pair serif + sans (hard rule)
| Role | Web choice |
|---|---|
| Display / hero | A display **serif** — New York / Canela / Freight feel; web: a self-hosted serif (e.g. a Freight/Canela-like or `Georgia` fallback) |
| Body / UI | A clean **sans** — Inter / Geist / system `-apple-system` |
| Mono / data | SF Mono / Geist Mono / Berkeley Mono |

Hierarchy, minimum 4 levels: **Display** 28–48px serif medium/semibold · **Title** 18–24px sans
semibold · **Body** 14–16px sans regular · **Caption** 11–12px sans reduced opacity. Tighten display
tracking (−0.02 to −0.04em), loosen captions (+0.02em). Line-height 1.4–1.6 body, 1.1–1.2 display.

### 5.3 Color — never pure black or pure white; always tint
Define a per-project palette as CSS custom properties, light + dark from day one. Lensy is a
**photography** tool: a warm, gallery-neutral base that lets the *photo* be the loud thing, with one
confident accent.

```css
:root {
  --bg:      #f6f4ef;  /* warm off-white, never #fff */
  --surface: #fffdf8;  /* raised cards */
  --text:    #1e1b18;  /* warm near-black, never #000 */
  --subtle:  #8c867d;  /* warm gray */
  --accent:  #ec734a;  /* terracotta — the one loud color */
  --accent-2:#2f6f8f;  /* cool counterpoint for focus/links */
  --shadow:  28 24 18; /* warm shadow rgb, used at low alpha, 2–3 layers */
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#16140f; --surface:#1f1c16; --text:#f2ede3; --subtle:#9b958a; --accent:#ff8a5c; }
}
```
Support dark mode via `prefers-color-scheme` and design both deliberately — not an inversion.

### 5.4 Shape & layout
- **Radii:** generous — 10–16px cards/panels, 6–8px buttons, 4px inputs.
- **Spacing scale:** 4 / 8 / 12 / 16 / 24 / 32 / 48 — stick to it.
- **Shadows:** soft, warm-tinted, 2–3 layers; never pure black.
- **Borders:** 0.5–1px semi-transparent — define without shouting. A little visible structure
  (neo-brutalist confidence) is welcome on key panels.

### 5.5 Motion & interaction
- Default spring feel: `transition` / Web Animations tuned like
  `cubic-bezier(.2,.8,.2,1)` ~320ms — the CSS analogue of `spring(response:0.35, damping:0.72)`.
- A micro-interaction on **every** interactive element (hover, press, focus-visible).
- Prefer combined **opacity + scale** over slide-ins. Never exceed ~400ms for a UI response.
- Respect `prefers-reduced-motion`.

### 5.6 Iconography
- Lean, linear icons for chrome; filled for active/selected. Minimal, characterful, recognizable small.
- The PWA app icon ships day one — playful, warm, matches Lensy's personality. Provide maskable +
  any-purpose at the standard PWA sizes (192, 512, and a maskable variant).

---

## 6. Code Quality & Patterns

### 6.1 Python (back-end)
- Type hints everywhere; `async def` handlers; Pydantic models for request/response.
- Pipeline stages are **pure-ish functions** with clear array in / array out (`np.ndarray`,
  premultiplied RGBA where relevant) so each stage is testable in isolation.
- Work in **linear light** inside the pipeline; convert sRGB↔linear only at the boundaries (see physics in `research.md`).
- Errors: raise typed exceptions, map to friendly JSON envelopes; log technical detail with `logging`.
  Never swallow an error silently.

### 6.2 TypeScript (front-end)
- Strict TS. Small, named modules (`api.ts`, `controls.ts`). No God-objects.
- Keep the Swift⇄JS-style discipline: rendering/state logic is clean and separate from DOM glue.
- No `console.log` debris in committed code.

### 6.3 Performance (Apple Silicon)
- Models warm-loaded once; **never** reload per request.
- Process at a sane working resolution (long edge ~1536–2048), then composite the matte back at full
  res for the final output. Profile before optimizing.
- CPU-heavy stages off the event loop (`run_in_executor` / worker), MPS work batched where possible.
- ⚠️ **Verify before relying on it:** some renderers (e.g. BokehMe) use custom **CUDA** kernels that
  won't run on MPS. The hand-rolled OpenCV/linear-light scatter renderer in `blur.py` is the MPS-safe
  **primary**; treat CUDA-bound options as opt-in upgrades only after confirming they run on Metal.

---

## 7. The Render Pipeline (must-follow ordering)

Full detail, model rationale, and the lens-blur math live in **`docs/research.md`**. The one rule that
must never be violated, because it *is* the product:

> **matte → decontaminate foreground color → remove + inpaint the subject's hole → blur the *clean*
> background (depth-graded, linear light, scatter) → recomposite the sharp subject on top with
> premultiplied alpha.**

"Blur the whole image then paste the subject back" is **forbidden** — it is the direct cause of edge
halos and chewed hair. Stages, in order:

1. **Matte** — BiRefNet (HR matting) → soft alpha `α`; optional ViTMatte trimap refine for hair.
2. **Refine** — guided filter (guide = RGB) snaps `α` to true edges.
3. **Decontaminate** — pymatting `estimate_foreground_ml` → true foreground `F` (the key anti-halo step).
4. **Depth** — Apple Depth Pro → depth/disparity (best boundary + hair recall; protects the edge).
5. **Inpaint** — LaMa fills the subject hole so blur near the silhouette only averages real background.
6. **Blur** — depth-graded **scatter** in linear/HDR light: signed CoC from `(disparity − disp_focus)·K`,
   energy-conserving premultiplied accumulation, aperture-shaped kernel (blade count), highlight boost,
   optional optical-vignetting (cat's-eye) and bloom.
7. **Compose** — premultiplied `F·α` **over** the blurred background; mild feather on slightly
   off-focal-plane subject regions to defeat the "sticker" cutout look.

Validate edge quality on **real hair photos** at every step — that's the acceptance bar.

---

## 8. Decision Autonomy

Make your own call on: file/module naming within this structure; front-end component breakdown;
exact palette values within §5.3's warm-neutral + one-accent direction; whether a stage is pure Python,
NumPy, or a model; SSE vs WebSocket for progress; bundler/plugin specifics.

**Ask the human only when:** a choice fundamentally changes scope/behavior; there are two genuinely
valid approaches with meaningfully different tradeoffs (e.g. swapping a core model); or a system access /
secret (Cloudflare token, etc.) needs surfacing. Otherwise: make the call, build it, validate it.

---

## 9. Checklist — Before Declaring a Build Done

- [ ] `./scripts/setup.sh` then `./scripts/dev.sh` runs clean from zero — backend warm, front-end up.
- [ ] PWA installs (manifest valid, service worker registers, opens offline shell, icon renders).
- [ ] A real photo round-trips: upload → render → before/after → download.
- [ ] **Edges are clean** on a hard test (flyaway hair, glasses, fur): no halo, no fringe, no chewed edge,
      no "sticker" cutout. This is the gate — fail it and the build is not done.
- [ ] Bokeh reads as a *lens*: highlights bloom into aperture-shaped balls, blur grades with depth.
- [ ] Light and dark modes both look intentional (not an inversion).
- [ ] Every interactive element has hover / press / focus-visible states; `prefers-reduced-motion` honored.
- [ ] Typography pairs serif + sans with a clear ≥4-level hierarchy.
- [ ] Friendly errors surfaced to the UI; technical detail only in logs; no `print`/`console.log` debris.
- [ ] Models load once and stay warm; a render at working res returns in a few seconds on Apple Silicon.
- [ ] First-run is smooth — no manual setup steps, no weight-fetching instructions.

---

*This document is the baseline for Lensy. `docs/research.md` is its technical companion. Extend either
with project notes as the app grows — but the edge-quality rule in §7 and the aesthetic in §5 do not bend.*

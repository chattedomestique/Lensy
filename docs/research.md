# Lensy — Research & Architecture

> Background separation + realistic lens blur (synthetic depth-of-field / "Portrait Mode")
> for a personal-use PWA, processed server-side on Apple Silicon.

**Context for every decision in this doc:**

| Constraint | Value | Consequence |
|---|---|---|
| Use | **100% personal** | License is **not** a constraint. Non-commercial (CC-BY-NC), research-only (Apple AMLR, NTU S-Lab), and GPL models are all fair game. Pick on quality. |
| Hosting | **PWA** behind **Cloudflare** | Static frontend; all heavy lifting on the server. |
| Processing | **Server-side**, **Apple Silicon (M-series)** | PyTorch **MPS** + CoreML. Some ops fall back to CPU; custom CUDA kernels are a risk — flagged below. |
| Media | **Still photos only** | No temporal/video matting needed. Simpler, highest quality per image. |

---

## 1. The core problem (why naive blur wrecks subject edges)

Both failure modes you want to avoid —

1. the subject's color smearing outward as a **halo**, and
2. the background blur **eating into hair / edges**

— have the **same root cause**: blurring *across the occlusion boundary*. Every edge pixel is already a
mixture of foreground and background, `C = αF + (1−α)B`. A blur kernel that straddles the silhouette
averages the two layers together. ([alpha compositing](https://en.wikipedia.org/wiki/Alpha_compositing),
[Dr.Bokeh, CVPR 2024](https://arxiv.org/abs/2308.08843))

**The fix is an ordering, and it is non-negotiable regardless of which models we use:**

> **matte → decontaminate foreground color → remove + inpaint the subject's hole → blur the *clean* background (depth-graded) → recomposite the sharp subject on top using premultiplied alpha.**

"Blur the whole image, then paste the sharp subject back" is the **wrong** order and is the direct cause of
the artifacts. The correct "inpaint-as-input" ordering is what Google's Portrait Mode and the modern bokeh
papers all formalize. ([Bokehlicious](https://arxiv.org/html/2503.16067v1),
[Google Synthetic DoF, SIGGRAPH 2018](https://arxiv.org/abs/1806.04171))

Why "blur first, then mask" fails, concretely:
- The blurred background near the silhouette already contains averaged-in **subject** color (the halo).
  Pasting the sharp subject on top does not remove that contaminated ring just *outside* the subject.
- The blur also pulled background color **inward** over the hair, so even a perfect matte has already lost
  fine hair pixels.

---

## 2. Final chosen stack (personal use · Apple Silicon · stills)

Because license no longer constrains us, this is a **quality-first** stack. Everything below runs on
Apple Silicon (MPS / CoreML), with flagged fallbacks where a model leans on CUDA.

```
Input photo
   │
   ▼
[1] MATTE ........... BiRefNet (HR matting)  →  ViTMatte trimap refine (hair)     → soft alpha α
   │
   ▼
[2] REFINE ......... guided filter (guide = RGB, src = α)                          → snapped α
   │
   ▼
[3] DECONTAMINATE .. pymatting estimate_foreground_ml(image, α)                    → true foreground F
   │
   ▼
[4] DEPTH .......... Apple Depth Pro  (sharpest edges + best hair recall)          → depth / disparity
   │
   ▼
[5] INPAINT ........ LaMa behind the subject hole                                  → clean background B*
   │
   ▼
[6] BLUR ........... depth-graded scatter renderer in linear/HDR light             → bokeh background
   │                 (BokehMe if MPS-viable, else custom OpenCV scatter)
   ▼
[7] COMPOSITE ...... premultiplied (F·α) OVER blurred B*, mild edge feather        → final image
```

### Why each choice

| Stage | Choice | License | Why it wins here |
|---|---|---|---|
| Matte | **BiRefNet** (`BiRefNet_HR-matting`, `BiRefNet_dynamic-matting`) | MIT | SOTA high-res dichotomous matting, **true soft alpha**, runs on MPS, ~2048² HR variant for fine detail. |
| Matte refine | **ViTMatte** (small/base) | MIT | Trimap-based detail refiner — feed it a trimap derived from BiRefNet's alpha to recover hair wisps. Optional but high payoff. |
| Edge snap | **OpenCV `ximgproc.guidedFilter`** (guide = image) | Apache-2.0 | Snaps matte to true image edges (matting-Laplacian link), no halos. Fast guided filter for speed. |
| Decontaminate | **pymatting `estimate_foreground_ml`** | MIT | Recovers true `F` so subject stops smearing background-tinted edge color into the bokeh. **The single most important anti-halo step.** |
| Depth | **Apple Depth Pro** | research-only weights (fine for personal use) | **Best-in-class boundary accuracy** + highest thin-structure (hair/fur) recall, eliminates "flying pixels" — exactly what protects edges. Native Apple Silicon. Metric scale without intrinsics. |
| Inpaint | **LaMa** (big-lama) | Apache-2.0 | Fast, high-quality hole fill so blur near the silhouette only averages *real* background. `cv2.inpaint` is the cheap fallback. |
| Blur | **BokehMe** (primary) / **custom OpenCV scatter** (fallback) | Apache-2.0 / own | Scatter renderer with K / focus / gamma / highlight / aperture controls; neural fix for imperfect depth. |
| Composite | premultiplied `over` | — | Clean edges; mild feather on slightly-off-focal-plane subject regions defeats the "sticker" look. |

> **Apple Silicon flag:** BokehMe's classical renderer historically uses a custom **CUDA** scatter
> extension. On MPS it may need the CPU fallback path or a reimplementation. **Mitigation:** the
> hand-rolled OpenCV/Metal scatter renderer (Section 4) is the MPS-safe primary; treat BokehMe as the
> "upgrade when proven on MPS" option. Verify before committing to it.

### Optional upgrades (have the GPU budget / want max realism)
- **BokehDiff** (ICCV 2025) — 1-step diffusion, physics-constrained CoC + self-occlusion. Runs on MPS but slow; best realism.
- **Dr.Bokeh** — differentiable, occlusion-aware; solves color-bleed *in the render stage*.
- **MPIB** — multiplane-image bokeh built specifically for partial occlusion, with its own inpainting module.
- **Marigold / Lotus** — diffusion depth alternatives to Depth Pro (Apache-2.0) if you prefer.

---

## 3. The physics recipe for realistic lens character

Do this regardless of renderer. This is what makes the blur read as a *lens*, not a Gaussian smudge.

1. **Work in linear light, ideally HDR.** Convert sRGB → linear **before** blurring, re-encode after.
   Without this you get no bokeh balls — bright highlights must stay bright as they spread.
   ([MJP, "Bokeh"](https://therealmjp.github.io/posts/bokeh/))
2. **Scatter, not gather, for highlights.** An out-of-focus point light physically *spreads its energy*
   into an aperture-shaped disk. Gather averages highlights down and kills the bokeh balls. Production
   trick: detect bright spots → splat aperture-shaped sprites; cheap gather for the rest.
   ([BokehMe](https://arxiv.org/abs/2206.12614))
3. **Signed Circle of Confusion (CoC) from depth.** Thin-lens: `1/u + 1/v = 1/f`, aperture diameter
   `A = f / N`. For artistic control most tools use a signed disparity form:
   `CoC = K · (disparity − disp_focus)` — sign = foreground vs background (drives occlusion order),
   `K` = blur strength slider, `disp_focus` = focal plane.
   ([Circle of confusion](https://en.wikipedia.org/wiki/Circle_of_confusion))
4. **Energy-conserving accumulation.** Each splat normalized by kernel area (`1/πr²`); accumulate
   `color·weight` and `weight` separately, then divide. Premultiplied. Prevents brightening/darkening as
   CoC changes. ([MJP](https://therealmjp.github.io/posts/bokeh/))
5. **Layer by depth, composite back-to-front** so near/far never pre-contaminate each other (the
   production generalization of the 2-layer subject/background split).
6. **Aperture = the splat kernel.** Disk (circular) or N-gon for N iris blades → polygonal bokeh.

### Lens-character extras ("accurate modeling of lens characteristics")
- **Cat's-eye / swirly bokeh** (optical vignetting): off-axis highlights clipped into lemon shapes toward
  the corners. Implement by varying splat kernel shape with image radius. **BokehMe++** models this via
  lens-barrel front length. ([BokehMe++, TPAMI 2024](https://ieeexplore.ieee.org/document/10756626/))
- **Anamorphic bokeh:** oval highlights via a squeeze/aspect factor on the kernel.
- **Bokeh fringing / chromatic aberration:** per-channel CoC or RGB bokeh textures.
- **Bloom / busy bokeh:** emerges naturally from the linear-HDR + highlight handling.

---

## 4. Edge-artifact prevention toolkit (the critical requirement)

All permissive, all CPU/MPS-friendly:

1. **Refine the matte to real edges** — `cv2.ximgproc.guidedFilter(guide=RGB, src=alpha, radius, eps)`.
   Recovers hair wisps, removes blockiness, no halos (matting-Laplacian connection). Fast Guided Filter
   gives >10× speedup. ([He et al.](https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf))
2. **Decontaminate foreground color** — pymatting:
   ```python
   from pymatting import estimate_foreground_ml
   F = estimate_foreground_ml(image, alpha)   # CPU/CUDA/OpenCL; MIT
   ```
   Recovers true `F` at edge pixels. Most important anti-halo step. ([FMLFE, ICPR 2020](https://arxiv.org/abs/2006.14970))
3. **Always filter in premultiplied alpha** — multiply RGB by α before any blur/resize/feather;
   un-premultiply only at final display. Removes a whole class of color fringing.
4. **Inpaint the hole** behind the subject before blurring (LaMa for quality, `cv2.inpaint` cheap) so the
   background blur near the silhouette averages only real background.
5. **Recomposite** premultiplied `F·α` over the blurred background with `over`; apply **mild** blur to
   slightly-off-focal-plane subject regions to defeat the cutout/"sticker" look.
6. **(Optional) Harmonize** subject to background lighting with **libcom** (Apache-2.0).

Key libraries: **pymatting** (MIT, v1.1.15 Jan 2026, CPU/CUDA/OpenCL),
**OpenCV ximgproc guidedFilter** (Apache-2.0), **libcom** (Apache-2.0).

---

## 5. System architecture (PWA + server + Cloudflare)

```
┌─────────────┐     HTTPS      ┌──────────────────┐   Cloudflare Tunnel   ┌────────────────────────┐
│  PWA (CF)   │  ───────────▶  │   Cloudflare      │  ───────────────────▶ │  Your Apple Silicon    │
│ static SPA  │   upload img   │   (CDN + Tunnel)  │   no exposed ports    │  server (FastAPI)      │
│ installable │  ◀───────────  │                   │  ◀─────────────────── │  PyTorch MPS pipeline  │
└─────────────┘  result image  └──────────────────┘                       └────────────────────────┘
```

- **Frontend (PWA):** static SPA served via Cloudflare Pages. Manifest + service worker for installability
  and offline shell. Uploads the photo, shows progress, displays/downloads the result. Optional client-side
  controls: blur strength (`K`), focal plane, aperture blades, highlight boost.
- **Backend:** Python **FastAPI** service. One `POST /render` endpoint runs the pipeline (Section 2) and
  returns the composited image. Models loaded once at startup (warm). Apple Silicon: PyTorch with
  `device="mps"`, CoreML for Depth Pro where convertible.
- **Cloudflare Tunnel** (`cloudflared`) from the server — no public IP, no port-forwarding; the PWA calls the
  tunnel hostname. **Note:** Cloudflare's proxy caps request body at **100 MB** (free), comfortably above
  phone-photo sizes; downscale very large inputs server-side before processing anyway.
- **Performance (Apple Silicon, per image):** expect a few seconds for matte + depth + blur on M-series.
  Diffusion options (BokehDiff/Marigold) add notably more. Cache loaded models; process at a sane working
  resolution (e.g. long edge 1536–2048) then optionally upscale the matte for the final composite.

---

## 6. Full model reference (as of June 2026)

Since this is personal-use, the "license" column is informational only — nothing here is disqualified.

### Matting / segmentation
| Model | License | Soft alpha | Edge/hair | Notes |
|---|---|---|---|---|
| **BiRefNet** ⭐ | MIT | ✅ | excellent (HR @2048²) | Chosen primary. `-matting`, `_HR-matting`, `_dynamic-matting`. ~17fps@1024² on 4090; runs MPS. |
| ViTMatte | MIT | ✅ | excellent **refiner** | Trimap-based 2nd stage. HF Transformers. |
| transparent-background (InSPyReNet) | MIT | ✅ | very good | `Remover().process(img)`. Easiest. |
| rembg (lib) | MIT (wrapper) | ✅ | backend-dependent | ONNX, one-liner. Backends carry own licenses. |
| BEN2 | MIT (base) | ✅ | strong, 4K | ONNX weights; best model API-gated. |
| IS-Net / DIS | Apache-2.0 | ⚠️ binary-ish | good | Powers rembg `isnet-general-use`. |
| MODNet | Apache-2.0 (code) | ✅ trimap-free | dated hair | Real-time; weights provenance unclear. |
| BRIA RMBG-2.0 | CC-BY-NC | ✅ | excellent | Built on BiRefNet arch — just use BiRefNet. |
| ZIM | CC-BY-NC | ✅ | strong fine edges | SAM-based, prompt-driven. |
| SAM / SAM 2 | Apache-2.0 | ❌ hard mask | — | Trimap/prompt generator only, not matting. |
| RVM | GPL-3.0 | ✅ | good | Real-time **video**; CoreML/ONNX/TF.js. (Not needed — stills only.) |
| MatAnyone / MatAnyone2 | NTU S-Lab (NC) | ✅ | SOTA video | Best video matting; not needed for stills. |

### Monocular depth
| Model | License | Type | Edges | Notes |
|---|---|---|---|---|
| **Apple Depth Pro** ⭐ | research (apple-amlr) | metric | **best boundaries** | Chosen. Sharpest edges + hair recall; native Apple Silicon; ONNX community export. |
| Depth Anything V2 (S/B/L/G) | S=Apache, B/L/G=CC-BY-NC | relative (+metric variants) | good (softer than Depth Pro) | De-facto baseline; huge ecosystem; MPS/CoreML/ONNX. |
| Depth Anything 3 (DA3) | mixed per variant | rel/metric | better geometry | Newest (Nov 2025). DA3MONO-LARGE relative, DA3METRIC-LARGE metric. |
| Marigold | Apache-2.0 | relative | crisp (diffusion) | Trivial `diffusers` integration; slow. |
| Lotus / Lotus-2 | Apache-2.0 | rel + normals | crisp | Apache alt to Marigold; faster (single-step). |
| Metric3D v2 | BSD-2 | metric + normals | sharp | Needs focal length; true metric CoC. |
| UniDepth V2 | CC-BY-NC | metric | sharp, +ONNX | Predicts own intrinsics. |
| MiDaS / ZoeDepth | MIT | rel / metric | softer | Legacy; tiny/fast ONNX/CoreML variants. |

### Bokeh / lens-blur renderers
| Tool | License | Inputs | Realism | Notes |
|---|---|---|---|---|
| **BokehMe** ⭐ | Apache-2.0 | RGB + disparity | high | K/focus/gamma/highlight/aperture; neural fix for bad depth. ⚠️ classical renderer uses CUDA — verify MPS. |
| BokehMe++ | paper (TPAMI 2024) | RGB + disp + **alpha** | higher | Cat-eye, highlight modes, alpha input to keep hair sharp. Standalone code unconfirmed. |
| MPIB | verify repo | RGB + disparity | high | Multiplane-image, **partial-occlusion** focus + inpainting module. |
| Dr.Bokeh | verify repo ([code](https://github.com/ShengCN/DrBokeh-Src)) | RGBD layers | high | Differentiable, occlusion-aware; color-bleed solved in render stage. |
| BokehDiff (ICCV 2025) | verify ([code](https://github.com/FreeButUselessSoul/bokehdiff)) | image + depth | very high | 1-step diffusion, physics-constrained CoC. Heavy. |
| OpenCV-bokeh (akakikuumeri) | verify (hobby) | image + depth | medium | Z-sliced slabs + faux-HDR. Simplest CPU/MPS-safe path. |
| three.js BokehShader | MIT | depth | real-time | If you ever move blur client-side. |
| Bart Wronski / McIntosh | shader refs | depth | real-time | Blueprints for separable disk / polygonal bokeh. |

---

## 7. Recommended build order (when we start coding)

1. **Skeleton:** FastAPI `POST /render` + minimal PWA upload/preview; echo image through (no processing).
2. **Matte:** BiRefNet on MPS → return cutout, validate alpha quality on real hair photos.
3. **Edge toolkit:** guided-filter refine + pymatting decontamination → verify clean edges on a transparent-bg test composite.
4. **Depth:** Apple Depth Pro → visualize depth/disparity, pick the `disp_focus` UX.
5. **Blur v1:** custom OpenCV linear-light **scatter** renderer (MPS-safe), signed CoC, disk kernel.
6. **Composite:** inpaint (LaMa) → premultiplied `over` → validate no halo / no sticker edge.
7. **Lens character:** aperture-blade kernel, highlight boost, optical-vignetting (cat-eye), bloom.
8. **Upgrade pass (optional):** swap in BokehMe / BokehDiff if MPS-viable and quality warrants.

---

## 8. Caveats & things to verify before relying on them

- **BokehMe classical renderer / CUDA** on Apple Silicon — verify MPS or CPU-fallback works; OpenCV scatter is the safe primary.
- **Apple Depth Pro** has a split license (permissive code, research-only weights) and an HF metadata inconsistency — irrelevant for personal use, noted for completeness.
- **Depth Anything** licensing is **per-size** (only some checkpoints Apache) — irrelevant for personal use.
- Renderer repos **MPIB, Dr.Bokeh, BokehDiff, OpenCV-bokeh** have unconfirmed licenses — irrelevant for personal use; check before any redistribution.
- **MODNet** strongest weights may be dataset-restricted; **BEN2** best model is API-gated.

---

## Sources

Matting: BiRefNet <https://github.com/ZhengPeng7/BiRefNet> · ViTMatte <https://github.com/hustvl/ViTMatte> ·
InSPyReNet/transparent-background <https://github.com/plemeri/transparent-background> ·
rembg <https://github.com/danielgatis/rembg> · BEN2 <https://github.com/PramaLLC/BEN2> ·
DIS <https://github.com/xuebinqin/DIS> · MODNet <https://github.com/ZHKKKe/MODNet> ·
RVM <https://github.com/PeterL1n/RobustVideoMatting> · MatAnyone <https://github.com/pq-yang/MatAnyone> ·
ZIM <https://github.com/naver-ai/ZIM> · SAM2 <https://github.com/facebookresearch/sam2> ·
BRIA RMBG-2.0 <https://huggingface.co/briaai/RMBG-2.0>

Depth: Depth Pro <https://github.com/apple/ml-depth-pro> · Depth Anything V2 <https://github.com/DepthAnything/Depth-Anything-V2> ·
Depth Anything 3 <https://github.com/ByteDance-Seed/Depth-Anything-3> · Marigold <https://github.com/prs-eth/Marigold> ·
Lotus <https://github.com/EnVision-Research/Lotus> · Metric3D <https://github.com/YvanYin/Metric3D> ·
UniDepth <https://github.com/lpiccinelli-eth/UniDepth> · MiDaS <https://github.com/isl-org/MiDaS>

Bokeh: BokehMe <https://github.com/JuewenPeng/BokehMe> · BokehMe++ <https://ieeexplore.ieee.org/document/10756626/> ·
MPIB <https://github.com/JuewenPeng/MPIB> · Dr.Bokeh <https://arxiv.org/abs/2308.08843> ·
BokehDiff <https://arxiv.org/abs/2507.18060> · Bokehlicious <https://arxiv.org/abs/2503.16067> ·
OpenCV-bokeh <https://github.com/akakikuumeri/OpenCV-bokeh> · MJP Bokeh <https://therealmjp.github.io/posts/bokeh/> ·
Separable bokeh (Wronski) <https://bartwronski.com/2017/08/06/separable-bokeh/> ·
Polygonal bokeh (McIntosh) <http://ivizlab.sfu.ca/papers/cgf2012.pdf>

Edges/compositing: alpha compositing <https://en.wikipedia.org/wiki/Alpha_compositing> ·
pymatting <https://github.com/pymatting/pymatting> · FMLFE <https://arxiv.org/abs/2006.14970> ·
guided filter <https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf> ·
fast guided filter <https://arxiv.org/abs/1505.00996> · OpenCV guidedFilter <https://docs.opencv.org/4.x/de/d73/classcv_1_1ximgproc_1_1GuidedFilter.html> ·
libcom <https://github.com/bcmi/libcom> · Google Synthetic DoF <https://arxiv.org/abs/1806.04171>

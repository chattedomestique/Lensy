# Lensy object removal — machine setup (no ComfyUI)

Complete, hand-off-able instructions to get Lensy's three-engine object removal running on a Mac
that **does not have ComfyUI**. On such a machine you get **Quick Erase (LaMa)** and **Deep Clean
(ObjectClear)**; **Reconstruct (Flux)** is disabled (it requires ComfyUI) so its button never shows.

---

## Why "all three modes failed" (read first)

Object removal was rebuilt as a **background job** with new endpoints (`/erase` → `job_id`,
`/erase/{id}/events`, `/erase/{id}/status`, `/engines`, `/warm`). If the **front end** updated
(e.g. via the GitHub Pages deploy) but the **back end on the Mac did not**, every erase — Quick
included — fails, because the old backend doesn't have those endpoints.

**The fix is always: update the code on the Mac, reinstall deps, rebuild the front end, restart the
backend.** Steps below. A 30-second confirmation that the backend is current:

```bash
curl -s http://localhost:8842/engines
# current backend → {"engines":{...},"enabled":[...],"errors":{}}
# 404 / "Not Found"  → the backend is STALE. Do step 6 (rebuild + restart).
```

---

## 0. Prerequisites

- Apple-Silicon Mac (the 16 GB M4 mini is the reference target).
- [Homebrew](https://brew.sh), then: `brew install python@3.11 node git`
- Your existing Cloudflare **named tunnel** mapping `lensy.sunhouse.media → http://localhost:8842`
  (unchanged from your current setup).

## 1. Get the code

```bash
# fresh machine:
git clone https://github.com/chattedomestique/Lensy.git ~/Lensy
# OR update an existing checkout:
cd ~/Lensy && git fetch origin && git checkout main && git pull
```

## 2. Base install (venv, models, weights) — this gives you Quick Erase

```bash
cd ~/Lensy
./scripts/setup.sh
```

Idempotent. Installs the backend venv + core deps, the model deps (`torch`, `transformers`,
`simple-lama-inpainting` = **LaMa/Quick Erase**, BiRefNet, SAM2, depth), and pre-caches weights.
After this, **Quick Erase works** with no further setup.

## 3. Deep Clean (ObjectClear) — optional, recommended

Two pieces: the Python deps, and the `objectclear` package.

**a) Deps** (into Lensy's venv):

```bash
cd ~/Lensy/backend
source .venv/bin/activate
pip install -e '.[removal]'        # diffusers + websocket-client
```

**b) The `objectclear` package.** It's the S-Lab ObjectClear code that backs the
`jixin0101/ObjectClear` weights (not on PyPI). If you already have your Vanish install, just reuse it
— it lives at `~/object-removal-studio/ObjectClear`. On a fresh machine, copy that folder over from
your dev machine (simplest — it also carries the cached weights), **or** clone the ObjectClear repo
there and install its requirements:

```bash
mkdir -p ~/object-removal-studio
# copy ~/object-removal-studio/ObjectClear from your dev machine, OR clone S-Lab's ObjectClear repo:
#   git clone <ObjectClear repo> ~/object-removal-studio/ObjectClear
pip install -r ~/object-removal-studio/ObjectClear/requirements.txt   # if the repo ships one
deactivate
```

Lensy finds it via `VANISH_OC_REPO` (defaults to `~/object-removal-studio/ObjectClear`). The ~7 GB
of weights auto-download from Hugging Face on the first Deep Clean.

> **Version-conflict note.** ObjectClear may pin specific `diffusers`/`torch` versions. If it won't
> import inside Lensy's venv, `/engines` will show the exact error (step 7). The clean fix in that
> case is to run the heavy engines from ObjectClear's own venv as a sidecar — ping me and I'll add
> that mode; the port is structured for it.

## 4. Reconstruct (Flux) — skip on this machine

Flux needs a ComfyUI install, which this machine doesn't have. Disable it so the picker never shows
Reconstruct and it's never attempted (Quick Erase remains the fallback anyway):

```bash
export VANISH_ENGINES=lama,objectclear
```

## 5. Environment

Put these in the shell that launches the backend (or append to `~/.zshrc`):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1          # unsupported ops → CPU, no crash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # use the full unified memory
export VANISH_LOW_MEM=1                        # 16 GB: only one heavy model resident at a time
export VANISH_ENGINES=lama,objectclear         # no Flux (no ComfyUI)
export VANISH_OC_REPO="$HOME/object-removal-studio/ObjectClear"   # only if not the default
# export HF_TOKEN=...    # only if a model turns out to be gated
```

## 6. Build the front end + run the backend

```bash
cd ~/Lensy
(cd frontend && npm run build)     # rebuild the PWA the backend serves
./scripts/serve.sh                 # serves app + API on :8842 (your tunnel → lensy.sunhouse.media)
```

> **After every code update you MUST do both:** rebuild `frontend/dist` **and** restart
> `serve.sh`. The backend both serves the built front end and holds the API, so a stale build or a
> stale process is the #1 cause of "erase failed." Prefer opening **lensy.sunhouse.media** (served by
> the Mac) over the GitHub Pages URL, so front end and back end are always the same commit.

## 7. Verify

```bash
curl -s http://localhost:8842/healthz | python3 -m json.tool
#   … "inpaint": "lama"   ← Quick Erase ready ("fallback(cv2)" means LaMa didn't load; re-run setup.sh)

curl -s http://localhost:8842/engines | python3 -m json.tool
#   {"engines":{"lama":"ready","objectclear":"idle","flux":"idle"},
#    "enabled":["lama","objectclear"], "errors":{}}

# warm Deep Clean and watch it load (first time also downloads ~7 GB):
curl -s -X POST -F engine=objectclear http://localhost:8842/warm
sleep 5 && curl -s http://localhost:8842/engines | python3 -m json.tool
#   objectclear: "loading" → "ready", OR an "errors":{"objectclear":"…"} you can paste to me
```

## 8. Using it in the app

Erase tool → pick **Quick** or **Deep Clean** → tap or brush the object → **Erase**. The progress
pill shows a real percentage. Deep Clean's first run is slow (weight download + model load, minutes
on 16 GB); after that it's warm. If a heavy engine can't load it falls back to Quick Erase and the
app tells you; `/engines` shows why.

---

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| **All modes fail**, `/engines` 404s | Backend is stale. `git pull`, `pip install -e '.[removal]'`, rebuild front end, restart `serve.sh`. |
| All modes fail, `/engines` works | Front end/back end on different commits. Use `lensy.sunhouse.media` (Mac-served) so they match; rebuild `frontend/dist`. |
| **Quick Erase** fails | `/healthz` shows `inpaint: fallback(cv2)` → LaMa didn't install. Re-run `./scripts/setup.sh` (model step); check the `serve.sh` log. |
| **Deep Clean** fails | `curl /engines` → read `errors.objectclear`. Usually the `objectclear` package isn't found (set `VANISH_OC_REPO`) or a `diffusers` version conflict (use the sidecar mode). |
| **Deep Clean** OOMs on 16 GB | Ensure `VANISH_LOW_MEM=1` and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` are set; close other heavy apps. |
| **Reconstruct** missing | Expected — no ComfyUI. `VANISH_ENGINES=lama,objectclear` hides it. |

**All `VANISH_*` / `LENSY_*` knobs:** `VANISH_ENGINES`, `VANISH_LOW_MEM`, `VANISH_STUDIO`,
`VANISH_OC_REPO`, `VANISH_COMFY_DIR`, `VANISH_COMFY_PY`, `VANISH_FLUX_WORKFLOW`, `VANISH_COMFY_ARGS`
(default `--lowvram`), `VANISH_FLUX_TIMEOUT`, `LENSY_PORT`, `LENSY_MAX_EXPORT_EDGE`,
`LENSY_LAMA_MAX_EDGE`, `LENSY_ERASE_MAX_EDGE`.

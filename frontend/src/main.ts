// Lensy front-end entry. Flow: upload → Generate depth map → edit the depth map live →
// Render → before/after → Save. Rendering/state logic lives in api.ts, depth.ts, controls.ts.

import "./styles/main.css";
import { registerSW } from "virtual:pwa-register";
import {
  ApiError,
  analyze,
  depthUrl,
  matteUrl,
  render,
  renderFromAnalyze,
  type RenderHandle,
} from "./api";
import { Controls } from "./controls";
import { DEFAULT_ADJUST, DepthEditor, type DepthAdjust } from "./depth";
import { setupServerPanel } from "./server";

registerSW({ immediate: true });

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const dropzone = $("dropzone");
const fileInput = $("file") as HTMLInputElement;
const stage = $("stage");
const generateBtn = $("generate") as HTMLButtonElement;
const renderBtn = $("render") as HTMLButtonElement;
const resetBtn = $("reset") as HTMLButtonElement;
const saveBtn = $("save") as HTMLButtonElement;
const progress = $("progress");
const progressFill = $("progress-fill");
const progressLabel = $("progress-label");
const progressPct = $("progress-pct");
const focusRing = $("focus-ring");
const depthControls = $("depth-controls");

let currentFile: File | null = null;
let originalUrl: string | null = null;
let resultUrl: string | null = null;
let resultBlob: Blob | null = null;
let analyzeId: string | null = null;
let inflight: RenderHandle | null = null;
let view: "photo" | "depth" = "photo";

const editor = new DepthEditor();
const depthCanvas = document.createElement("canvas");
depthCanvas.className = "stage-img";

const controls = new Controls(() => {});

// --- depth adjustment sliders ---
const adjust: DepthAdjust = { ...DEFAULT_ADJUST };
const depthSliders = {
  black: $("d-black") as HTMLInputElement,
  white: $("d-white") as HTMLInputElement,
  contrast: $("d-contrast") as HTMLInputElement,
  smoothing: $("d-smooth") as HTMLInputElement,
};
function readAdjust(): void {
  adjust.black = Number(depthSliders.black.value) / 100;
  adjust.white = Number(depthSliders.white.value) / 100;
  adjust.contrast = Number(depthSliders.contrast.value) / 100;
  adjust.smoothing = Number(depthSliders.smoothing.value) / 100;
  $("d-black-val").textContent = adjust.black.toFixed(2);
  $("d-white-val").textContent = adjust.white.toFixed(2);
  $("d-contrast-val").textContent = adjust.contrast.toFixed(2);
  $("d-smooth-val").textContent = adjust.smoothing.toFixed(2);
}
Object.values(depthSliders).forEach((s) =>
  s.addEventListener("input", () => {
    if (!editor.ready) return;
    readAdjust();
    editor.apply(adjust);
    if (view === "depth") editor.drawTo(depthCanvas);
  }),
);

// --- toast ---
let toastTimer: number | undefined;
function toast(message: string): void {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove("show"), 4500);
}

// --- stage content ---
function showPhoto(): void {
  if (!originalUrl) return;
  stage.innerHTML = "";
  const img = document.createElement("img");
  img.src = originalUrl;
  img.className = "stage-img";
  img.alt = "Your photo";
  img.addEventListener("click", onFocusTap);
  stage.appendChild(img);
  stage.appendChild(focusRing);
}
function showDepth(): void {
  editor.drawTo(depthCanvas);
  stage.innerHTML = "";
  depthCanvas.addEventListener("click", onFocusTap);
  stage.appendChild(depthCanvas);
  stage.appendChild(focusRing);
}
function setView(v: "photo" | "depth"): void {
  view = v;
  $("view")
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === v)));
  if (v === "depth") showDepth();
  else showPhoto();
}

// tap-to-focus: sample the depth at the tapped point (falls back to a vertical guess)
function onFocusTap(e: MouseEvent): void {
  const el = e.currentTarget as HTMLElement;
  const rect = el.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;
  const focus = editor.ready ? editor.sampleAt(nx, ny) : 1 - ny * 0.85;
  controls.setFocus(focus);
  focusRing.style.left = `${e.clientX - rect.left}px`;
  focusRing.style.top = `${e.clientY - rect.top}px`;
  focusRing.classList.remove("show");
  void focusRing.offsetWidth;
  focusRing.classList.add("show");
  toast(`Focus set at depth ${focus.toFixed(2)}`);
}

// --- before/after compare ---
function showCompare(beforeUrl: string, afterUrl: string): void {
  stage.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "compare";
  wrap.innerHTML = `
    <img class="before" src="${beforeUrl}" alt="Before" />
    <img class="after" src="${afterUrl}" alt="After" />
    <span class="badge l">Before</span>
    <span class="badge r">After</span>
    <div class="handle" aria-label="Drag to compare"></div>`;
  stage.appendChild(wrap);
  const handle = wrap.querySelector<HTMLDivElement>(".handle")!;
  const setSplit = (clientX: number) => {
    const rect = wrap.getBoundingClientRect();
    const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    wrap.style.setProperty("--split", `${pct}%`);
  };
  wrap.style.setProperty("--split", "50%");
  let dragging = false;
  handle.addEventListener("pointerdown", (e) => {
    dragging = true;
    setSplit(e.clientX);
    handle.setPointerCapture(e.pointerId);
  });
  wrap.addEventListener("pointermove", (e) => dragging && setSplit(e.clientX));
  window.addEventListener("pointerup", () => (dragging = false));
}

// --- file intake ---
function acceptFile(file: File): void {
  if (!file.type.startsWith("image/")) {
    toast("That doesn't look like an image.");
    return;
  }
  currentFile = file;
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);
  analyzeId = null;
  resetDepthUI();
  showPhoto();
  generateBtn.disabled = false;
  generateBtn.classList.remove("hidden");
  resetBtn.disabled = false;
  renderBtn.classList.add("hidden");
  saveBtn.classList.add("hidden");
}

function resetDepthUI(): void {
  depthControls.classList.add("hidden");
  depthSliders.black.value = "0";
  depthSliders.white.value = "100";
  depthSliders.contrast.value = "0";
  depthSliders.smoothing.value = "0";
  readAdjust();
  view = "photo";
}

// --- generate depth map ---
async function doGenerate(): Promise<void> {
  if (!currentFile) return;
  generateBtn.disabled = true;
  setProgress(0.1, "Analyzing…");
  progress.classList.remove("hidden");
  try {
    const { analyzeId: aid } = await analyze(currentFile);
    analyzeId = aid;
    await editor.load(depthUrl(aid), matteUrl(aid));
    readAdjust();
    editor.apply(adjust);
    progress.classList.add("hidden");
    depthControls.classList.remove("hidden");
    generateBtn.classList.add("hidden");
    renderBtn.classList.remove("hidden");
    setView("depth"); // show them the depth map they can now edit
    toast("Depth map ready — edit it, then Render.");
  } catch (err) {
    progress.classList.add("hidden");
    generateBtn.disabled = false;
    toast(err instanceof ApiError ? err.message : "Could not generate the depth map.");
  }
}

// --- render ---
function setProgress(frac: number, label: string): void {
  progress.classList.remove("hidden");
  progressFill.style.width = `${Math.round(frac * 100)}%`;
  progressPct.textContent = `${Math.round(frac * 100)}%`;
  progressLabel.textContent = label;
}

async function doRender(): Promise<void> {
  if (!currentFile || !originalUrl) return;
  renderBtn.disabled = true;
  saveBtn.classList.add("hidden");
  setProgress(0.02, "Sending…");
  inflight?.cancel();

  try {
    if (analyzeId && editor.ready) {
      const depthPng = await editor.exportPng();
      inflight = renderFromAnalyze(analyzeId, depthPng, controls.params(), (p) =>
        setProgress(p.progress, p.label),
      );
    } else {
      inflight = render(currentFile, controls.params(), (p) => setProgress(p.progress, p.label));
    }
    const url = await inflight.done;
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = url;
    resultBlob = await (await fetch(url)).blob();
    showCompare(originalUrl, url);
    progress.classList.add("hidden");
    saveBtn.classList.remove("hidden");
  } catch (err) {
    progress.classList.add("hidden");
    toast(err instanceof ApiError ? err.message : "Something went wrong rendering.");
  } finally {
    renderBtn.disabled = false;
    inflight = null;
  }
}

function reset(): void {
  inflight?.cancel();
  controls.reset();
  resetDepthUI();
  if (originalUrl) {
    showPhoto();
    renderBtn.classList.add("hidden");
    generateBtn.classList.remove("hidden");
    generateBtn.disabled = false;
  }
  progress.classList.add("hidden");
  saveBtn.classList.add("hidden");
}

// --- save to phone (Web Share → Save to Photos; falls back to download) ---
async function save(): Promise<void> {
  if (!resultBlob) return;
  const file = new File([resultBlob], "lensy.jpg", { type: "image/jpeg" });
  const nav = navigator as Navigator & { canShare?: (d: unknown) => boolean; share?: (d: unknown) => Promise<void> };
  if (nav.canShare?.({ files: [file] }) && nav.share) {
    try {
      await nav.share({ files: [file], title: "Lensy" });
      return;
    } catch {
      /* user cancelled or share failed → fall through to download */
    }
  }
  const a = document.createElement("a");
  a.href = resultUrl!;
  a.download = "lensy.jpg";
  a.click();
}

// --- wiring ---
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) acceptFile(f);
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }),
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  }),
);
dropzone.addEventListener("drop", (e) => {
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (f) acceptFile(f);
});

$("view")
  .querySelectorAll<HTMLButtonElement>("button")
  .forEach((b) => b.addEventListener("click", () => setView(b.dataset.view as "photo" | "depth")));

generateBtn.addEventListener("click", doGenerate);
renderBtn.addEventListener("click", doRender);
resetBtn.addEventListener("click", reset);
saveBtn.addEventListener("click", save);

setupServerPanel();

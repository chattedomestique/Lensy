// Lensy front-end. Flow: upload → Generate depth map → place anchors + tune the focus map live
// (Depth tab) → Render (Lens/Export) → before/after → Save. Tabs keep the image visible.

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
import { type AnchorName, DEFAULT_SETTINGS, DepthEditor, type DepthSettings } from "./depth";
import { setupServerPanel } from "./server";

registerSW({ immediate: true });

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const dropzone = $("dropzone");
const fileInput = $("file") as HTMLInputElement;
const stage = $("stage");
const generateBtn = $("generate") as HTMLButtonElement;
const renderBtn = $("render") as HTMLButtonElement;
const resetBtn = $("reset") as HTMLButtonElement;
const resetDepthBtn = $("reset-depth") as HTMLButtonElement;
const saveBtn = $("save") as HTMLButtonElement;
const progress = $("progress");
const progressFill = $("progress-fill");
const progressLabel = $("progress-label");
const progressPct = $("progress-pct");
const focusRing = $("focus-ring");
const depthTools = $("depth-tools");
const anchorHint = $("anchor-hint");

let currentFile: File | null = null;
let originalUrl: string | null = null;
let resultUrl: string | null = null;
let resultBlob: Blob | null = null;
let analyzeId: string | null = null;
let inflight: RenderHandle | null = null;
let view: "photo" | "depth" = "depth";
let armed: AnchorName | null = null;

const editor = new DepthEditor();
const depthCanvas = document.createElement("canvas");
depthCanvas.className = "stage-img";

const loupe = $("loupe");
const loupeCanvas = $("loupe-canvas") as HTMLCanvasElement;
const origImg = new Image();
origImg.crossOrigin = "anonymous";

// anchor placement (drag) state
let placing = false;
let placingName: AnchorName = "subject";
let rafPending = false;
let lastNx = 0;
let lastNy = 0;

const controls = new Controls(() => {});
const settings: DepthSettings = { ...DEFAULT_SETTINGS };

// --- toast ---
let toastTimer: number | undefined;
function toast(message: string): void {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove("show"), 4000);
}

// --- tabs ---
document.querySelectorAll<HTMLButtonElement>(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab!;
    document.querySelectorAll<HTMLButtonElement>(".tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t === tab)),
    );
    document.querySelectorAll<HTMLElement>(".tabpanel").forEach((p) =>
      p.classList.toggle("hidden", p.dataset.panel !== name),
    );
  });
});

// --- stage content ---
function showPhoto(): void {
  if (!originalUrl) return;
  stage.innerHTML = "";
  const img = document.createElement("img");
  img.src = originalUrl;
  img.className = "stage-img";
  img.alt = "Your photo";
  bindPlacement(img);
  stage.appendChild(img);
  stage.appendChild(focusRing);
}
function showDepth(): void {
  editor.drawFocus(depthCanvas);
  stage.innerHTML = "";
  bindPlacement(depthCanvas);
  stage.appendChild(depthCanvas);
  stage.appendChild(focusRing);
}
function refreshDepthView(): void {
  if (view === "depth" && editor.ready) editor.drawFocus(depthCanvas);
}
function setView(v: "photo" | "depth"): void {
  view = v;
  $("view")
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === v)));
  if (v === "depth") showDepth();
  else showPhoto();
}

// --- anchors ---
function setArmed(name: AnchorName | null): void {
  armed = name;
  document.querySelectorAll<HTMLButtonElement>("#anchors button").forEach((b) =>
    b.classList.toggle("armed", b.dataset.anchor === name),
  );
  anchorHint.textContent = name ? `tap to place ${name}` : "pick one";
}
document.querySelectorAll<HTMLButtonElement>("#anchors button").forEach((b) => {
  b.addEventListener("click", () => setArmed(armed === b.dataset.anchor ? null : (b.dataset.anchor as AnchorName)));
});

// drag an anchor around the image with a zoomed crosshair loupe; live-updates the depth map
function bindPlacement(el: HTMLElement): void {
  el.style.touchAction = "none";
  el.addEventListener("pointerdown", (e) => {
    if (!editor.ready) return;
    e.preventDefault();
    placing = true;
    placingName = armed ?? "subject"; // a plain drag sets the Subject (focus) point
    loupe.classList.remove("hidden");
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      /* some pointer types can't be captured — fine */
    }
    updatePlacement(e, el);
  });
  el.addEventListener("pointermove", (e) => {
    if (placing) updatePlacement(e, el);
  });
  const end = () => {
    if (!placing) return;
    placing = false;
    loupe.classList.add("hidden");
    document
      .querySelector<HTMLButtonElement>(`#anchors button[data-anchor="${placingName}"]`)
      ?.classList.add("set");
    setArmed(null);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}

function updatePlacement(e: PointerEvent, el: HTMLElement): void {
  const rect = el.getBoundingClientRect();
  lastNx = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  lastNy = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
  positionLoupe(e.clientX, e.clientY);
  drawLoupe(lastNx, lastNy);
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      editor.setAnchor(placingName, lastNx, lastNy);
      refreshDepthView();
    });
  }
}

function positionLoupe(clientX: number, clientY: number): void {
  const size = 150;
  const x = Math.min(window.innerWidth - size - 8, Math.max(8, clientX - size / 2));
  const y = Math.max(8, clientY - size - 24); // float above the finger
  loupe.style.left = `${x}px`;
  loupe.style.top = `${y}px`;
}

function drawLoupe(nx: number, ny: number): void {
  const ctx = loupeCanvas.getContext("2d")!;
  const iw = origImg.naturalWidth;
  const ih = origImg.naturalHeight;
  ctx.clearRect(0, 0, loupeCanvas.width, loupeCanvas.height);
  if (!iw) return;
  const zoom = 4;
  const cw = loupeCanvas.width / zoom;
  const ch = loupeCanvas.height / zoom;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(origImg, nx * iw - cw / 2, ny * ih - ch / 2, cw, ch, 0, 0, loupeCanvas.width, loupeCanvas.height);
}

// --- depth sliders ---
const sepSlider = $("d-sep") as HTMLInputElement;
const smoothSlider = $("d-smooth") as HTMLInputElement;
function readSettings(): void {
  settings.separation = Number(sepSlider.value) / 100;
  settings.smoothing = Number(smoothSlider.value) / 100;
  $("d-sep-val").textContent = settings.separation.toFixed(2);
  $("d-smooth-val").textContent = settings.smoothing.toFixed(2);
}
[sepSlider, smoothSlider].forEach((s) =>
  s.addEventListener("input", () => {
    if (!editor.ready) return;
    readSettings();
    editor.setSettings({ ...settings });
    refreshDepthView();
  }),
);

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
    wrap.style.setProperty("--split", `${Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100))}%`);
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
  origImg.src = originalUrl; // loupe source
  analyzeId = null;
  depthTools.classList.add("hidden");
  view = "photo";
  showPhoto();
  generateBtn.classList.remove("hidden");
  generateBtn.disabled = false;
  renderBtn.classList.add("hidden");
  saveBtn.classList.add("hidden");
  resetBtn.disabled = false;
}

// --- generate depth map ---
async function doGenerate(): Promise<void> {
  if (!currentFile) return;
  generateBtn.disabled = true;
  setProgress(0.1, "Analyzing…");
  try {
    const { analyzeId: aid } = await analyze(currentFile);
    analyzeId = aid;
    await editor.load(depthUrl(aid), matteUrl(aid));
    readSettings();
    editor.setSettings({ ...settings });
    progress.classList.add("hidden");
    depthTools.classList.remove("hidden");
    generateBtn.classList.add("hidden");
    renderBtn.classList.remove("hidden");
    document.querySelectorAll<HTMLButtonElement>("#anchors button").forEach((b) => b.classList.remove("set"));
    setView("depth");
    toast("Depth map ready — place anchors, tune, then Render.");
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
      const depthPng = await editor.exportDepthPng();
      const params = { ...controls.params(), disp_focus: editor.focalValue, autofocus: false };
      inflight = renderFromAnalyze(analyzeId, depthPng, params, (p) => setProgress(p.progress, p.label));
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
    $("export-empty").classList.add("hidden");
    toast("Rendered. Save it from the Export tab.");
  } catch (err) {
    progress.classList.add("hidden");
    toast(err instanceof ApiError ? err.message : "Something went wrong rendering.");
  } finally {
    renderBtn.disabled = false;
    inflight = null;
  }
}

function resetDepth(): void {
  editor.anchors = {};
  sepSlider.value = "0";
  smoothSlider.value = "0";
  readSettings();
  editor.setSettings({ ...settings });
  document.querySelectorAll<HTMLButtonElement>("#anchors button").forEach((b) => b.classList.remove("set"));
  setArmed(null);
  refreshDepthView();
}

function resetAll(): void {
  inflight?.cancel();
  controls.reset();
  resetDepth();
  if (originalUrl) {
    view = "photo";
    showPhoto();
    renderBtn.classList.add("hidden");
    generateBtn.classList.remove("hidden");
    generateBtn.disabled = false;
    depthTools.classList.add("hidden");
  }
  progress.classList.add("hidden");
  saveBtn.classList.add("hidden");
}

// --- save to phone (Web Share → Save to Photos; download fallback) ---
async function save(): Promise<void> {
  if (!resultBlob) return;
  const file = new File([resultBlob], "lensy.jpg", { type: "image/jpeg" });
  const nav = navigator as Navigator & { canShare?: (d: unknown) => boolean; share?: (d: unknown) => Promise<void> };
  if (nav.canShare?.({ files: [file] }) && nav.share) {
    try {
      await nav.share({ files: [file], title: "Lensy" });
      return;
    } catch {
      /* cancelled → download */
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
resetBtn.addEventListener("click", resetAll);
resetDepthBtn.addEventListener("click", resetDepth);
saveBtn.addEventListener("click", save);

setupServerPanel();

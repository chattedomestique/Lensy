// Lensy front-end entry — DOM glue that wires the dropzone, controls, render stream, and the
// before/after compare together. Rendering/state logic lives in api.ts and controls.ts.

import "./styles/main.css";
import { registerSW } from "virtual:pwa-register";
import { ApiError, render, type RenderHandle } from "./api";
import { Controls } from "./controls";
import { setupServerPanel } from "./server";

registerSW({ immediate: true });

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const dropzone = $("dropzone");
const fileInput = $("file") as HTMLInputElement;
const stage = $("stage");
const renderBtn = $("render") as HTMLButtonElement;
const resetBtn = $("reset") as HTMLButtonElement;
const downloadBtn = $("download") as HTMLButtonElement;
const progress = $("progress");
const progressFill = $("progress-fill");
const progressLabel = $("progress-label");
const progressPct = $("progress-pct");
const focusRing = $("focus-ring");

let originalUrl: string | null = null; // object URL of the uploaded photo
let resultUrl: string | null = null; // object URL of the rendered result
let currentFile: File | null = null;
let inflight: RenderHandle | null = null;

const controls = new Controls(() => {
  /* live param change — could trigger auto-render later; for now manual */
});

// --- toast -------------------------------------------------------------------
let toastTimer: number | undefined;
function toast(message: string): void {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove("show"), 4200);
}

// --- file intake -------------------------------------------------------------
function acceptFile(file: File): void {
  if (!file.type.startsWith("image/")) {
    toast("That doesn't look like an image.");
    return;
  }
  currentFile = file;
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);
  showPreview(originalUrl);
  renderBtn.disabled = false;
  resetBtn.disabled = false;
  downloadBtn.classList.add("hidden");
}

function showPreview(url: string): void {
  stage.innerHTML = "";
  const img = document.createElement("img");
  img.src = url;
  img.className = "stage-img";
  img.alt = "Your photo";
  img.addEventListener("click", onFocusTap);
  stage.appendChild(img);
  stage.appendChild(focusRing);
}

// tap-to-focus: without client-side depth we use a vertical proxy (lower in frame = nearer).
function onFocusTap(e: MouseEvent): void {
  const img = e.currentTarget as HTMLImageElement;
  const rect = img.getBoundingClientRect();
  const yNorm = (e.clientY - rect.top) / rect.height;
  controls.setFocus(1 - yNorm * 0.85); // bias toward "nearer" lower down
  focusRing.style.left = `${e.clientX - rect.left}px`;
  focusRing.style.top = `${e.clientY - rect.top}px`;
  focusRing.classList.remove("show");
  void focusRing.offsetWidth; // restart the pulse animation
  focusRing.classList.add("show");
}

// --- before/after compare ----------------------------------------------------
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
  const down = (x: number) => { dragging = true; setSplit(x); };
  handle.addEventListener("pointerdown", (e) => { down(e.clientX); handle.setPointerCapture(e.pointerId); });
  wrap.addEventListener("pointermove", (e) => { if (dragging) setSplit(e.clientX); });
  window.addEventListener("pointerup", () => { dragging = false; });
}

// --- render flow -------------------------------------------------------------
function setProgress(frac: number, label: string): void {
  progress.classList.remove("hidden");
  progressFill.style.width = `${Math.round(frac * 100)}%`;
  progressPct.textContent = `${Math.round(frac * 100)}%`;
  progressLabel.textContent = label;
}

async function doRender(): Promise<void> {
  if (!currentFile || !originalUrl) return;
  renderBtn.disabled = true;
  downloadBtn.classList.add("hidden");
  setProgress(0.02, "Sending photo…");

  inflight?.cancel();
  inflight = render(currentFile, controls.params(), (p) => setProgress(p.progress, p.label));

  try {
    const url = await inflight.done;
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = url;
    showCompare(originalUrl, url);
    progress.classList.add("hidden");
    downloadBtn.classList.remove("hidden");
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
  if (originalUrl) showPreview(originalUrl);
  progress.classList.add("hidden");
  downloadBtn.classList.add("hidden");
}

// --- wiring -------------------------------------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) acceptFile(f);
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }),
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }),
);
dropzone.addEventListener("drop", (e) => {
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (f) acceptFile(f);
});

renderBtn.addEventListener("click", doRender);
resetBtn.addEventListener("click", reset);
downloadBtn.addEventListener("click", () => {
  if (!resultUrl) return;
  const a = document.createElement("a");
  a.href = resultUrl;
  a.download = "lensy.jpg";
  a.click();
});

// --- server connect pill (shows backend status; lets you point at your tunnel) ---
setupServerPanel();

// Lensy front-end — a VSCO-style, no-scroll photo editor. The image fills the stage; a bottom
// tab bar picks a tool; you adjust each value by holding anywhere on the image and dragging up or
// down (a big finger slider with an odometer read-out). Depth previews live; every other change
// re-renders on the backend when you lift your finger.

import "./styles/main.css";
import { registerSW } from "virtual:pwa-register";
import {
  ApiError,
  type RenderParams,
  analyze,
  depthUrl,
  matteUrl,
  renderFromAnalyze,
  type RenderHandle,
} from "./api";
import { DepthEditor } from "./depth";
import { setupServerPanel } from "./server";

registerSW({ immediate: true });

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- element refs ----
const dropzone = $("dropzone");
const fileInput = $("file") as HTMLInputElement;
const stage = $("stage");
const resultImg = $("result") as HTMLImageElement;
const depthView = $("depthview") as HTMLCanvasElement;
const dragSurface = $("drag-surface");
const toolLabel = $("tool-label");
const ticker = $("ticker");
const subrow = $("subrow");
const tabbar = $("tabbar");
const saveBtn = $("save") as HTMLButtonElement;
const compareBtn = $("compare") as HTMLButtonElement;
const progress = $("progress");
const progressLabel = $("progress-label");

// ---- editable state (all values live in 0..100 UI units) ----
type Key =
  | "amount" | "position" | "contrast" | "falloff"       // depth (client-side)
  | "k" | "highlight" | "halation" | "halationSize"       // lens (backend)
  | "ca" | "swirl" | "sweet" | "sweetSize";

const DEFAULTS: Record<Key, number> = {
  amount: 50, position: 50, contrast: 0, falloff: 20,
  k: 60, highlight: 18, halation: 0, halationSize: 40,
  ca: 0, swirl: 0, sweet: 0, sweetSize: 35,
};
const state: Record<Key, number> = { ...DEFAULTS };
let blades = 0;

interface Tool {
  id: string;
  label: string;
  params: { key: Key; label: string }[];
  shapes?: boolean; // Bokeh shows an aperture-shape picker
}

const TOOLS: Tool[] = [
  {
    id: "depth", label: "Depth",
    params: [
      { key: "amount", label: "Depth amount" },
      { key: "position", label: "Depth position" },
      { key: "contrast", label: "Depth contrast" },
      { key: "falloff", label: "Depth falloff" },
    ],
  },
  { id: "bokeh", label: "Bokeh", params: [{ key: "k", label: "Blur" }], shapes: true },
  { id: "bloom", label: "Bloom", params: [{ key: "highlight", label: "Bloom" }] },
  {
    id: "halation", label: "Halation",
    params: [
      { key: "halation", label: "Halation" },
      { key: "halationSize", label: "Spread" },
    ],
  },
  { id: "chroma", label: "Chroma", params: [{ key: "ca", label: "Chromatic aberration" }] },
  { id: "petzval", label: "Petzval", params: [{ key: "swirl", label: "Swirl" }] },
  {
    id: "lensbaby", label: "Lensbaby",
    params: [
      { key: "sweet", label: "Lensbaby" },
      { key: "sweetSize", label: "Spot size" },
    ],
  },
];

let activeTool = TOOLS[0];
let activeKey: Key = "amount";

// ---- runtime ----
const editor = new DepthEditor();
let currentFile: File | null = null;
let originalUrl: string | null = null;
let resultUrl: string | null = null;
let resultBlob: Blob | null = null;
let analyzeId: string | null = null;
let inflight: RenderHandle | null = null;
let renderTimer: number | undefined;
let rafPending = false;

// ---- toast ----
let toastTimer: number | undefined;
function toast(message: string): void {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove("show"), 4000);
}

// ---- odometer ticker (0..100) ----
// Each digit is a vertical reel of 0-9; we translate it to the target digit so it rolls.
function buildTicker(): void {
  ticker.innerHTML = "";
  for (let d = 0; d < 3; d++) {
    const reel = document.createElement("div");
    reel.className = "reel";
    const strip = document.createElement("div");
    strip.className = "strip";
    for (let n = 0; n <= 9; n++) {
      const cell = document.createElement("span");
      cell.textContent = String(n);
      strip.appendChild(cell);
    }
    reel.appendChild(strip);
    ticker.appendChild(reel);
  }
}
function setTicker(value: number): void {
  const v = Math.round(value);
  const digits = String(v).padStart(3, "0").split("").map(Number);
  const reels = ticker.querySelectorAll<HTMLElement>(".reel");
  const lead = v >= 100 ? 0 : v >= 10 ? 1 : 2; // hide leading-zero reels
  reels.forEach((reel, i) => {
    reel.classList.toggle("hidden", i < lead);
    const strip = reel.querySelector<HTMLElement>(".strip")!;
    strip.style.transform = `translateY(${-digits[i] * 10}%)`;
  });
}

// ---- stage media ----
function showResult(url: string): void {
  resultImg.src = url;
  resultImg.classList.remove("hidden");
  depthView.classList.add("hidden");
}
function showDepthLive(): void {
  editor.drawFocus(depthView);
  depthView.classList.remove("hidden");
  resultImg.classList.add("hidden");
}
function hideDepthLive(): void {
  depthView.classList.add("hidden");
  if (resultImg.src) resultImg.classList.remove("hidden");
}

// ---- tab bar + sub-parameter row ----
function buildTabs(): void {
  tabbar.innerHTML = "";
  for (const t of TOOLS) {
    const b = document.createElement("button");
    b.className = "tab";
    b.textContent = t.label;
    b.dataset.tool = t.id;
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => selectTool(t));
    tabbar.appendChild(b);
  }
}

function selectTool(t: Tool): void {
  activeTool = t;
  activeKey = t.params[0].key;
  tabbar.querySelectorAll<HTMLButtonElement>(".tab").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.tool === t.id)),
  );
  buildSubrow();
  updateOverlayLabel();
  // depth tool → show the live focus map; lens tools → show the current result
  if (t.id === "depth") showDepthLive();
  else hideDepthLive();
}

function buildSubrow(): void {
  subrow.innerHTML = "";
  const multi = activeTool.params.length > 1;
  for (const p of activeTool.params) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = p.label;
    chip.dataset.key = p.key;
    chip.setAttribute("aria-pressed", String(p.key === activeKey));
    chip.addEventListener("click", () => {
      activeKey = p.key;
      subrow.querySelectorAll<HTMLButtonElement>(".chip[data-key]").forEach((c) =>
        c.setAttribute("aria-pressed", String(c.dataset.key === p.key)),
      );
      updateOverlayLabel();
    });
    subrow.appendChild(chip);
  }
  if (activeTool.shapes) buildShapeChips();
  subrow.classList.toggle("hidden", !multi && !activeTool.shapes);
}

function buildShapeChips(): void {
  const wrap = document.createElement("div");
  wrap.className = "shapes";
  const SHAPES: [number, string][] = [[0, "○"], [5, "5"], [6, "6"], [8, "8"]];
  for (const [n, glyph] of SHAPES) {
    const b = document.createElement("button");
    b.className = "chip shape";
    b.textContent = glyph;
    b.dataset.blades = String(n);
    b.setAttribute("aria-pressed", String(n === blades));
    b.addEventListener("click", () => {
      blades = n;
      wrap.querySelectorAll<HTMLButtonElement>("button").forEach((x) =>
        x.setAttribute("aria-pressed", String(Number(x.dataset.blades) === blades)),
      );
      scheduleRender();
    });
    wrap.appendChild(b);
  }
  subrow.appendChild(wrap);
}

function updateOverlayLabel(): void {
  const p = activeTool.params.find((x) => x.key === activeKey)!;
  toolLabel.textContent = p.label;
  setTicker(state[activeKey]);
}

// ---- the big drag slider ----
let dragging = false;
let dragStartY = 0;
let dragStartVal = 0;

function bindDrag(): void {
  dragSurface.style.touchAction = "none";
  dragSurface.addEventListener("pointerdown", (e) => {
    if (!editor.ready) return;
    e.preventDefault();
    dragging = true;
    dragStartY = e.clientY;
    dragStartVal = state[activeKey];
    dragSurface.classList.add("dragging");
    try {
      dragSurface.setPointerCapture(e.pointerId);
    } catch {
      /* fine */
    }
    setTicker(state[activeKey]);
  });
  dragSurface.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    // drag up = increase. Full 0→100 sweep over ~65% of the stage height.
    const span = Math.max(180, stage.clientHeight * 0.65);
    const delta = ((dragStartY - e.clientY) / span) * 100;
    const v = Math.min(100, Math.max(0, dragStartVal + delta));
    state[activeKey] = v;
    setTicker(v);
    if (activeTool.id === "depth" && !rafPending) {
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        editor.setSettings(depthSettings());
        showDepthLive();
      });
    }
  });
  const end = () => {
    if (!dragging) return;
    dragging = false;
    dragSurface.classList.remove("dragging");
    scheduleRender();
  };
  dragSurface.addEventListener("pointerup", end);
  dragSurface.addEventListener("pointercancel", end);
}

// ---- params ----
function depthSettings() {
  return {
    amount: state.amount / 100,
    position: state.position / 100,
    contrast: state.contrast / 100,
    falloff: state.falloff / 100,
  };
}

function renderParams(): RenderParams {
  return {
    k: state.k,
    disp_focus: editor.focalValue,
    autofocus: false,
    subject_dof: false,
    blades,
    highlight_boost: state.highlight / 100,
    cat_eye: 0.2,
    swirl: state.swirl / 100,
    sweet: state.sweet / 100,
    sweet_size: state.sweetSize / 100,
    halation: state.halation / 100,
    halation_size: state.halationSize / 100,
    ca: state.ca / 100,
  };
}

// ---- render (debounced on finger-lift) ----
function setProgress(label: string, show = true): void {
  progress.classList.toggle("hidden", !show);
  progressLabel.textContent = label;
}

function scheduleRender(): void {
  window.clearTimeout(renderTimer);
  renderTimer = window.setTimeout(doRender, 450);
}

async function doRender(): Promise<void> {
  if (!currentFile || !analyzeId || !editor.ready) return;
  inflight?.cancel();
  setProgress("Rendering…");
  try {
    const depthPng = await editor.exportDepthPng();
    inflight = renderFromAnalyze(analyzeId, depthPng, renderParams(), (p) =>
      setProgress(p.label),
    );
    const url = await inflight.done;
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = url;
    resultBlob = await (await fetch(url)).blob();
    if (activeTool.id !== "depth") showResult(url);
    else resultImg.src = url; // keep it ready behind the live depth view
    setProgress("", false);
    saveBtn.classList.remove("hidden");
    compareBtn.classList.remove("hidden");
  } catch (err) {
    setProgress("", false);
    if (!(err instanceof ApiError && err.message === "cancelled")) {
      toast(err instanceof ApiError ? err.message : "Something went wrong rendering.");
    }
  } finally {
    inflight = null;
  }
}

// ---- file intake → auto-analyze ----
async function acceptFile(file: File): Promise<void> {
  if (!file.type.startsWith("image/")) {
    toast("That doesn't look like an image.");
    return;
  }
  currentFile = file;
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);
  analyzeId = null;
  resultImg.src = originalUrl;
  resultImg.classList.remove("hidden");
  dropzone.classList.add("hidden");
  depthView.classList.add("hidden");
  dragSurface.classList.add("hidden");
  saveBtn.classList.add("hidden");
  compareBtn.classList.add("hidden");

  Object.assign(state, DEFAULTS);
  blades = 0;

  setProgress("Reading depth…");
  try {
    const { analyzeId: aid } = await analyze(file);
    analyzeId = aid;
    await editor.load(depthUrl(aid), matteUrl(aid));
    editor.setSettings(depthSettings());
    setProgress("", false);
    dragSurface.classList.remove("hidden");
    selectTool(TOOLS[0]);
    void doRender(); // first render so lens tools have something to show
    toast("Drag up/down on the photo to adjust. Tabs switch tools.");
  } catch (err) {
    setProgress("", false);
    dropzone.classList.remove("hidden");
    toast(err instanceof ApiError ? err.message : "Could not analyze the photo.");
  }
}

// ---- save (Web Share → Save to Photos; download fallback) ----
async function save(): Promise<void> {
  if (!resultBlob) return;
  const file = new File([resultBlob], "lensy.jpg", { type: "image/jpeg" });
  const nav = navigator as Navigator & {
    canShare?: (d: unknown) => boolean;
    share?: (d: unknown) => Promise<void>;
  };
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

// ---- compare: hold to peek the original ----
function bindCompare(): void {
  const showOrig = (on: boolean) => {
    if (!originalUrl || !resultUrl) return;
    if (on) {
      resultImg.src = originalUrl;
      resultImg.classList.remove("hidden");
      depthView.classList.add("hidden");
    } else {
      resultImg.src = resultUrl;
    }
  };
  compareBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    showOrig(true);
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach((ev) =>
    compareBtn.addEventListener(ev, () => showOrig(false)),
  );
}

// ---- wiring ----
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) void acceptFile(f);
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
  if (f) void acceptFile(f);
});
saveBtn.addEventListener("click", save);

buildTicker();
buildTabs();
bindDrag();
bindCompare();
selectTool(TOOLS[0]);
setupServerPanel();

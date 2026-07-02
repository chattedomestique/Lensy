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
  eraseObject,
  matteUrl,
  photoUrl,
  renderFromAnalyze,
  segment,
  selectSubject,
  type RenderHandle,
} from "./api";
import { DepthEditor } from "./depth";
import { EraseSelection } from "./erase";
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
const toolHint = $("tool-hint");
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
  | "ca" | "swirl" | "sweet" | "sweetSize" | "distortion"
  | "grain" | "grainSize";

const DEFAULTS: Record<Key, number> = {
  amount: 50, position: 50, contrast: 0, falloff: 20,
  k: 60, highlight: 0, halation: 0, halationSize: 40,
  ca: 0, swirl: 0, sweet: 0, sweetSize: 35, distortion: 0,
  grain: 0, grainSize: 40,
};
const state: Record<Key, number> = { ...DEFAULTS };
let blades = 0;

// The character effects live in the low end of their raw 0..1 range, so the on-screen 0..100 maps
// to a smaller backend ceiling per effect (UI 100 = that ceiling), giving fine control where it
// matters. Blur, the sizes, distortion, and the depth sliders keep (near-)full range.
const BACKEND_MAX: Partial<Record<Key, number>> = {
  highlight: 0.4, halation: 0.4, ca: 0.25, swirl: 0.25, sweet: 0.25, distortion: 1.0,
};
const backendVal = (key: Key): number => (state[key] / 100) * (BACKEND_MAX[key] ?? 1);

interface Tool {
  id: string;
  label: string;
  params: { key: Key; label: string }[];
  shapes?: boolean; // Bokeh shows an aperture-shape picker
  erase?: boolean; // object-removal mode (tap/brush select + erase), not a drag effect
  refine?: boolean; // spot edge refinement — brush a local depth fix (kills halos)
  subject?: boolean; // tap to select which person(s) are the subject
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
  { id: "subject", label: "Subject", params: [], subject: true },
  { id: "bokeh", label: "Bokeh", params: [{ key: "k", label: "Blur" }], shapes: true },
  { id: "bloom", label: "Bloom", params: [{ key: "highlight", label: "Bloom" }] },
  {
    id: "halation", label: "Halation",
    params: [
      { key: "halation", label: "Halation" },
      { key: "halationSize", label: "Spread" },
    ],
  },
  {
    id: "chroma", label: "Chroma",
    params: [
      { key: "ca", label: "Chromatic aberration" },
      { key: "distortion", label: "Lens distortion" },
    ],
  },
  { id: "petzval", label: "Petzval", params: [{ key: "swirl", label: "Swirl" }] },
  {
    id: "lensbaby", label: "Lensbaby",
    params: [
      { key: "sweet", label: "Lensbaby" },
      { key: "sweetSize", label: "Spot size" },
    ],
  },
  {
    id: "grain", label: "Grain",
    params: [
      { key: "grain", label: "Grain" },
      { key: "grainSize", label: "Grain size" },
    ],
  },
  { id: "refine", label: "Refine", params: [], refine: true },
  { id: "erase", label: "Erase", params: [], erase: true },
];
let refineMode: "sharpen" | "recede" | "dissolve" = "sharpen";
let eraseTarget: "auto" | "subject" | "background" = "auto";
let brushSize = 4; // % of the long edge
let brushHardness = 0.5; // 0 soft → 1 hard

let activeTool = TOOLS[0];
let activeKey: Key = "amount";

// ---- runtime ----
const editor = new DepthEditor();
const eraseSel = new EraseSelection();
const eraseLayer = $("erase-layer") as HTMLCanvasElement;
let currentFile: File | null = null;
let originalUrl: string | null = null;
let resultUrl: string | null = null;
let resultBlob: Blob | null = null;
let analyzeId: string | null = null;
let analyzeW = 0;
let analyzeH = 0;
let dataVersion = 0; // bumped after an erase so depth/matte/photo re-fetch past the cache
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
// Lean linear glyphs per tool (§5). currentColor stroke; the active tab brightens + underlines.
const SVG = (inner: string) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`;
const DOT = '<circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none"/>';
const ICONS: Record<string, string> = {
  // stacked planes = depth layers
  depth: SVG('<rect x="3.5" y="3.5" width="12" height="12" rx="2.5"/><rect x="8.5" y="8.5" width="12" height="12" rx="2.5"/>'),
  // person = subject select
  subject: SVG('<circle cx="12" cy="8" r="3.6"/><path d="M5.5 20a6.5 6.5 0 0 1 13 0"/>'),
  // out-of-focus bokeh balls
  bokeh: SVG('<circle cx="9" cy="10" r="4.3"/><circle cx="16.5" cy="14.5" r="3"/><circle cx="15.5" cy="7" r="1.8"/>'),
  // highlight starburst
  bloom: SVG(`${DOT}<path d="M12 3.5v2.5M12 18v2.5M3.5 12H6M18 12h2.5M6 6l1.7 1.7M16.3 16.3 18 18M18 6l-1.7 1.7M7.7 16.3 6 18"/>`),
  // highlight with a glow ring
  halation: SVG(`${DOT}<circle cx="12" cy="12" r="7"/>`),
  // offset rings = channel fringing
  chroma: SVG('<circle cx="9.5" cy="12" r="6"/><circle cx="14.5" cy="12" r="6"/>'),
  // swirl
  petzval: SVG('<path d="M12 4a8 8 0 1 1-7.6 5.6"/><path d="M12 8.2a3.8 3.8 0 1 0 3.6 2.7"/>'),
  // sweet-spot reticle
  lensbaby: SVG(`<circle cx="12" cy="12" r="8"/>${DOT}`),
  // film-grain speckle
  grain: SVG('<circle cx="6" cy="7" r="1.1" fill="currentColor" stroke="none"/><circle cx="12.5" cy="5.5" r="0.9" fill="currentColor" stroke="none"/><circle cx="18" cy="8" r="1.1" fill="currentColor" stroke="none"/><circle cx="8.5" cy="12.5" r="0.9" fill="currentColor" stroke="none"/><circle cx="15" cy="13.5" r="1.1" fill="currentColor" stroke="none"/><circle cx="5.5" cy="17.5" r="1.1" fill="currentColor" stroke="none"/><circle cx="11.5" cy="18.5" r="0.9" fill="currentColor" stroke="none"/><circle cx="17.5" cy="18" r="1.0" fill="currentColor" stroke="none"/>'),
  // refine sparkle
  refine: SVG('<path d="M12 3.5l1.9 4.6L18.5 10l-4.6 1.9L12 16.5l-1.9-4.6L5.5 10l4.6-1.9z"/><path d="M18.5 16.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>'),
  // eraser
  erase: SVG('<path d="M4.5 15.5 12 8l4.5 4.5L11 18H6.5z"/><path d="M9 20h10"/>'),
};

function buildTabs(): void {
  tabbar.innerHTML = "";
  for (const t of TOOLS) {
    const b = document.createElement("button");
    b.className = "tab";
    b.innerHTML = ICONS[t.id] ?? t.label;
    b.dataset.tool = t.id;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-label", t.label);
    b.title = t.label;
    b.addEventListener("click", () => selectTool(t));
    tabbar.appendChild(b);
  }
}

function selectTool(t: Tool): void {
  activeTool = t;
  activeKey = t.params[0]?.key ?? activeKey;
  tabbar.querySelectorAll<HTMLButtonElement>(".tab").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.tool === t.id)),
  );
  buildSubrow();
  updateOverlayLabel();
  // switching tools brings the floating instructions back
  dragSurface.classList.remove("dismissed", "dragging");
  // brush size/hardness bar only for the brush tools
  $("brush-bar").classList.toggle("hidden", !(t.erase || t.refine));
  if (t.erase) {
    enterEraseMode();
  } else if (t.refine) {
    enterRefineMode();
  } else if (t.subject) {
    enterSubjectMode();
  } else {
    eraseLayer.classList.add("hidden");
    // depth tool → show the live focus map; lens tools → show the current result
    if (t.id === "depth") showDepthLive();
    else hideDepthLive();
  }
}

function enterSubjectMode(): void {
  // tap the people you want in focus; resultImg already holds the current render — just show it
  // (don't reassign src: reloading it briefly zeroes the element size and a fast tap would miss)
  resultImg.classList.remove("hidden");
  depthView.classList.add("hidden");
  eraseLayer.classList.add("hidden");
}

function enterRefineMode(): void {
  if (!editor.ready) return;
  // brush on the rendered photo so the halo is visible; overlay shows what's painted
  resultImg.classList.remove("hidden");
  depthView.classList.add("hidden");
  editor.drawRefineOverlay(eraseLayer);
  eraseLayer.classList.remove("hidden");
}

function enterEraseMode(): void {
  if (!analyzeId) return;
  // select on the cleaned source photo (not the blurred render)
  resultImg.src = photoUrl(analyzeId, dataVersion);
  resultImg.classList.remove("hidden");
  depthView.classList.add("hidden");
  eraseSel.init(analyzeW, analyzeH, eraseLayer);
  eraseLayer.classList.remove("hidden");
}

function buildSubrow(): void {
  subrow.innerHTML = "";
  if (activeTool.erase) {
    buildEraseActions();
    subrow.classList.remove("hidden");
    return;
  }
  if (activeTool.refine) {
    buildRefineActions();
    subrow.classList.remove("hidden");
    return;
  }
  if (activeTool.subject) {
    const reset = document.createElement("button");
    reset.className = "chip";
    reset.textContent = "Reset to auto";
    reset.style.marginLeft = "auto";
    reset.addEventListener("click", () => void resetSubject());
    subrow.appendChild(reset);
    subrow.classList.remove("hidden");
    return;
  }
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
  if (activeTool.erase) {
    toolLabel.textContent = "Erase";
    toolHint.textContent = "tap an object, or brush over it, then Erase";
    return;
  }
  if (activeTool.refine) {
    toolLabel.textContent = "Refine";
    toolHint.textContent = "brush over a halo or bad edge to fix it";
    return;
  }
  if (activeTool.subject) {
    toolLabel.textContent = "Subject";
    toolHint.textContent = "tap each person you want in focus";
    return;
  }
  const p = activeTool.params.find((x) => x.key === activeKey)!;
  toolLabel.textContent = p.label;
  toolHint.textContent = "hold & drag up or down to adjust";
  setTicker(state[activeKey]);
}

// ---- the big drag slider ----
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const brushFrac = () => brushSize / 100; // brush radius as a fraction of the image's long edge

let dragging = false;
let moved = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartVal = 0;
let strokePainted = false;

/** Normalized coords within the displayed image (clamped). Works under zoom/pan because the
 * image's bounding rect already reflects the CSS transform on #stage-media. */
function normOf(e: PointerEvent): { nx: number; ny: number } {
  const rect = resultImg.getBoundingClientRect();
  return {
    nx: clamp01((e.clientX - rect.left) / Math.max(1, rect.width)),
    ny: clamp01((e.clientY - rect.top) / Math.max(1, rect.height)),
  };
}

// ---- pinch-zoom / pan (2-finger) ----
const stageMedia = $("stage-media");
const pointers = new Map<number, { x: number; y: number }>();
let zoom = 1;
let panX = 0;
let panY = 0;
let zooming = false;
let pinch = { dist: 1, zoom0: 1, lx: 0, ly: 0, rl: 0, rt: 0 };

function applyZoom(): void {
  stageMedia.style.transform = zoom === 1 && panX === 0 && panY === 0
    ? ""
    : `translate(${panX}px, ${panY}px) scale(${zoom})`;
}
function resetZoom(): void {
  zoom = 1;
  panX = 0;
  panY = 0;
  applyZoom();
}
function clampPan(): void {
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  panX = Math.min(0, Math.max(sw * (1 - zoom), panX));
  panY = Math.min(0, Math.max(sh * (1 - zoom), panY));
}
function pinchStart(): void {
  const pts = [...pointers.values()];
  if (pts.length < 2) return;
  const [a, b] = pts;
  const rect = stage.getBoundingClientRect();
  const mx = (a.x + b.x) / 2 - rect.left;
  const my = (a.y + b.y) / 2 - rect.top;
  pinch = {
    dist: Math.max(1, Math.hypot(b.x - a.x, b.y - a.y)),
    zoom0: zoom,
    lx: (mx - panX) / zoom, // the media-local point under the pinch midpoint
    ly: (my - panY) / zoom,
    rl: rect.left,
    rt: rect.top,
  };
  zooming = true;
}
function pinchMove(): void {
  const pts = [...pointers.values()];
  if (pts.length < 2) return;
  const [a, b] = pts;
  const dist = Math.hypot(b.x - a.x, b.y - a.y);
  const mx = (a.x + b.x) / 2 - pinch.rl;
  const my = (a.y + b.y) / 2 - pinch.rt;
  zoom = Math.min(6, Math.max(1, (pinch.zoom0 * dist) / pinch.dist));
  panX = mx - pinch.lx * zoom; // keep the same media point under the moving midpoint
  panY = my - pinch.ly * zoom;
  if (zoom <= 1.01) resetZoom();
  else {
    clampPan();
    applyZoom();
  }
}

function bindDrag(): void {
  dragSurface.style.touchAction = "none";
  dragSurface.addEventListener("pointerdown", (e) => {
    if (!editor.ready) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try {
      dragSurface.setPointerCapture(e.pointerId);
    } catch {
      /* fine */
    }
    if (pointers.size >= 2) {
      // a second finger → pinch/pan; abandon any in-progress single-finger tool action
      dragging = false;
      dragSurface.classList.remove("dragging");
      pinchStart();
      return;
    }
    e.preventDefault();
    dragging = true;
    moved = false;
    strokePainted = false;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartVal = state[activeKey];
    // touching the image dismisses the floating instructions so the photo is unobstructed
    dragSurface.classList.add("dismissed");
    if (!activeTool.erase) setTicker(state[activeKey]);
  });

  dragSurface.addEventListener("pointermove", (e) => {
    const p = pointers.get(e.pointerId);
    if (p) {
      p.x = e.clientX;
      p.y = e.clientY;
    }
    if (zooming || pointers.size >= 2) {
      pinchMove();
      return;
    }
    if (!dragging) return;
    if (!moved && Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY) > 3) moved = true;

    if (activeTool.erase) {
      if (!moved) return; // a drag = brush; a still tap = SAM2 select (on pointerup)
      const { nx, ny } = normOf(e);
      eraseSel.paint(nx, ny, brushFrac(), brushHardness, !strokePainted);
      strokePainted = true;
      return;
    }

    if (activeTool.refine) {
      const { nx, ny } = normOf(e); // paint even on a tap-dab, then commit on lift
      editor.paintRefine(nx, ny, brushFrac(), refineMode);
      editor.drawRefineOverlay(eraseLayer);
      strokePainted = true;
      return;
    }

    if (activeTool.subject) return; // tap-only; ignore drags

    // drag up = increase. Full 0→100 sweep over ~65% of the stage height.
    const span = Math.max(180, stage.clientHeight * 0.65);
    const delta = ((dragStartY - e.clientY) / span) * 100;
    if (moved) dragSurface.classList.add("dragging"); // reveal the odometer once we actually drag
    if (!moved) return;
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

  const end = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    if (zooming) {
      // a pinch finger lifted; stay out of tool mode until all fingers are up (avoids a stray brush)
      if (pointers.size < 2) zooming = false;
      dragging = false;
      return;
    }
    if (!dragging) return;
    dragging = false;
    dragSurface.classList.remove("dragging"); // hide the odometer; label stays dismissed
    if (activeTool.erase) {
      // a still tap selects the object under the finger (guard against a not-yet-laid-out image)
      if (!moved && resultImg.getBoundingClientRect().width > 4) void tapSelect(normOf(e));
      moved = false;
      return;
    }
    if (activeTool.subject) {
      // ignore taps until the image has laid out (a 0-size rect would map to the corner)
      if (!moved && resultImg.getBoundingClientRect().width > 4) void tapSubject(normOf(e));
      moved = false;
      return;
    }
    if (activeTool.refine) {
      if (!strokePainted) {
        const { nx, ny } = normOf(e); // a tap paints one dab
        editor.paintRefine(nx, ny, brushFrac(), refineMode);
        editor.drawRefineOverlay(eraseLayer);
      }
      editor.commit(); // recompute once, then render with the refined depth
      scheduleRender();
      moved = false;
      strokePainted = false;
      return;
    }
    if (moved) scheduleRender();
    moved = false;
  };
  dragSurface.addEventListener("pointerup", end);
  dragSurface.addEventListener("pointercancel", end);
}

// ---- erase (object removal) ----
function buildEraseActions(): void {
  const mk = (label: string, cls: string, on: () => void) => {
    const b = document.createElement("button");
    b.className = `chip ${cls}`;
    b.textContent = label;
    b.addEventListener("click", on);
    return b;
  };
  // target toggle: which layer the erase is allowed to touch
  const targets: [typeof eraseTarget, string][] = [
    ["auto", "Auto"],
    ["background", "Bg"],
    ["subject", "Subject"],
  ];
  for (const [t, label] of targets) {
    const chip = mk(label, "", () => {
      eraseTarget = t;
      subrow.querySelectorAll<HTMLButtonElement>(".chip[data-target]").forEach((c) =>
        c.setAttribute("aria-pressed", String(c.dataset.target === t)),
      );
    });
    chip.dataset.target = t;
    chip.setAttribute("aria-pressed", String(t === eraseTarget));
    subrow.appendChild(chip);
  }
  subrow.appendChild(mk("Undo", "", () => eraseSel.undo()));
  const erase = mk("Erase", "erase-go", () => void doErase());
  erase.style.marginLeft = "auto";
  subrow.appendChild(erase);
}

function buildRefineActions(): void {
  const modes: [typeof refineMode, string][] = [
    ["sharpen", "Sharpen"],
    ["recede", "Blur"],
    ["dissolve", "Dissolve"],
  ];
  for (const [m, label] of modes) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = label;
    chip.dataset.mode = m;
    chip.setAttribute("aria-pressed", String(m === refineMode));
    chip.addEventListener("click", () => {
      refineMode = m;
      subrow.querySelectorAll<HTMLButtonElement>(".chip[data-mode]").forEach((c) =>
        c.setAttribute("aria-pressed", String(c.dataset.mode === m)),
      );
    });
    subrow.appendChild(chip);
  }
  const clear = document.createElement("button");
  clear.className = "chip";
  clear.textContent = "Clear";
  clear.style.marginLeft = "auto";
  clear.addEventListener("click", () => {
    editor.clearRefine();
    editor.drawRefineOverlay(eraseLayer);
    scheduleRender();
  });
  subrow.appendChild(clear);
}

async function applySubject(points: [number, number, number][], reset: boolean): Promise<void> {
  if (!analyzeId) return;
  inflight?.cancel();
  setProgress(reset ? "Resetting…" : "Selecting subject…");
  try {
    await selectSubject(analyzeId, points, reset);
    dataVersion++;
    // matte changed → reload it so the depth editor re-centres focus on the new subject
    await editor.load(depthUrl(analyzeId, dataVersion), matteUrl(analyzeId, dataVersion));
    editor.setSettings(depthSettings());
    setProgress("", false);
    void doRender();
  } catch (err) {
    setProgress("", false);
    toast(err instanceof ApiError ? err.message : "Couldn't set the subject.");
  }
}

const tapSubject = (p: { nx: number; ny: number }) => applySubject([[p.nx, p.ny, 1]], false);
const resetSubject = () => applySubject([], true);

async function tapSelect(p: { nx: number; ny: number }): Promise<void> {
  if (!analyzeId) return;
  setProgress("Selecting…");
  try {
    const img = await segment(analyzeId, [[p.nx, p.ny, 1]]);
    eraseSel.addMaskImage(img);
  } catch (err) {
    toast(err instanceof ApiError ? err.message : "Couldn't select that.");
  } finally {
    setProgress("", false);
  }
}

async function doErase(): Promise<void> {
  if (!analyzeId || eraseSel.isEmpty()) {
    toast("Tap or brush what you want to remove first.");
    return;
  }
  inflight?.cancel();
  setProgress("Erasing…");
  try {
    const mask = await eraseSel.exportPng();
    await eraseObject(analyzeId, mask, eraseTarget);
    dataVersion++;
    // the scene changed — reload depth/matte and the cleaned source photo
    await editor.load(depthUrl(analyzeId, dataVersion), matteUrl(analyzeId, dataVersion));
    editor.setSettings(depthSettings());
    resultImg.src = photoUrl(analyzeId, dataVersion);
    eraseSel.init(analyzeW, analyzeH, eraseLayer); // reset selection for more removals
    setProgress("", false);
    void doRender(); // refresh the cached render behind the scenes
    toast("Erased. Select more, or switch tools to style it.");
  } catch (err) {
    setProgress("", false);
    toast(err instanceof ApiError ? err.message : "Erase failed.");
  }
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
    highlight_boost: backendVal("highlight"),
    cat_eye: 0.2,
    swirl: backendVal("swirl"),
    sweet: backendVal("sweet"),
    sweet_size: state.sweetSize / 100,
    halation: backendVal("halation"),
    halation_size: state.halationSize / 100,
    ca: backendVal("ca"),
    distortion: backendVal("distortion"),
    grain: state.grain / 100,
    grain_size: state.grainSize / 100,
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
    if (activeTool.erase) {
      /* keep the cleaned source shown for selecting; result stays cached for save/other tabs */
    } else if (activeTool.id === "depth") {
      resultImg.src = url; // keep it ready behind the live depth view
    } else {
      showResult(url);
    }
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
  dataVersion = 0;
  resetZoom();
  resultImg.src = originalUrl;
  resultImg.classList.remove("hidden");
  dropzone.classList.add("hidden");
  depthView.classList.add("hidden");
  eraseLayer.classList.add("hidden");
  dragSurface.classList.add("hidden");
  saveBtn.classList.add("hidden");
  compareBtn.classList.add("hidden");

  Object.assign(state, DEFAULTS);
  blades = 0;

  setProgress("Reading depth…");
  try {
    const { analyzeId: aid, width, height } = await analyze(file);
    analyzeId = aid;
    analyzeW = width;
    analyzeH = height;
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

// brush size / hardness
const brushSizeInput = $("brush-size") as HTMLInputElement;
const brushHardnessInput = $("brush-hardness") as HTMLInputElement;
brushSizeInput.addEventListener("input", () => (brushSize = Number(brushSizeInput.value)));
brushHardnessInput.addEventListener("input", () => (brushHardness = Number(brushHardnessInput.value) / 100));

buildTicker();
buildTabs();
bindDrag();
bindCompare();
selectTool(TOOLS[0]);
setupServerPanel();

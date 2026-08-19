// Client-side DEPTH editor — physical model.
//
// The depth map (from Depth Pro / Depth Anything) is GROUND TRUTH geometry and is what gets sent
// to the renderer: near = 1, far = 0. Nothing here fabricates a "focus map" — focus is a *lens
// parameter*, not a paint layer, so everything in the frame (the subject included) obeys
//
//     CoC = K · falloff(|depth − focus|)
//
// computed by the backend from this real depth. The controls map to real lens quantities:
//
//   position — the focal plane's location in depth (0.5 = the subject's own plane; tap-to-focus
//              sets it from the tapped pixel's depth). Sent as `disp_focus`.
//   range    — width of the in-focus zone (depth-of-field dead band). Sent as `focus_range`.
//   falloff  — how abruptly blur ramps in past that zone. Sent as `focus_gamma`.
//   smooth   — optional feathering of depth discontinuities (a mild blur of the exported map).
//
// Brush edits are DEPTH CORRECTIONS, not focus overrides: painting blends a region's depth toward
// a target distance (the focal plane / farther / nearer) with a soft, hardness-controlled falloff.
// The region then blurs (or doesn't) because of where it now *is*, exactly like a lens — move the
// focal plane onto it later and it comes back into focus. No pure-black/white stamping.

export interface FocusSettings {
  position: number; // 0..1 — focal-plane position (0.5 = subject plane)
  range: number; // 0..1 — in-focus band width (→ backend focus_range 0..0.32)
  falloff: number; // 0..1 — focus-falloff contrast (→ backend focus_gamma 0.45..2.2)
  smooth: number; // 0..1 — feathering of depth discontinuities in the exported map
}

export const DEFAULT_SETTINGS: FocusSettings = {
  position: 0.5,
  range: 0.37, // ≈ the renderer's default focus_range of 0.12
  falloff: 0.5, // gamma 1.0 — the plain physical ramp
  smooth: 0,
};

export type DepthBrushMode = "focus" | "far" | "near" | "clear";

const clampIdx = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function loadImg(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

function toGray(img: HTMLImageElement | HTMLCanvasElement, w: number, h: number): Float32Array {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = data[i * 4] / 255;
  return out;
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  if (r < 1) return src.slice();
  const win = 2 * r + 1;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[row + clampIdx(x, 0, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / win;
      sum += src[row + clampIdx(x + r + 1, 0, w - 1)] - src[row + clampIdx(x - r, 0, w - 1)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[clampIdx(y, 0, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / win;
      sum += tmp[clampIdx(y + r + 1, 0, h - 1) * w + x] - tmp[clampIdx(y - r, 0, h - 1) * w + x];
    }
  }
  return out;
}

function percentile(src: Float32Array, p: number): number {
  const step = Math.max(1, Math.floor(src.length / 4096)); // sampled — plenty for a reference depth
  const vals: number[] = [];
  for (let i = 0; i < src.length; i += step) vals.push(src[i]);
  vals.sort((a, b) => a - b);
  return vals[clampIdx(Math.round(p * (vals.length - 1)), 0, vals.length - 1)];
}

// overlay tints per brush mode (matches the app accents)
const MODE_TINT: Record<number, [number, number, number]> = {
  1: [111, 182, 214], // focus — slate/cyan
  2: [207, 138, 95], // far — terracotta
  3: [168, 120, 200], // near — violet
};

export class DepthEditor {
  private w = 0;
  private h = 0;
  private rawDepth: Float32Array = new Float32Array(); // near = 1 (model output, real geometry)
  private matte: Float32Array = new Float32Array();
  // depth-correction layer: per-pixel blend toward a painted target distance.
  private editTarget: Float32Array = new Float32Array();
  private editStrength: Float32Array = new Float32Array(); // 0 = untouched → 1 = fully re-assigned
  private editMode: Uint8Array = new Uint8Array(); // 1 focus / 2 far / 3 near (for the overlay tint)
  private subjectDepth = 0.5; // median depth under the matte — the autofocus plane
  private farRef = 0.05; // the scene's far field (2nd percentile)
  private nearRef = 0.95; // the scene's near field (98th percentile)

  settings: FocusSettings = { ...DEFAULT_SETTINGS };

  get ready(): boolean {
    return this.w > 0;
  }
  get width(): number {
    return this.w;
  }
  get height(): number {
    return this.h;
  }

  /** The focal plane's depth value (near = 1), in the same space as the exported map. */
  get focusValue(): number {
    return clamp01(this.subjectDepth + (this.settings.position - 0.5) * 0.9);
  }
  /** Backend focus_range: width of the in-focus zone in normalized disparity. */
  get focusRange(): number {
    return 0.005 + this.settings.range * 0.315;
  }
  /** Backend focus_gamma: falloff 0 → 0.45 (blur ramps in early/gently), 1 → 2.2 (late/hard). */
  get focusGamma(): number {
    return Math.pow(2.2, (this.settings.falloff - 0.5) / 0.5);
  }
  get hasEdits(): boolean {
    for (let i = 0; i < this.editStrength.length; i++) if (this.editStrength[i] > 0.01) return true;
    return false;
  }

  async load(depthSrc: string, matteSrc: string, maxEdge = 768): Promise<void> {
    const [d, m] = await Promise.all([loadImg(depthSrc), loadImg(matteSrc)]);
    const scale = Math.min(1, maxEdge / Math.max(d.width, d.height));
    this.w = Math.max(1, Math.round(d.width * scale));
    this.h = Math.max(1, Math.round(d.height * scale));
    this.rawDepth = toGray(d, this.w, this.h);
    this.matte = toGray(m, this.w, this.h);
    this.editTarget = new Float32Array(this.w * this.h);
    this.editStrength = new Float32Array(this.w * this.h);
    this.editMode = new Uint8Array(this.w * this.h);
    this.settings = { ...DEFAULT_SETTINGS };
    // subject plane = median depth under the matte (fallback: middle of the scene's range)
    const vals: number[] = [];
    for (let i = 0; i < this.w * this.h; i++) if (this.matte[i] > 0.5) vals.push(this.rawDepth[i]);
    this.farRef = percentile(this.rawDepth, 0.02);
    this.nearRef = Math.max(0.9, percentile(this.rawDepth, 0.98));
    if (vals.length > 16) {
      vals.sort((a, b) => a - b);
      this.subjectDepth = vals[vals.length >> 1];
    } else {
      this.subjectDepth = (this.farRef + this.nearRef) / 2;
    }
  }

  setSettings(s: FocusSettings): void {
    this.settings = s;
  }

  /** Edited depth at a normalized point (small window average) — for tap-to-focus. */
  depthAt(nx: number, ny: number): number {
    const cx = clampIdx(Math.round(nx * this.w), 0, this.w - 1);
    const cy = clampIdx(Math.round(ny * this.h), 0, this.h - 1);
    const r = Math.max(2, Math.round(Math.max(this.w, this.h) * 0.008));
    let sum = 0;
    let cnt = 0;
    for (let y = Math.max(0, cy - r); y <= Math.min(this.h - 1, cy + r); y++) {
      for (let x = Math.max(0, cx - r); x <= Math.min(this.w - 1, cx + r); x++) {
        sum += this.depthPx(y * this.w + x);
        cnt++;
      }
    }
    return cnt ? sum / cnt : 0.5;
  }

  /** The `position` slider value (0..1) that puts the focal plane at depth d — for tap-to-focus. */
  positionForDepth(d: number): number {
    return clamp01(0.5 + (clamp01(d) - this.subjectDepth) / 0.9);
  }

  private depthPx(i: number): number {
    const s = this.editStrength[i];
    return s > 0 ? this.rawDepth[i] * (1 - s) + this.editTarget[i] * s : this.rawDepth[i];
  }

  private targetFor(mode: DepthBrushMode): number {
    if (mode === "focus") return this.focusValue; // assert: this thing sits at the focal plane
    if (mode === "far") return this.farRef; // assert: it's actually in the far field
    return this.nearRef; // "near": pull it to the near field (occluder melt)
  }

  /** Paint a soft depth correction. Hardness shapes the dab: 1 ≈ crisp edge, 0 = long feather.
   * The dab writes blend STRENGTH toward the mode's target depth — never a hard 0/1 stamp. */
  paintDepth(nx: number, ny: number, radiusFrac: number, hardness: number, mode: DepthBrushMode): void {
    const cx = nx * this.w;
    const cy = ny * this.h;
    const r = Math.max(2, radiusFrac * Math.max(this.w, this.h));
    const feather = Math.max(0.12, 1 - clamp01(hardness)); // fraction of the radius that ramps
    const target = mode === "clear" ? 0 : this.targetFor(mode);
    const modeId = mode === "focus" ? 1 : mode === "far" ? 2 : mode === "near" ? 3 : 0;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(this.w - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(this.h - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > r) continue;
        // soft radial falloff: full strength in the core, smoothstep ramp across the feather band
        let t = (1 - dist / r) / feather;
        t = clamp01(t);
        const a = t * t * (3 - 2 * t);
        if (a <= 0) continue;
        const i = y * this.w + x;
        if (mode === "clear") {
          this.editStrength[i] *= 1 - a;
          if (this.editStrength[i] < 0.01) this.editMode[i] = 0;
          continue;
        }
        if (a >= this.editStrength[i]) {
          this.editTarget[i] = target;
          this.editMode[i] = modeId;
          this.editStrength[i] = a;
        }
      }
    }
  }

  /** Apply a selection mask (e.g. SAM2, white = selected) as a feathered depth correction — the
   * "tap an object → put it at a distance" path. The mask edge is feathered so the corrected
   * region blends into the surrounding geometry instead of stamping a hard depth cliff. */
  applyMaskDepth(mask: HTMLImageElement, mode: DepthBrushMode): void {
    if (!this.ready) return;
    const sel = toGray(mask, this.w, this.h);
    const r = Math.max(1, Math.round(Math.max(this.w, this.h) * 0.006));
    const soft = boxBlur(sel, this.w, this.h, r);
    const target = mode === "clear" ? 0 : this.targetFor(mode);
    const modeId = mode === "focus" ? 1 : mode === "far" ? 2 : mode === "near" ? 3 : 0;
    for (let i = 0; i < this.w * this.h; i++) {
      const a = clamp01(soft[i]);
      if (a <= 0.02) continue;
      if (mode === "clear") {
        this.editStrength[i] *= 1 - a;
        if (this.editStrength[i] < 0.01) this.editMode[i] = 0;
      } else if (a >= this.editStrength[i]) {
        this.editTarget[i] = target;
        this.editMode[i] = modeId;
        this.editStrength[i] = a;
      }
    }
  }

  clearEdits(): void {
    if (!this.hasEdits) return;
    this.editStrength.fill(0);
    this.editMode.fill(0);
  }

  /** Tint the painted corrections over the photo, alpha-weighted by their strength. */
  drawEditOverlay(canvas: HTMLCanvasElement): void {
    canvas.width = this.w;
    canvas.height = this.h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < this.w * this.h; i++) {
      const s = this.editStrength[i];
      const tint = MODE_TINT[this.editMode[i]];
      if (s < 0.02 || !tint) continue;
      img.data[i * 4] = tint[0];
      img.data[i * 4 + 1] = tint[1];
      img.data[i * 4 + 2] = tint[2];
      img.data[i * 4 + 3] = Math.round(40 + 110 * s);
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Live focus preview: the PREDICTED sharpness field, from the same CoC model the renderer
   * uses — white = in focus, darker = more blur, with real gradients (never a binary stamp).
   * The subject is NOT forced white: move the focal plane off it and it visibly defocuses. */
  drawFocus(canvas: HTMLCanvasElement): void {
    canvas.width = this.w;
    canvas.height = this.h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    const f = this.focusValue;
    const range = this.focusRange;
    const gamma = this.focusGamma;
    const norm = Math.max(f, 1 - f, 1e-3);
    for (let i = 0; i < this.w * this.h; i++) {
      const d = this.depthPx(i);
      const eff = clamp01((Math.abs(d - f) - range) / norm);
      const coc = Math.pow(eff, gamma); // mirror of blur.focal_radius (non-metric path)
      const v = (255 * (1 - coc) + 0.5) | 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Export the REAL (edited) depth map for the renderer: near = 1, far = 0. Optional `smooth`
   * feathers depth discontinuities with a mild blur — geometry averaging, not focus painting. */
  async exportDepthPng(): Promise<Blob> {
    const n = this.w * this.h;
    let depth: Float32Array = new Float32Array(n);
    for (let i = 0; i < n; i++) depth[i] = this.depthPx(i);
    if (this.settings.smooth > 0.01) {
      const r = Math.max(1, Math.round(this.settings.smooth * Math.max(this.w, this.h) * 0.02));
      depth = boxBlur(depth, this.w, this.h, r);
    }
    const c = document.createElement("canvas");
    c.width = this.w;
    c.height = this.h;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < n; i++) {
      const v = (clamp01(depth[i]) * 255 + 0.5) | 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return await new Promise<Blob>((resolve) => c.toBlob((b) => resolve(b!), "image/png"));
  }
}

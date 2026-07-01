// Client-side depth-map editor. Loads the analyzed depth + matte, lets you reshape the depth
// map live (levels/"expander", contrast, background smoothing), and exports the edited map as a
// PNG the backend renders from. Works at a reduced resolution (depth is smooth, so this is plenty
// and keeps every slider move instant); the backend upscales it to the working resolution.

export interface DepthAdjust {
  black: number; // 0..1 — anything at/below maps to 0 (farthest)
  white: number; // 0..1 — anything at/above maps to 1 (nearest); pulling this in = "expander"
  contrast: number; // 0..1 — steepen separation around the midpoint
  smoothing: number; // 0..1 — smooth the BACKGROUND gradient (subject depth kept crisp)
}

export const DEFAULT_ADJUST: DepthAdjust = { black: 0, white: 1, contrast: 0, smoothing: 0 };

const clampIdx = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

function loadImg(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

function toGray(img: HTMLImageElement, w: number, h: number): Float32Array {
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

/** Separable sliding-window box blur on a single-channel float image. */
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

export class DepthEditor {
  private w = 0;
  private h = 0;
  private rawDepth: Float32Array = new Float32Array();
  private matte: Float32Array = new Float32Array();
  private adjusted: Float32Array = new Float32Array();

  get ready(): boolean {
    return this.w > 0;
  }

  async load(depthSrc: string, matteSrc: string, maxEdge = 768): Promise<void> {
    const [d, m] = await Promise.all([loadImg(depthSrc), loadImg(matteSrc)]);
    const scale = Math.min(1, maxEdge / Math.max(d.width, d.height));
    this.w = Math.max(1, Math.round(d.width * scale));
    this.h = Math.max(1, Math.round(d.height * scale));
    this.rawDepth = toGray(d, this.w, this.h);
    this.matte = toGray(m, this.w, this.h);
    this.adjusted = new Float32Array(this.w * this.h);
    this.apply(DEFAULT_ADJUST);
  }

  /** Recompute the adjusted depth from the raw map + settings. */
  apply(a: DepthAdjust): void {
    const n = this.w * this.h;
    const span = Math.max(a.white - a.black, 1e-3);
    const cf = 1 + a.contrast * 2.5; // contrast steepness
    const tmp = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let d = (this.rawDepth[i] - a.black) / span; // levels / expander
      d = d < 0 ? 0 : d > 1 ? 1 : d;
      d = 0.5 + (d - 0.5) * cf; // contrast around mid
      tmp[i] = d < 0 ? 0 : d > 1 ? 1 : d;
    }
    if (a.smoothing > 0.001) {
      const r = Math.max(1, Math.round(a.smoothing * Math.max(this.w, this.h) * 0.06));
      const blurred = boxBlur(tmp, this.w, this.h, r);
      for (let i = 0; i < n; i++) {
        const bg = 1 - this.matte[i]; // smooth only the background; keep the subject crisp
        this.adjusted[i] = tmp[i] * (1 - bg * a.smoothing) + blurred[i] * (bg * a.smoothing);
      }
    } else {
      this.adjusted.set(tmp);
    }
  }

  /** Paint the current adjusted depth (grayscale) into a canvas. */
  drawTo(canvas: HTMLCanvasElement): void {
    canvas.width = this.w;
    canvas.height = this.h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < this.w * this.h; i++) {
      const v = (this.adjusted[i] * 255 + 0.5) | 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Read the adjusted depth value (0..1) at normalized coords — for tap-to-focus / anchors. */
  sampleAt(nx: number, ny: number): number {
    const x = clampIdx(Math.round(nx * (this.w - 1)), 0, this.w - 1);
    const y = clampIdx(Math.round(ny * (this.h - 1)), 0, this.h - 1);
    return this.adjusted[y * this.w + x];
  }

  async exportPng(): Promise<Blob> {
    const c = document.createElement("canvas");
    this.drawTo(c);
    return await new Promise<Blob>((resolve) => c.toBlob((b) => resolve(b!), "image/png"));
  }
}

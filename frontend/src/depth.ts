// Client-side FOCUS-MAP editor. The subject is the brightest (in focus); anything nearer OR
// farther fades to black (more blur). You place four anchors — subject / foreground / midground
// / background — by tapping, and the map is built from them + two global sliders. Everything is
// instant (works at a reduced res; the backend upscales). What's exported to the backend is a
// depth map centered on the subject (subject = 0.5, nearer → 1, farther → 0) so the renderer's
// |depth − focus| gives exactly the blur shown, with front/back preserved for occlusion.

export type AnchorName = "subject" | "foreground" | "background";

export interface DepthSettings {
  separation: number; // 0..1 — steepen the in-focus ↔ blurred split
  smoothing: number; // 0..1 — smooth the background gradient (subject kept crisp)
}

export const DEFAULT_SETTINGS: DepthSettings = { separation: 0, smoothing: 0 };

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
  private rawDepth: Float32Array = new Float32Array(); // near = 1
  private matte: Float32Array = new Float32Array();
  private focus: Float32Array = new Float32Array(); // 1 = in focus (white), 0 = max blur
  private edited: Float32Array = new Float32Array(); // subject=0.5, near→1, far→0 (for backend)

  anchors: Partial<Record<AnchorName, number>> = {}; // stored raw-depth values
  settings: DepthSettings = { ...DEFAULT_SETTINGS };

  get ready(): boolean {
    return this.w > 0;
  }
  get focalValue(): number {
    return 0.5; // subject is mapped to the middle of the edited depth
  }

  async load(depthSrc: string, matteSrc: string, maxEdge = 768): Promise<void> {
    const [d, m] = await Promise.all([loadImg(depthSrc), loadImg(matteSrc)]);
    const scale = Math.min(1, maxEdge / Math.max(d.width, d.height));
    this.w = Math.max(1, Math.round(d.width * scale));
    this.h = Math.max(1, Math.round(d.height * scale));
    this.rawDepth = toGray(d, this.w, this.h);
    this.matte = toGray(m, this.w, this.h);
    this.focus = new Float32Array(this.w * this.h);
    this.edited = new Float32Array(this.w * this.h);
    this.anchors = {};
    this.settings = { ...DEFAULT_SETTINGS };
    this.recompute();
  }

  sampleRaw(nx: number, ny: number): number {
    const x = clampIdx(Math.round(nx * (this.w - 1)), 0, this.w - 1);
    const y = clampIdx(Math.round(ny * (this.h - 1)), 0, this.h - 1);
    return this.rawDepth[y * this.w + x];
  }

  setAnchor(name: AnchorName, nx: number, ny: number): void {
    this.anchors[name] = this.sampleRaw(nx, ny);
    this.recompute();
  }
  setSettings(s: DepthSettings): void {
    this.settings = s;
    this.recompute();
  }

  /** Anchor depths, filling in sensible defaults from the raw map + matte. */
  private resolvedAnchors(): { s: number; f: number; b: number } {
    let subjSum = 0;
    let subjN = 0;
    let mn = 1;
    let mx = 0;
    for (let i = 0; i < this.w * this.h; i++) {
      const d = this.rawDepth[i];
      if (d < mn) mn = d;
      if (d > mx) mx = d;
      if (this.matte[i] > 0.5) {
        subjSum += d;
        subjN++;
      }
    }
    const autoS = subjN > 0 ? subjSum / subjN : (mn + mx) / 2;
    const s = this.anchors.subject ?? autoS;
    let f = this.anchors.foreground ?? mx; // nearest
    let b = this.anchors.background ?? mn; // farthest
    f = Math.max(f, s + 1e-3);
    b = Math.min(b, s - 1e-3);
    return { s, f, b };
  }

  private recompute(): void {
    const n = this.w * this.h;
    const { s, f, b } = this.resolvedAnchors();
    const sep = 1 + this.settings.separation * 2.2;

    // 1) background focus field — the falloff for everything OUTSIDE the subject. Separation and
    //    smoothing act here only, so the subject (forced white below) is never touched.
    const bgf = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const d = this.rawDepth[i];
      let v = d >= s ? 1 - (d - s) / Math.max(f - s, 1e-3) : 1 - (s - d) / Math.max(s - b, 1e-3);
      v = clamp01(v);
      bgf[i] = clamp01(0.5 + (v - 0.5) * sep); // separation steepens the fg↔bg split (bg only)
    }

    // 2) smoothing — a matte-EXCLUDING blur so the subject's white never bleeds into the gradient;
    //    it only blends foreground↔background.
    if (this.settings.smoothing > 0.001) {
      const r = Math.max(1, Math.round(this.settings.smoothing * Math.max(this.w, this.h) * 0.06));
      const wmap = new Float32Array(n);
      const wbgf = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const w = 1 - this.matte[i]; // background weight
        wmap[i] = w;
        wbgf[i] = bgf[i] * w;
      }
      const num = boxBlur(wbgf, this.w, this.h, r);
      const den = boxBlur(wmap, this.w, this.h, r);
      const a = this.settings.smoothing;
      for (let i = 0; i < n; i++) {
        const sm = den[i] > 1e-3 ? num[i] / den[i] : bgf[i];
        bgf[i] = bgf[i] * (1 - a) + sm * a;
      }
    }

    // 3) blend: the subject (matte) is always in focus (white); the rest uses the falloff.
    for (let i = 0; i < n; i++) {
      const mt = this.matte[i];
      const fo = clamp01(mt + (1 - mt) * bgf[i]);
      this.focus[i] = fo;
      const blur = 1 - fo;
      this.edited[i] = this.rawDepth[i] >= s ? 0.5 + 0.5 * blur : 0.5 - 0.5 * blur;
    }
  }

  /** Paint the focus map (white = in focus) for the Depth view. */
  drawFocus(canvas: HTMLCanvasElement): void {
    canvas.width = this.w;
    canvas.height = this.h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < this.w * this.h; i++) {
      const v = (this.focus[i] * 255 + 0.5) | 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  async exportDepthPng(): Promise<Blob> {
    const c = document.createElement("canvas");
    c.width = this.w;
    c.height = this.h;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < this.w * this.h; i++) {
      const v = (this.edited[i] * 255 + 0.5) | 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return await new Promise<Blob>((resolve) => c.toBlob((b) => resolve(b!), "image/png"));
  }
}

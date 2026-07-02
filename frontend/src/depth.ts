// Client-side FOCUS-MAP editor, driven by four continuous sliders (no anchor tapping).
// The subject (matte) is always in focus (white); everything nearer OR farther fades to black
// (more blur). The four controls shape that falloff:
//
//   amount    — width of the separation between the sharp subject and the blurred surround.
//               Higher = a narrower in-focus band = more of the scene blurred (stronger pop).
//   position  — slides the focal plane through depth; moves the sharp band toward the
//               foreground or the background together.
//   contrast  — a levels curve on the focus map: darks darker, lights lighter, so the split
//               between in-focus and out-of-focus is crisper.
//   falloff   — feathering of the blur gradient (a matte-excluding blur; subject stays crisp).
//
// Everything runs instantly at reduced resolution; the backend upscales. What's exported is a
// depth map centered on the subject (subject = 0.5, nearer → 1, farther → 0) so the renderer's
// |depth − focus| gives exactly the blur shown, with front/back preserved for occlusion.

export interface DepthSettings {
  amount: number; // 0..1 — width of fg/bg separation (higher = narrower in-focus band)
  position: number; // 0..1 — focal-plane position (0.5 = subject plane)
  contrast: number; // 0..1 — levels/contrast on the focus map
  falloff: number; // 0..1 — feathering of the blur gradient
}

export const DEFAULT_SETTINGS: DepthSettings = {
  amount: 0.5,
  position: 0.5,
  contrast: 0,
  falloff: 0.2,
};

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
  private subjectDepth = 0.5; // auto median depth under the matte
  // spot refinement: per-pixel local depth override painted by the Refine brush.
  //   +1 = force in-focus (sharp) — kills foreground occlusion halos
  //   -1 = force max blur (recede)
  private refine: Int8Array = new Int8Array();

  settings: DepthSettings = { ...DEFAULT_SETTINGS };

  get ready(): boolean {
    return this.w > 0;
  }
  get width(): number {
    return this.w;
  }
  get height(): number {
    return this.h;
  }
  get focalValue(): number {
    return 0.5; // subject is mapped to the middle of the edited depth
  }
  get hasRefine(): boolean {
    for (let i = 0; i < this.refine.length; i++) if (this.refine[i]) return true;
    return false;
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
    this.refine = new Int8Array(this.w * this.h);
    this.settings = { ...DEFAULT_SETTINGS };
    // auto subject plane = median depth under the matte (fallback: mid of the range)
    let sum = 0;
    let cnt = 0;
    let mn = 1;
    let mx = 0;
    for (let i = 0; i < this.w * this.h; i++) {
      const dv = this.rawDepth[i];
      if (dv < mn) mn = dv;
      if (dv > mx) mx = dv;
      if (this.matte[i] > 0.5) {
        sum += dv;
        cnt++;
      }
    }
    this.subjectDepth = cnt > 0 ? sum / cnt : (mn + mx) / 2;
    this.recompute();
  }

  setSettings(s: DepthSettings): void {
    this.settings = s;
    this.recompute();
  }

  private recompute(): void {
    const n = this.w * this.h;
    const { amount, position, contrast, falloff } = this.settings;

    // focal plane: position slides the sharp band through depth (0.5 = the subject's own plane)
    const s = clamp01(this.subjectDepth + (position - 0.5) * 0.9);
    // width of the in-focus band: more "amount" → narrower band → stronger separation
    const width = 0.9 - amount * 0.78; // 0.9 (soft) → 0.12 (hard)
    const cGain = 1 + contrast * 3; // levels curve steepness

    // 1) background focus field (everything OUTSIDE the subject). Triangular falloff around s,
    //    then a levels curve. Subject is forced white in step 3, so this never touches it.
    const bgf = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const d = this.rawDepth[i];
      let v = 1 - Math.abs(d - s) / Math.max(width, 1e-3);
      v = clamp01(v);
      v = clamp01(0.5 + (v - 0.5) * cGain); // contrast: darks darker, lights lighter
      bgf[i] = v;
    }

    // 2) falloff — feather the gradient with a matte-EXCLUDING blur so the subject's white can't
    //    bleed into the surround; it only softens the foreground↔background transition.
    if (falloff > 0.001) {
      const r = Math.max(1, Math.round(falloff * Math.max(this.w, this.h) * 0.06));
      const wmap = new Float32Array(n);
      const wbgf = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const wt = 1 - this.matte[i];
        wmap[i] = wt;
        wbgf[i] = bgf[i] * wt;
      }
      const num = boxBlur(wbgf, this.w, this.h, r);
      const den = boxBlur(wmap, this.w, this.h, r);
      for (let i = 0; i < n; i++) {
        bgf[i] = den[i] > 1e-3 ? num[i] / den[i] : bgf[i];
      }
    }

    // 3) blend: subject (matte) always in focus (white); the rest uses the falloff field.
    //    Export map centers the subject at 0.5, with near→1 / far→0 for occlusion ordering.
    for (let i = 0; i < n; i++) {
      const mt = this.matte[i];
      const fo = clamp01(mt + (1 - mt) * bgf[i]);
      this.focus[i] = fo;
      const blur = 1 - fo;
      this.edited[i] = this.rawDepth[i] >= s ? 0.5 + 0.5 * blur : 0.5 - 0.5 * blur;
    }

    // 4) spot refinement overrides (persist across slider edits). +1 → in focus (sharp);
    //    -1 → max blur, keeping the near/far side so occlusion ordering is preserved.
    for (let i = 0; i < n; i++) {
      if (this.refine[i] === 1) {
        this.focus[i] = 1;
        this.edited[i] = 0.5;
      } else if (this.refine[i] === -1) {
        this.focus[i] = 0;
        this.edited[i] = this.rawDepth[i] >= s ? 1.0 : 0.0;
      }
    }
  }

  /** Paint a spot refinement (disc, nx/ny normalized). mode: sharpen (+1) / recede (-1) / clear (0). */
  paintRefine(nx: number, ny: number, radiusFrac: number, mode: "sharpen" | "recede" | "clear"): void {
    const val = mode === "sharpen" ? 1 : mode === "recede" ? -1 : 0;
    const cx = nx * this.w;
    const cy = ny * this.h;
    const r = Math.max(2, radiusFrac * Math.max(this.w, this.h));
    const r2 = r * r;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(this.w - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(this.h - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy <= r2) this.refine[y * this.w + x] = val as number;
      }
    }
    // no recompute here — the brush repaints many times per stroke; call commit() on finger-lift
  }

  /** Recompute the maps after a refinement stroke (call once when the brush lifts). */
  commit(): void {
    this.recompute();
  }

  clearRefine(): void {
    if (!this.hasRefine) return;
    this.refine.fill(0);
    this.recompute();
  }

  /** Tint the painted refinement regions over the photo (cyan = sharpen, warm = recede). */
  drawRefineOverlay(canvas: HTMLCanvasElement): void {
    canvas.width = this.w;
    canvas.height = this.h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < this.w * this.h; i++) {
      const v = this.refine[i];
      if (v === 1) {
        img.data[i * 4] = 111;
        img.data[i * 4 + 1] = 182;
        img.data[i * 4 + 2] = 214; // --accent-2 slate/cyan
        img.data[i * 4 + 3] = 120;
      } else if (v === -1) {
        img.data[i * 4] = 207;
        img.data[i * 4 + 1] = 138;
        img.data[i * 4 + 2] = 95; // --accent terracotta
        img.data[i * 4 + 3] = 120;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Paint the focus map (white = in focus) for the live Depth preview. */
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

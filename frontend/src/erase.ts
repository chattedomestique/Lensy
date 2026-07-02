// Object-removal selection. Holds a binary selection mask at the working-image resolution and
// paints a translucent overlay of what's currently selected. Taps add a SAM2 object mask (fetched
// by main.ts); drags brush freehand. Undo snapshots the mask before each op. exportPng() hands the
// final mask (white = erase) to the backend /erase endpoint.

const SELECT_RGB = [207, 138, 95]; // --accent terracotta, for the selection tint

export class EraseSelection {
  private w = 0;
  private h = 0;
  private mask: Uint8Array<ArrayBuffer> = new Uint8Array(0); // 0 or 255 at (w,h)
  private undoStack: Uint8Array<ArrayBuffer>[] = [];
  private overlay: HTMLCanvasElement | null = null;

  /** Size the selection to the working image and bind the overlay canvas used to show it. */
  init(w: number, h: number, overlay: HTMLCanvasElement): void {
    this.w = w;
    this.h = h;
    this.mask = new Uint8Array(w * h);
    this.undoStack = [];
    this.overlay = overlay;
    overlay.width = w;
    overlay.height = h;
    this.redraw();
  }

  get ready(): boolean {
    return this.w > 0;
  }
  isEmpty(): boolean {
    for (let i = 0; i < this.mask.length; i++) if (this.mask[i]) return false;
    return true;
  }
  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  private snapshot(): void {
    this.undoStack.push(Uint8Array.from(this.mask));
    if (this.undoStack.length > 20) this.undoStack.shift();
  }
  undo(): void {
    const prev = this.undoStack.pop();
    if (prev) {
      this.mask = prev;
      this.redraw();
    }
  }
  clear(): void {
    if (this.isEmpty()) return;
    this.snapshot();
    this.mask.fill(0);
    this.redraw();
  }

  /** Union in a SAM2 mask image (grayscale PNG, white = object) at working resolution. */
  addMaskImage(img: HTMLImageElement): void {
    this.snapshot();
    const c = document.createElement("canvas");
    c.width = this.w;
    c.height = this.h;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0, this.w, this.h);
    const d = ctx.getImageData(0, 0, this.w, this.h).data;
    for (let i = 0; i < this.w * this.h; i++) if (d[i * 4] > 127) this.mask[i] = 255;
    this.redraw();
  }

  /** Brush: paint a filled disc (nx,ny normalized; radiusFrac of the long edge). */
  paint(nx: number, ny: number, radiusFrac: number, startStroke: boolean): void {
    if (startStroke) this.snapshot();
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
        if (dx * dx + dy * dy <= r2) this.mask[y * this.w + x] = 255;
      }
    }
    this.redraw();
  }

  /** Paint the translucent selection overlay. */
  private redraw(): void {
    if (!this.overlay) return;
    const ctx = this.overlay.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    const [r, g, b] = SELECT_RGB;
    for (let i = 0; i < this.w * this.h; i++) {
      if (this.mask[i]) {
        img.data[i * 4] = r;
        img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = 120;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  /** The final mask as a PNG blob (white = erase) at working resolution. */
  async exportPng(): Promise<Blob> {
    const c = document.createElement("canvas");
    c.width = this.w;
    c.height = this.h;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(this.w, this.h);
    for (let i = 0; i < this.w * this.h; i++) {
      const v = this.mask[i] ? 255 : 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return await new Promise<Blob>((resolve) => c.toBlob((bb) => resolve(bb!), "image/png"));
  }
}

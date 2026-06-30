// Lens control panel — owns the control *state* and reflects it to the DOM. Kept separate
// from render/network glue (main.ts) so the param model stays testable and tidy.

import type { RenderParams } from "./api";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const BLADE_LABEL: Record<number, string> = { 0: "circular", 5: "5-blade", 6: "6-blade", 8: "8-blade" };

export class Controls {
  private k = $("k") as HTMLInputElement;
  private focus = $("focus") as HTMLInputElement;
  private highlight = $("highlight") as HTMLInputElement;
  private bladesGroup = $("blades");
  private blades = 0;

  constructor(private onChange: () => void) {
    this.k.addEventListener("input", () => this.reflect());
    this.focus.addEventListener("input", () => this.reflect());
    this.highlight.addEventListener("input", () => this.reflect());

    this.bladesGroup.querySelectorAll<HTMLButtonElement>("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.blades = Number(btn.dataset.blades ?? "0");
        this.bladesGroup
          .querySelectorAll<HTMLButtonElement>("button")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        this.reflect();
      });
    });
    this.reflect();
  }

  /** Set the focal plane from a tap on the preview. `t` is 0..1 (disparity-ish). */
  setFocus(t: number): void {
    this.focus.value = String(Math.round(Math.min(1, Math.max(0, t)) * 100));
    this.reflect();
  }

  params(): RenderParams {
    return {
      k: Number(this.k.value),
      disp_focus: Number(this.focus.value) / 100,
      blades: this.blades,
      highlight_boost: Number(this.highlight.value) / 100,
      cat_eye: 0.35,
    };
  }

  reset(): void {
    this.k.value = "60";
    this.focus.value = "70";
    this.highlight.value = "60";
    this.blades = 0;
    this.bladesGroup
      .querySelectorAll<HTMLButtonElement>("button")
      .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.blades === "0")));
    this.reflect();
  }

  private reflect(): void {
    $("k-val").textContent = this.k.value;
    $("focus-val").textContent = (Number(this.focus.value) / 100).toFixed(2);
    $("highlight-val").textContent = (Number(this.highlight.value) / 100).toFixed(2);
    $("blades-val").textContent = BLADE_LABEL[this.blades] ?? `${this.blades}-blade`;
    this.onChange();
  }
}

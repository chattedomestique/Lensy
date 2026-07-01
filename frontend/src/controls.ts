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
  private dofGroup = $("dof");
  private subjectDof = true; // cinematic (subject blurs by depth) vs sharp cutout
  private autofocus = true; // on until the user drags the focal slider or taps the photo

  constructor(private onChange: () => void) {
    this.k.addEventListener("input", () => this.reflect());
    this.focus.addEventListener("input", () => {
      this.autofocus = false; // manual override
      this.reflect();
    });
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

    this.dofGroup.querySelectorAll<HTMLButtonElement>("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.subjectDof = btn.dataset.dof === "1";
        this.dofGroup
          .querySelectorAll<HTMLButtonElement>("button")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        this.reflect();
      });
    });
    this.reflect();
  }

  /** Set the focal plane from a tap on the preview. `t` is 0..1 (disparity-ish). */
  setFocus(t: number): void {
    this.autofocus = false; // tapping = manual focus
    this.focus.value = String(Math.round(Math.min(1, Math.max(0, t)) * 100));
    this.reflect();
  }

  params(): RenderParams {
    return {
      k: Number(this.k.value),
      disp_focus: Number(this.focus.value) / 100,
      autofocus: this.autofocus,
      subject_dof: this.subjectDof,
      blades: this.blades,
      highlight_boost: Number(this.highlight.value) / 100,
      cat_eye: 0.2,
    };
  }

  reset(): void {
    this.k.value = "60";
    this.focus.value = "70";
    this.highlight.value = "18";
    this.blades = 0;
    this.autofocus = true;
    this.subjectDof = true;
    this.bladesGroup
      .querySelectorAll<HTMLButtonElement>("button")
      .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.blades === "0")));
    this.dofGroup
      .querySelectorAll<HTMLButtonElement>("button")
      .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.dof === "1")));
    this.reflect();
  }

  private reflect(): void {
    $("k-val").textContent = this.k.value;
    $("focus-val").textContent = this.autofocus ? "auto" : (Number(this.focus.value) / 100).toFixed(2);
    $("highlight-val").textContent = (Number(this.highlight.value) / 100).toFixed(2);
    $("blades-val").textContent = BLADE_LABEL[this.blades] ?? `${this.blades}-blade`;
    $("dof-val").textContent = this.subjectDof ? "cinematic" : "sharp";
    this.onChange();
  }
}

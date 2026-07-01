// Lens control panel (Lens tab): blur strength, aperture shape, highlight bloom. Focus is set by
// the depth editor's Subject anchor now, and the subject is always composited sharp, so this is
// just the optics. Kept separate from render/network glue (main.ts).

import type { RenderParams } from "./api";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const BLADE_LABEL: Record<number, string> = { 0: "circular", 5: "5-blade", 6: "6-blade", 8: "8-blade" };

export class Controls {
  private k = $("k") as HTMLInputElement;
  private highlight = $("highlight") as HTMLInputElement;
  private swirl = $("swirl") as HTMLInputElement;
  private sweet = $("sweet") as HTMLInputElement;
  private sweetSize = $("sweet-size") as HTMLInputElement;
  private bladesGroup = $("blades");
  private blades = 0;

  constructor(private onChange: () => void) {
    this.k.addEventListener("input", () => this.reflect());
    this.highlight.addEventListener("input", () => this.reflect());
    [this.swirl, this.sweet, this.sweetSize].forEach((s) =>
      s.addEventListener("input", () => this.reflect()),
    );
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

  /** Lens params. Focus (disp_focus) is filled in by main.ts from the depth editor. */
  params(): RenderParams {
    return {
      k: Number(this.k.value),
      disp_focus: 0.5, // subject is centered at 0.5 in the edited depth; overridden by main.ts
      autofocus: false,
      subject_dof: false, // cinematic removed — subject always sharp
      blades: this.blades,
      highlight_boost: Number(this.highlight.value) / 100,
      cat_eye: 0.2,
      swirl: Number(this.swirl.value) / 100,
      sweet: Number(this.sweet.value) / 100,
      sweet_size: Number(this.sweetSize.value) / 100,
    };
  }

  reset(): void {
    this.k.value = "60";
    this.highlight.value = "18";
    this.swirl.value = "0";
    this.sweet.value = "0";
    this.sweetSize.value = "35";
    this.blades = 0;
    this.bladesGroup
      .querySelectorAll<HTMLButtonElement>("button")
      .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.blades === "0")));
    this.reflect();
  }

  private reflect(): void {
    $("k-val").textContent = this.k.value;
    $("highlight-val").textContent = (Number(this.highlight.value) / 100).toFixed(2);
    $("blades-val").textContent = BLADE_LABEL[this.blades] ?? `${this.blades}-blade`;
    $("swirl-val").textContent = (Number(this.swirl.value) / 100).toFixed(2);
    $("sweet-val").textContent = (Number(this.sweet.value) / 100).toFixed(2);
    $("sweet-size-val").textContent = (Number(this.sweetSize.value) / 100).toFixed(2);
    this.onChange();
  }
}

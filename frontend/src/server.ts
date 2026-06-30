// The "Connect to your render server" pill + panel. Lets you point the hosted PWA at your
// Mac's backend (Cloudflare Tunnel URL) and shows live connection status. State lives in
// config.ts (localStorage); network probing in api.ts.

import { checkHealth, pingHealth } from "./api";
import { getApiBase, setApiBase } from "./config";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export function setupServerPanel(): void {
  const pill = $("server-pill");
  const dot = $("server-dot");
  const label = $("server-label");
  const panel = $("server-panel");
  const input = $("server-url") as HTMLInputElement;
  const testBtn = $("server-test") as HTMLButtonElement;
  const status = $("server-status");

  const setDot = (state: "ok" | "bad" | "idle", text: string) => {
    dot.classList.remove("ok", "bad");
    if (state !== "idle") dot.classList.add(state);
    label.textContent = text;
  };

  // reflect the currently-saved server's health in the pill
  async function refresh(): Promise<void> {
    const base = getApiBase();
    setDot("idle", base ? "Checking…" : "Set server");
    const { ok, detail } = await checkHealth();
    setDot(ok ? "ok" : "bad", ok ? "Connected" : base ? "Offline" : "Set server");
    if (base) input.value = base;
    void detail;
  }

  const togglePanel = (open?: boolean) => {
    const show = open ?? panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !show);
    pill.setAttribute("aria-expanded", String(show));
    if (show) input.focus();
  };

  pill.addEventListener("click", () => togglePanel());

  async function connect(): Promise<void> {
    const url = input.value.trim();
    if (!url) {
      status.className = "caption bad";
      status.textContent = "Paste your tunnel URL first.";
      return;
    }
    testBtn.disabled = true;
    status.className = "caption";
    status.textContent = "Connecting…";
    const { ok, detail } = await pingHealth(url);
    testBtn.disabled = false;
    if (ok) {
      setApiBase(url);
      status.className = "caption ok";
      status.textContent = `Connected — ${detail}`;
      setDot("ok", "Connected");
      setTimeout(() => togglePanel(false), 900);
    } else {
      status.className = "caption bad";
      status.textContent = `Couldn't reach it (${detail}). Is serve.sh running?`;
      setDot("bad", "Offline");
    }
  }

  testBtn.addEventListener("click", connect);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") connect();
  });

  void refresh();
}

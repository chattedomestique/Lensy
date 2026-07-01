// Network layer — talks to the FastAPI backend. The base URL is resolved by config.ts:
// same-origin in dev (Vite proxy), or your Mac's Cloudflare Tunnel URL when hosted.
// Rendering is a job: POST /render → job_id, then stream stage progress over SSE. No polling.

import { apiUrl } from "./config";

export interface RenderParams {
  k: number; // 0..100 blur strength
  disp_focus: number; // 0..1 focal plane (used only when autofocus is false)
  autofocus: boolean; // lock focus to the subject
  subject_dof: boolean; // cinematic (blur subject by depth) vs sharp cutout
  blades: number; // 0 = circular, else N-gon
  highlight_boost: number; // 0..2 bloom strength
  cat_eye: number; // 0..1
}

export interface ProgressEvent {
  stage: string;
  label: string;
  progress: number; // 0..1
}

export interface RenderHandle {
  /** Resolves to an object URL for the finished JPEG. */
  done: Promise<string>;
  /** Abort the in-flight stream. */
  cancel(): void;
}

export class ApiError extends Error {}

export async function checkHealth(): Promise<{ ok: boolean; detail: string }> {
  try {
    const r = await fetch(apiUrl("/healthz"));
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      return { ok: false, detail: body?.error?.message ?? `server ${r.status}` };
    }
    const body = await r.json();
    return { ok: true, detail: body?.models?.device ?? "ready" };
  } catch {
    return { ok: false, detail: "backend unreachable" };
  }
}

/** Probe an arbitrary backend base URL (used by the connect panel before saving it). */
export async function pingHealth(base: string): Promise<{ ok: boolean; detail: string }> {
  const url = base.trim().replace(/\/+$/, "") + "/healthz";
  try {
    const r = await fetch(url, { method: "GET" });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      return { ok: false, detail: body?.error?.message ?? `server ${r.status}` };
    }
    const body = await r.json();
    const m = body?.models;
    return { ok: true, detail: m ? `ready · ${m.device} · ${m.matte}` : "ready" };
  } catch {
    return { ok: false, detail: "unreachable" };
  }
}

/** Analyze a photo → matte + depth map (fast). Returns an id + the working size. */
export async function analyze(file: File): Promise<{ analyzeId: string; width: number; height: number }> {
  const form = new FormData();
  form.append("photo", file);
  const r = await fetch(apiUrl("/analyze"), { method: "POST", body: form });
  if (!r.ok) {
    const b = await r.json().catch(() => ({}));
    throw new ApiError(b?.error?.message ?? `could not analyze (${r.status})`);
  }
  const d = await r.json();
  return { analyzeId: d.analyze_id as string, width: d.width as number, height: d.height as number };
}

export const depthUrl = (analyzeId: string) => apiUrl(`/analyze/${analyzeId}/depth.png`);
export const matteUrl = (analyzeId: string) => apiUrl(`/analyze/${analyzeId}/matte.png`);

function appendParams(form: FormData, params: RenderParams): void {
  form.append("k", String(params.k));
  form.append("disp_focus", String(params.disp_focus));
  form.append("autofocus", String(params.autofocus));
  form.append("subject_dof", String(params.subject_dof));
  form.append("blades", String(params.blades));
  form.append("highlight_boost", String(params.highlight_boost));
  form.append("cat_eye", String(params.cat_eye));
}

/** Render from an analysis + an (edited) depth-map PNG blob. */
export function renderFromAnalyze(
  analyzeId: string,
  depthPng: Blob,
  params: RenderParams,
  onProgress: (p: ProgressEvent) => void,
): RenderHandle {
  const form = new FormData();
  form.append("analyze_id", analyzeId);
  form.append("depth", depthPng, "depth.png");
  appendParams(form, params);
  return runRenderJob(form, onProgress);
}

/** Start a render and stream progress. Returns a handle whose `done` resolves to an image URL. */
export function render(
  file: File,
  params: RenderParams,
  onProgress: (p: ProgressEvent) => void,
): RenderHandle {
  const form = new FormData();
  form.append("photo", file);
  appendParams(form, params);
  return runRenderJob(form, onProgress);
}

function runRenderJob(form: FormData, onProgress: (p: ProgressEvent) => void): RenderHandle {
  let source: EventSource | null = null;
  let cancelled = false;
  let serverError: string | null = null;

  const done = (async (): Promise<string> => {
    const start = await fetch(apiUrl("/render"), { method: "POST", body: form });
    if (!start.ok) {
      const body = await start.json().catch(() => ({}));
      throw new ApiError(body?.error?.message ?? `could not start render (${start.status})`);
    }
    const { job_id } = (await start.json()) as { job_id: string };
    const resultPath = `/render/${job_id}/result`;

    // The render can take ~90s. Live progress comes over SSE, but a long-held stream can be
    // dropped by the tunnel/proxy — so the SSE is best-effort for the progress bar only, and the
    // RESULT is obtained by polling the result endpoint, which is robust to a dropped stream.
    source = new EventSource(apiUrl(`/render/${job_id}/events`));
    source.addEventListener("progress", (e) => {
      if (cancelled) return;
      try {
        onProgress(JSON.parse((e as MessageEvent).data));
      } catch {
        /* ignore malformed frame */
      }
    });
    source.addEventListener("error", (e) => {
      // a *server* error event carries data; a transport drop does not (polling will recover it)
      const data = (e as MessageEvent).data;
      if (data) {
        try {
          serverError = JSON.parse(data).error ?? "render failed";
        } catch {
          /* transport drop — ignore, polling handles it */
        }
      }
    });

    // poll the result until it's ready (200), the server reports a real error, or we time out.
    // 409 = still rendering; 502/503/504 + network blips = transient tunnel hiccups → keep polling
    // (a gateway blip during a 90s render must not fail it). Only a real 5xx/4xx from the app fails.
    const deadline = Date.now() + 6 * 60 * 1000;
    const wait = () => new Promise((res) => setTimeout(res, 1500));
    try {
      while (!cancelled) {
        if (serverError) throw new ApiError(serverError);
        if (Date.now() > deadline) throw new ApiError("render timed out");
        let r: Response;
        try {
          r = await fetch(apiUrl(resultPath), { cache: "no-store" });
        } catch {
          await wait(); // network hiccup — retry
          continue;
        }
        if (r.status === 200) return URL.createObjectURL(await r.blob());
        if (r.status === 409 || r.status === 502 || r.status === 503 || r.status === 504) {
          await wait(); // still rendering, or transient gateway error — retry
          continue;
        }
        const body = await r.json().catch(() => ({}));
        throw new ApiError(body?.error?.message ?? `render failed (${r.status})`);
      }
      throw new ApiError("cancelled");
    } finally {
      source?.close();
    }
  })();

  return {
    done,
    cancel() {
      cancelled = true;
      source?.close();
    },
  };
}

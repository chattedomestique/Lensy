// Network layer — talks to the FastAPI backend (same-origin via the Vite proxy / CF Tunnel).
// Rendering is a job: POST /render → job_id, then stream stage progress over SSE. No polling.

export interface RenderParams {
  k: number; // 0..100 blur strength
  disp_focus: number; // 0..1 focal plane
  blades: number; // 0 = circular, else N-gon
  highlight_boost: number; // 0..2
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
    const r = await fetch("/healthz");
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

/** Start a render and stream progress. Returns a handle whose `done` resolves to an image URL. */
export function render(
  file: File,
  params: RenderParams,
  onProgress: (p: ProgressEvent) => void,
): RenderHandle {
  let source: EventSource | null = null;
  let cancelled = false;

  const done = (async (): Promise<string> => {
    const form = new FormData();
    form.append("photo", file);
    form.append("k", String(params.k));
    form.append("disp_focus", String(params.disp_focus));
    form.append("blades", String(params.blades));
    form.append("highlight_boost", String(params.highlight_boost));
    form.append("cat_eye", String(params.cat_eye));

    const start = await fetch("/render", { method: "POST", body: form });
    if (!start.ok) {
      const body = await start.json().catch(() => ({}));
      throw new ApiError(body?.error?.message ?? `could not start render (${start.status})`);
    }
    const { job_id } = (await start.json()) as { job_id: string };

    const resultUrl = await new Promise<string>((resolve, reject) => {
      source = new EventSource(`/render/${job_id}/events`);
      source.addEventListener("progress", (e) => {
        if (cancelled) return;
        try {
          onProgress(JSON.parse((e as MessageEvent).data));
        } catch {
          /* ignore malformed frame */
        }
      });
      source.addEventListener("done", (e) => {
        try {
          const data = JSON.parse((e as MessageEvent).data);
          resolve(data.result_url ?? `/render/${job_id}/result`);
        } catch {
          resolve(`/render/${job_id}/result`);
        }
      });
      source.addEventListener("error", (e) => {
        // distinguish a server "error" event (has data) from a transport drop
        const data = (e as MessageEvent).data;
        if (data) {
          try {
            reject(new ApiError(JSON.parse(data).error ?? "render failed"));
            return;
          } catch {
            /* fall through */
          }
        }
        if (source && source.readyState === EventSource.CLOSED) {
          reject(new ApiError("connection to the render server was lost"));
        }
      });
    });

    (source as EventSource | null)?.close();
    if (cancelled) throw new ApiError("cancelled");

    // fetch the finished image as a blob → object URL the <img> can show
    const img = await fetch(resultUrl);
    if (!img.ok) {
      const body = await img.json().catch(() => ({}));
      throw new ApiError(body?.error?.message ?? "could not fetch the result");
    }
    return URL.createObjectURL(await img.blob());
  })();

  return {
    done,
    cancel() {
      cancelled = true;
      source?.close();
    },
  };
}

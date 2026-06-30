// Where the render backend lives. In local dev the Vite proxy serves /render + /healthz
// same-origin, so the base is "". When the PWA is hosted (GitHub Pages) the backend is your
// Mac reached over a Cloudflare Tunnel — a different origin — so the base is that tunnel URL.
//
// Resolution order: a URL you set in the app (saved in localStorage) → a build-time default
// (VITE_LENSY_API) → same-origin "". The in-app setting wins so an ephemeral quick-tunnel URL
// can change without a rebuild.

const LS_KEY = "lensy.apiBase";

const clean = (u: string): string => u.trim().replace(/\/+$/, "");

export function getApiBase(): string {
  const stored = localStorage.getItem(LS_KEY);
  if (stored) return clean(stored);
  const built = import.meta.env.VITE_LENSY_API as string | undefined;
  return built ? clean(built) : "";
}

export function setApiBase(url: string): void {
  const c = clean(url);
  if (c) localStorage.setItem(LS_KEY, c);
  else localStorage.removeItem(LS_KEY);
}

/** Build a full URL for a backend path, honoring the configured base. */
export function apiUrl(path: string): string {
  return getApiBase() + path;
}

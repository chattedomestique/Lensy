import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

// Backend origin for dev. The Vite dev server proxies /render + /healthz to FastAPI so the
// PWA talks to a same-origin path during local development. Port matches scripts (LENSY_PORT,
// default 8842 — chosen to avoid the rest of the sunhouse.media stack).
const BACKEND = process.env.LENSY_BACKEND ?? `http://localhost:${process.env.LENSY_PORT ?? "8842"}`;

// Base public path. GitHub Pages serves a project site under /<repo>/, so the Pages build
// sets LENSY_BASE=/Lensy/. Local dev and root-domain hosts use "/".
const BASE = process.env.LENSY_BASE ?? "/";

// Build stamp shown in the top bar so you can tell which build is live. Version from package.json
// + the short git SHA (falls back to the date if git isn't available at build time).
const PKG = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf-8"));
let GIT_SHA = "";
try {
  GIT_SHA = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] })
    .toString()
    .trim();
} catch {
  /* not a git checkout / git absent — fall back below */
}
const BUILD_ID = GIT_SHA || new Date().toISOString().slice(0, 10);

export default defineConfig({
  base: BASE,
  define: {
    __APP_VERSION__: JSON.stringify(PKG.version),
    __BUILD_ID__: JSON.stringify(BUILD_ID),
  },
  server: {
    port: 5173,
    proxy: {
      "/render": { target: BACKEND, changeOrigin: true },
      "/analyze": { target: BACKEND, changeOrigin: true },
      "/segment": { target: BACKEND, changeOrigin: true },
      "/subject": { target: BACKEND, changeOrigin: true },
      "/erase": { target: BACKEND, changeOrigin: true },
      "/undo": { target: BACKEND, changeOrigin: true },
      "/healthz": { target: BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2022",
  },
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icons/favicon.svg"],
      manifest: {
        name: "Lensy",
        short_name: "Lensy",
        description: "A handcrafted Portrait Mode for still photos.",
        theme_color: "#16140f",
        background_color: "#0f0e0c",
        display: "standalone",
        scope: BASE,
        start_url: BASE,
        id: BASE,
        icons: [
          { src: "icons/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
          { src: "icons/maskable-192.png", sizes: "192x192", type: "image/png", purpose: "maskable" },
          { src: "icons/maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        // cache the app shell so Lensy opens instantly / offline (rendering still needs the server)
        globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
        navigateFallback: "/index.html",
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith("/render") || url.pathname.startsWith("/healthz"),
            handler: "NetworkOnly", // never cache renders
          },
        ],
      },
      devOptions: { enabled: true },
    }),
  ],
});

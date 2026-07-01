import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

// Backend origin for dev. The Vite dev server proxies /render + /healthz to FastAPI so the
// PWA talks to a same-origin path during local development. Port matches scripts (LENSY_PORT,
// default 8842 — chosen to avoid the rest of the sunhouse.media stack).
const BACKEND = process.env.LENSY_BACKEND ?? `http://localhost:${process.env.LENSY_PORT ?? "8842"}`;

// Base public path. GitHub Pages serves a project site under /<repo>/, so the Pages build
// sets LENSY_BASE=/Lensy/. Local dev and root-domain hosts use "/".
const BASE = process.env.LENSY_BASE ?? "/";

export default defineConfig({
  base: BASE,
  server: {
    port: 5173,
    proxy: {
      "/render": { target: BACKEND, changeOrigin: true },
      "/analyze": { target: BACKEND, changeOrigin: true },
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
        theme_color: "#ec734a",
        background_color: "#f6f4ef",
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

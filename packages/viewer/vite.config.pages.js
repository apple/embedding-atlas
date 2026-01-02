import { svelte } from "@sveltejs/vite-plugin-svelte";
import icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

import { tsJsonSchemaPlugin } from "./scripts/vite-plugin-ts-json-schema.js";

// GitHub Pages configuration
// Set GITHUB_PAGES_BASE to your repo name, e.g., "/embedding-atlas/"
const base = process.env.GITHUB_PAGES_BASE || "/embedding-atlas/";

// https://vitejs.dev/config/
export default defineConfig({
  base: base,
  plugins: [svelte(), wasm(), icons({ compiler: "svelte" }), tsJsonSchemaPlugin()],
  worker: {
    format: "es",
    plugins: () => [wasm()],
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 4096,
    outDir: "dist-pages",
  },
});

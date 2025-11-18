import { svelte } from "@sveltejs/vite-plugin-svelte";
import icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

import { tsJsonSchemaPlugin } from "./scripts/vite-plugin-ts-json-schema.js";

// https://vitejs.dev/config/
export default defineConfig({
  base: "",
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
  },
});

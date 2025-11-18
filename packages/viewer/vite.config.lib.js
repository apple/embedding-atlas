import { svelte } from "@sveltejs/vite-plugin-svelte";
import icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import wasm from "vite-plugin-wasm";

import { tsJsonSchemaPlugin } from "./scripts/vite-plugin-ts-json-schema.js";

function fixAbsoluteImport() {
  // Fix a bug where vite outputs absolute paths for workers.
  return {
    name: "fix-absolute-import",
    renderChunk(code) {
      // new URL(/* @vite-ignore */ "/assets/worker_main-DWGFbKCZ.js"
      // ->
      // new URL("./assets/worker_main-DWGFbKCZ.js"
      return code.replace(/new URL\((\/\*.*?\*\/ *)?"\//g, `new URL("./`);
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    svelte(),
    wasm(),
    icons({ compiler: "svelte" }),
    dts({ rollupTypes: true }),
    tsJsonSchemaPlugin(),
    fixAbsoluteImport(),
  ],
  worker: {
    format: "es",
    plugins: () => [wasm()],
    rollupOptions: {
      output: {
        entryFileNames: "[name].js",
        inlineDynamicImports: true,
      },
    },
  },
  build: {
    outDir: "distlib",
    target: "esnext",
    lib: {
      entry: {
        index: "./src/index.ts",
      },
      fileName: (_, entryName) => `${entryName}.js`,
      formats: ["es"],
    },
    rollupOptions: {
      external: [
        "@uwdata/mosaic-core",
        "@uwdata/mosaic-spec",
        "@uwdata/mosaic-sql",
        "@uwdata/vgplot",
        "@embedding-atlas/utils",
      ],
      output: {
        chunkFileNames: "chunk-[hash].js",
      },
    },
    copyPublicDir: false,
    chunkSizeWarningLimit: 4096,
  },
  optimizeDeps: {
    exclude: ["svelte"],
  },
});

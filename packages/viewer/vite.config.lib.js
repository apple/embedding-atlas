import { fixAbsoluteImport } from "@embedding-atlas/utils/vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import wasm from "vite-plugin-wasm";

import { tsJsonSchemaPlugin } from "./scripts/vite-plugin-ts-json-schema.js";

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
    rolldownOptions: {
      output: {
        entryFileNames: "[name].js",
        codeSplitting: false,
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
    rolldownOptions: {
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

import { fixAbsoluteImport } from "@embedding-atlas/utils/vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import wasm from "vite-plugin-wasm";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte(), wasm(), dts({ rollupTypes: true }), fixAbsoluteImport()],
  worker: {
    plugins: () => [wasm()],
    format: "es",
    rolldownOptions: {
      output: {
        entryFileNames: "[name].js",
      },
    },
  },
  build: {
    target: "esnext",
    lib: {
      entry: {
        index: "./src/lib/index.ts",
      },
      fileName: (_, entryName) => `${entryName}.js`,
      formats: ["es"],
    },
    rolldownOptions: {
      external: ["@uwdata/mosaic-core", "@uwdata/mosaic-sql", "@embedding-atlas/utils"],
    },
    copyPublicDir: false,
  },
  resolve: {
    alias: [{ find: /(.*\/worker_functions)\.js$/, replacement: "$1.ts" }],
  },
  optimizeDeps: {
    exclude: ["svelte"],
  },
});

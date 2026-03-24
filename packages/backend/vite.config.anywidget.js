import { forceInlineWorker } from "@embedding-atlas/utils/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [forceInlineWorker()],
  worker: {
    format: "es",
    plugins: () => [wasm()],
  },
  build: {
    outDir: "./embedding_atlas/widget_static/anywidget",
    target: "esnext",
    lib: {
      entry: {
        index: "./src/anywidget/index.ts",
      },
      fileName: (_, entryName) => `${entryName}.js`,
      formats: ["es"],
    },
    copyPublicDir: false,
    chunkSizeWarningLimit: 4096,
    rolldownOptions: {
      output: {
        codeSplitting: false,
      },
    },
  },
});

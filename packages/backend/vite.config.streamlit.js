import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

function forceInlineWorker() {
  return {
    name: "force-inline-worker",
    transform(code, id) {
      // Hack: if we prefix the url with `"" +`, rollup will inline the worker.
      return code.replace(/new Worker\(new URL\(/g, `new Worker("" + new URL(`);
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [forceInlineWorker()],
  worker: {
    format: "es",
    plugins: () => [wasm()],
  },
  base: "./",
  root: "./src/streamlit",
  build: {
    outDir: "../../embedding_atlas/widget_static/streamlit",
    target: "esnext",
    copyPublicDir: false,
    chunkSizeWarningLimit: 4096,
    emptyOutDir: true,
  },
});

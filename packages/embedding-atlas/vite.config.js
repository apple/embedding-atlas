import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

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
    dts({
      rollupTypes: true,
      bundledPackages: [
        "@embedding-atlas/component",
        "@embedding-atlas/table",
        "@embedding-atlas/viewer",
        "@embedding-atlas/umap-wasm",
        "@embedding-atlas/density-clustering",
      ],
    }),
    fixAbsoluteImport(),
  ],
  worker: {
    format: "es",
    rollupOptions: {
      output: {
        entryFileNames: "[name].js",
      },
    },
  },
  build: {
    outDir: "dist",
    target: "esnext",
    lib: {
      entry: {
        index: "./src/index.ts",
        component: "./src/component.ts",
        viewer: "./src/viewer.ts",
        umap: "./src/umap.ts",
        react: "./src/react.ts",
      },
      fileName: (_, entryName) => `${entryName}.js`,
      formats: ["es"],
    },
    rollupOptions: {
      external: ["react", "@uwdata/mosaic-core", "@uwdata/mosaic-spec", "@uwdata/mosaic-sql", "@uwdata/vgplot"],
    },
    copyPublicDir: false,
    chunkSizeWarningLimit: 4096,
  },
});

import { fixAbsoluteImport } from "@embedding-atlas/utils/vite";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    dts({
      rollupTypes: true,
      bundledPackages: [
        "@embedding-atlas/component",
        "@embedding-atlas/viewer",
        "@embedding-atlas/umap-wasm",
        "@embedding-atlas/density-clustering",
      ],
    }),
    fixAbsoluteImport(),
  ],
  worker: {
    format: "es",
    rolldownOptions: {
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
    rolldownOptions: {
      external: ["react", "@uwdata/mosaic-core", "@uwdata/mosaic-spec", "@uwdata/mosaic-sql", "@uwdata/vgplot"],
      output: {
        chunkFileNames: "chunk-[hash].js",
      },
    },
    copyPublicDir: false,
    chunkSizeWarningLimit: 4096,
  },
});

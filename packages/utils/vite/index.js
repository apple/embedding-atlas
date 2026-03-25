// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export function fixAbsoluteImport() {
  // Fix a bug where vite outputs absolute paths for workers.
  return {
    name: "fix-absolute-import",
    renderChunk(code) {
      // new URL(/* @vite-ignore */ "/assets/worker_main-DWGFbKCZ.js"
      // ->
      // new URL("./assets/worker_main-DWGFbKCZ.js"
      return code
        .replace(/new\s+URL\(\s*(\/\*[\s\S]*?\*\/\s*)?"\//g, `new URL("./`)
        .replace(/"" \+ import\.meta\.url/g, "import.meta.url");
    },
  };
}

export function forceInlineWorker() {
  return {
    name: "force-inline-worker",
    transform(code, id) {
      // Hack: if we prefix the url with `"" +`, rollup will inline the worker.
      return code.replace(/new Worker\(new URL\(/g, `new Worker("" + new URL(`);
    },
  };
}

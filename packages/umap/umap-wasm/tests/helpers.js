import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { initSync } from "../pkg/umap_wasm.js";

/**
 * Initialize WASM synchronously for Node.js (vitest).
 * Call this in a beforeAll() hook.
 */
export function initWasm() {
  const wasmPath = fileURLToPath(new URL("../pkg/umap_wasm_bg.wasm", import.meta.url));
  initSync({ module: readFileSync(wasmPath) });
}

/**
 * Deterministic pseudo-random data generator (mulberry32).
 * @param {number} count number of data points
 * @param {number} dim dimension per point
 * @param {number} [seed=42]
 * @returns {Float32Array}
 */
export function makeRandomData(count, dim, seed = 42) {
  const data = new Float32Array(count * dim);
  let s = seed;
  for (let i = 0; i < data.length; i++) {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    data[i] = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  return data;
}

/** Helpers of benchmark.html: the sweep dataset index, its downloads, and a seeded random source. */

export const DEFAULT_INDEX = "../../benchmark/results/sweep/datasets/index.json";
export const LARGE_DATASET_BYTES = 100e6;
export const SWEEP_MISSING =
  "Sweep datasets not found: serve packages/umap and run `uv run --frozen python -m benchmark datasets` once";

/**
 * Checks the downloaded buffers against the index entry. `data` is float32 little-endian,
 * row-major (n x dim). `labels` is one uint8 per row or null.
 */
export function parseDataset(data, labels, n, dim) {
  if (data.byteLength !== n * dim * 4) {
    throw new Error(`data.bin has ${data.byteLength} bytes, but ${n} x ${dim} float32 values need ${n * dim * 4}.`);
  }
  if (labels != null && labels.byteLength !== n) {
    throw new Error(`labels.bin has ${labels.byteLength} labels, but data.bin has ${n} rows.`);
  }
  return {
    data: new Float32Array(data),
    count: n,
    dim,
    labels: labels == null ? null : new Uint8Array(labels),
  };
}

/** The first n rows (and labels) of a dataset, as copies. */
export function firstRows(dataset, n) {
  const rows = Math.max(0, Math.min(Math.floor(n), dataset.count));
  return {
    data: dataset.data.slice(0, rows * dataset.dim),
    count: rows,
    dim: dataset.dim,
    labels: dataset.labels?.slice(0, rows) ?? null,
  };
}

/** A URL from a query parameter, or the default; both relative to `base`. */
export function queryUrl(search, parameter, fallback, base) {
  return new URL(new URLSearchParams(search).get(parameter) ?? fallback, base).href;
}
export const indexUrl = (search, base) => queryUrl(search, "datasets", DEFAULT_INDEX, base);

/** The datasets of datasets/index.json, written by `python -m benchmark datasets`. Paths are relative to the index. */
export function indexDatasets(index, base) {
  return (index.datasets ?? []).map((entry) => ({
    name: entry.name,
    n: entry.n,
    dim: entry.dim,
    metric: entry.metric === "cosine" ? "cosine" : "euclidean",
    urls: {
      data: new URL(`${entry.folder}/data.bin`, base).href,
      labels: entry.labels ? new URL(`${entry.folder}/labels.bin`, base).href : null,
    },
  }));
}

export const datasetBytes = (n, dim) => n * dim * 4;

/** The float32 size of a dataset, for example "220 MB". */
export function describeSize(n, dim) {
  const bytes = datasetBytes(n, dim);
  return bytes >= 1e6 ? `${Math.round(bytes / 1e6)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} kB`;
}

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Downloads a URL into an ArrayBuffer. `onProgress(loadedBytes, totalBytes)`; the total is 0 when unknown. */
export async function fetchBytes(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const chunks = [];
  let loaded = 0;
  const reader = response.body.getReader();
  for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) {
    chunks.push(chunk.value);
    loaded += chunk.value.byteLength;
    onProgress(loaded, total);
  }
  return await new Blob(chunks).arrayBuffer();
}

export async function fetchJson(url) {
  try {
    const response = await fetch(url);
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

/** The sweep datasets from index.json. Null when it cannot be fetched. */
export async function fetchSweep() {
  const index = indexUrl(location.search, location.href);
  const fromIndex = await fetchJson(index);
  return fromIndex == null ? null : indexDatasets(fromIndex, index);
}

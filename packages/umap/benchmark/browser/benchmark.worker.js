/**
 * The benchmark of benchmark.html. It runs in a module worker, so the page stays responsive.
 *
 * Messages from the page to the worker:
 *   { type: "run", jobs, options }
 *     jobs: [{ id, count, dim, metric, urls: { data } | null }]. `urls` is null for synthetic datasets. URLs are absolute.
 *     options: { variants: [{ label, gpu, options }], warmUp: boolean, rowCounts: number[] }
 *
 * Messages from the worker to the page:
 *   { type: "status", text }     what the worker does now (setup stage and fraction, dataset loading)
 *   { type: "system", info }     { userAgent, cores, gpu }, sent once when the worker starts
 *   { type: "result", row }      one finished measurement
 *   { type: "done" }             all jobs are finished
 *   { type: "error", message }   the run failed as a whole
 */
import { createUMAP } from "../../umap-wasm/index.js";
import { fetchBytes, firstRows, parseDataset, seededRandom } from "./shared.js";

const CLUSTERS = 20;
const STATUS_INTERVAL_MS = 100;

/** The sizes to run for a dataset: all rows when no counts are listed, else the listed counts that fit. */
export function rowsToRun(count, rowCounts) {
  if (rowCounts.length === 0) return [count];
  return [...new Set(rowCounts.filter((n) => n <= count))].sort((a, b) => a - b);
}

/** Gaussian clusters: cluster centers have spread 3, points have spread 1. */
export function makeClusters(count, dim, seed = 1) {
  const random = seededRandom(seed);
  const gaussian = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  const centers = Float32Array.from({ length: CLUSTERS * dim }, () => 3 * gaussian());
  const data = new Float32Array(count * dim);
  for (let i = 0; i < count; i++) {
    const center = Math.floor(random() * CLUSTERS) * dim;
    for (let d = 0; d < dim; d++) data[i * dim + d] = centers[center + d] + gaussian();
  }
  return data;
}

/** Maps the stage names that the progress callback reports to a pipeline step. */
function stepOfStage(stage) {
  if (/neighbors|forest|descent|sorting/i.test(stage)) return "knn";
  if (/graph/i.test(stage)) return "graph";
  if (/initializ/i.test(stage)) return "init";
  if (/optimiz/i.test(stage)) return "layout";
  return "other";
}

/** Adds up the time of each pipeline step from the stage names that a progress callback reports. */
export function createStageTimer(now) {
  const totals = { knn: 0, graph: 0, init: 0, layout: 0, other: 0 };
  let current = null;
  let since = now();
  const close = () => {
    if (current != null) totals[stepOfStage(current)] += now() - since;
    current = null;
  };
  return {
    totals,
    close,
    enter(stage) {
      if (stage === current) return;
      close();
      current = stage;
      since = now();
    },
  };
}

const isGpuMessage = (text) => /gpu/i.test(text);

/** Records the console messages that mention the GPU. Returns a function that restores the console. */
function captureGpuMessages(console, messages) {
  const { error, warn } = console;
  const wrap =
    (original) =>
    (...args) => {
      const text = args.map(String).join(" ");
      if (isGpuMessage(text)) messages.push(text);
      original.apply(console, args);
    };
  console.error = wrap(error);
  console.warn = wrap(warn);
  return () => Object.assign(console, { error, warn });
}

export async function collectSystem(scope = globalThis) {
  const { navigator } = scope;
  let gpu = "WebGPU is not available";
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      const info = adapter.info ?? {};
      gpu =
        [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ") ||
        "unknown adapter";
    }
  } catch (error) {
    gpu = `WebGPU error: ${error}`;
  }
  return { userAgent: navigator.userAgent, cores: navigator.hardwareConcurrency, gpu };
}

/**
 * Runs one variant: createUMAP (kNN, graph, init) then run() (layout).
 * The progress callback receives (fraction, stage name). We time the stages.
 * When the GPU fails, umap-wasm logs "GPU layout failed ... continuing on the CPU"
 * with console.error. The result is then correct but slow, so we capture it.
 */
async function measure(dataset, data, variant, warmedUp, post, deps) {
  const { createUMAP, performance, console } = deps;
  const result = {
    dataset: dataset.id,
    metric: dataset.metric,
    rows: dataset.count,
    variant: variant.label,
    warmedUp,
    gpuMessages: [],
    error: null,
  };
  const timer = createStageTimer(() => performance.now());
  const restoreConsole = captureGpuMessages(console, result.gpuMessages);
  const start = performance.now();
  let lastStatus = -Infinity;
  const progress = (fraction, stage) => {
    timer.enter(stage);
    const now = performance.now();
    if (now - lastStatus < STATUS_INTERVAL_MS) return;
    lastStatus = now;
    post({ type: "status", text: `${stage} (${Math.round(fraction * 100)}%)` });
  };
  try {
    const umap = await createUMAP(dataset.count, dataset.dim, 2, data, {
      ...variant.options,
      metric: dataset.metric,
      gpu: variant.gpu,
      progress,
    });
    await umap.run();
    timer.close();
    result.totalMs = performance.now() - start;
    result.totals = { ...timer.totals };
    umap.destroy();
  } catch (error) {
    result.error = error instanceof Error ? error.message : String(error);
  } finally {
    restoreConsole();
  }
  return result;
}

/** The dataset with its data: downloaded (urls) or generated (synthetic). */
async function loadDataset(dataset, post) {
  if (dataset.urls == null) {
    return { data: makeClusters(dataset.count, dataset.dim), count: dataset.count, dim: dataset.dim, labels: null };
  }
  const show = (loaded, total) =>
    post({
      type: "status",
      text: `Downloading ${dataset.id}: ${(loaded / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ""} MB`,
    });
  const data = await fetchBytes(dataset.urls.data, show);
  return parseDataset(data, null, dataset.count, dataset.dim);
}

function errorRow(dataset, message) {
  return { dataset: dataset.id, metric: dataset.metric, rows: "-", variant: "-", gpuMessages: [], error: message };
}

/** Runs all jobs and reports through `post`. `deps` replaces createUMAP, performance and console in tests. */
export async function runJobs(jobs, options, post, deps = {}) {
  const resolved = { createUMAP, performance: globalThis.performance, console: globalThis.console, ...deps };
  const { variants, warmUp, rowCounts } = options;
  try {
    for (const dataset of jobs) {
      post({ type: "status", text: `Preparing ${dataset.id}...` });
      let full;
      try {
        full = await loadDataset(dataset, post);
      } catch (error) {
        post({ type: "result", row: errorRow(dataset, error.message) });
        continue;
      }
      const sizes = rowsToRun(dataset.count, rowCounts);
      if (sizes.length === 0) {
        post({ type: "result", row: errorRow(dataset, `No listed row count fits (${dataset.count} rows).`) });
      }
      for (const rows of sizes) {
        const part = firstRows(full, rows);
        const sized = { ...dataset, count: part.count };
        for (const variant of variants) {
          const label = `${dataset.id} (${rows} rows), ${variant.label}`;
          if (warmUp) {
            post({ type: "status", text: `Warm-up (discarded): ${label}` });
            await measure(sized, part.data, variant, true, post, resolved);
          }
          post({ type: "status", text: `Measuring: ${label}` });
          post({ type: "result", row: await measure(sized, part.data, variant, warmUp, post, resolved) });
        }
      }
    }
    post({ type: "done" });
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
}

if (typeof WorkerGlobalScope !== "undefined" && globalThis instanceof WorkerGlobalScope) {
  const post = (message) => globalThis.postMessage(message);
  globalThis.onmessage = (event) => {
    if (event.data.type === "run") runJobs(event.data.jobs, event.data.options, post);
  };
  post({ type: "system", info: await collectSystem() });
}

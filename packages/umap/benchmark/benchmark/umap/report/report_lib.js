// Pure functions of the standalone sweep report. No DOM access, so Node can test all of them.

var NO_AXIS = "none";
var DEFAULT_VALUE = "__default__";
var HASH_PREFIX = "rp.";
var IMPLEMENTATION_OPTIONS = [
  { id: "python", label: "Python" },
  { id: "rust-cpu", label: "Rust CPU" },
  { id: "rust-gpu", label: "Rust GPU" },
];
var OPTIMIZER_OPTIONS = [
  { id: "sgd", label: "SGD (0.5)" },
  { id: "adam", label: "Adam (0.6)" },
  { id: "momentum", label: "Momentum (0.6)" },
];
var METRIC_INFO = {
  time_ms: { label: "Time", higherIsBetter: false, digits: 0, tintScale: 3000 },
  trustworthiness: { label: "Trustworthiness", higherIsBetter: true, digits: 4, tintScale: 0.03 },
  knn_recall: { label: "kNN recall", higherIsBetter: true, digits: 4, tintScale: 0.03 },
  procrustes_vs_python: { label: "Procrustes vs Python", higherIsBetter: false, digits: 4, tintScale: 0.2 },
};
var METRIC_KEYS = Object.keys(METRIC_INFO);
var CATEGORY_COLORS = [
  "#1f77b4",
  "#aec7e8",
  "#ff7f0e",
  "#ffbb78",
  "#2ca02c",
  "#98df8a",
  "#d62728",
  "#ff9896",
  "#9467bd",
  "#c5b0d5",
  "#8c564b",
  "#c49c94",
  "#e377c2",
  "#f7b6d2",
  "#7f7f7f",
  "#c7c7c7",
  "#bcbd22",
  "#dbdb8d",
  "#17becf",
  "#9edae5",
];
var CATEGORY_COLORS_10 = CATEGORY_COLORS.filter((_, i) => i % 2 === 0);
var IDENTITY_VIEW = { zoom: 1, panX: 0, panY: 0 };
var MIN_ZOOM = 0.5;
var MAX_ZOOM = 200;
var THUMB_PADDING = 8;
var HOVER_RADIUS = 12;

// ---- Encoding ----

/** Decodes a base64 string into bytes. */
function base64ToBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** Maps uint16 codes back to [min, max]. */
function dequantize(codes, min, max) {
  return Float32Array.from(codes, (code) => min + (code / 65535) * (max - min));
}

/**
 * Decodes the quantized 2D points of a run. The run holds `q` (base64, all x codes then all y codes)
 * and `r` ([xMin, xMax, yMin, yMax]).
 */
function decodeRunPoints(run) {
  if (run == null || run.q == null) return null;
  const codes = new Uint16Array(base64ToBytes(run.q).buffer);
  const n = codes.length >> 1;
  const r = run.r;
  return {
    x: dequantize(codes.subarray(0, n), r[0], r[1]),
    y: dequantize(codes.subarray(n), r[2], r[3]),
  };
}

/** Decodes the subset of a dataset: original row indices and class labels. */
function decodeDataset(dataset) {
  return {
    indices: new Uint32Array(base64ToBytes(dataset.indices).buffer),
    labels: base64ToBytes(dataset.labels),
  };
}

// ---- Geometry ----

/** Centers the points and scales them to unit root-mean-square radius. Non-finite points go to the origin. */
function normalize(points) {
  const n = points.x.length;
  const finite = Array.from({ length: n }, (_, i) => Number.isFinite(points.x[i]) && Number.isFinite(points.y[i]));
  const count = Math.max(finite.filter(Boolean).length, 1);
  const mean = (values) => values.reduce((sum, v, i) => (finite[i] ? sum + v : sum), 0) / count;
  const cx = mean(points.x);
  const cy = mean(points.y);
  let sumSquares = 0;
  for (let i = 0; i < n; i++) {
    if (finite[i]) sumSquares += Math.pow(points.x[i] - cx, 2) + Math.pow(points.y[i] - cy, 2);
  }
  const scale = sumSquares > 0 ? 1 / Math.sqrt(sumSquares / count) : 1;
  const center = (values, c) => Float32Array.from(values, (v, i) => (finite[i] ? (v - c) * scale : 0));
  return { x: center(points.x, cx), y: center(points.y, cy) };
}

/**
 * Aligns source to target with the best rotation, optional reflection, and uniform scale.
 * Both inputs must be normalized. Returns the aligned source and the residual disparity in [0, 1].
 */
function procrustesAlign(source, target) {
  const n = Math.min(source.x.length, target.x.length);
  let targetNorm = 0;
  for (let i = 0; i < n; i++) targetNorm += target.x[i] * target.x[i] + target.y[i] * target.y[i];

  const fit = (flip) => {
    let dot = 0;
    let cross = 0;
    let sourceNorm = 0;
    for (let k = 0; k < n; k++) {
      const sx = flip * source.x[k];
      const sy = source.y[k];
      dot += sx * target.x[k] + sy * target.y[k];
      cross += sx * target.y[k] - sy * target.x[k];
      sourceNorm += sx * sx + sy * sy;
    }
    const gain = sourceNorm > 0 ? (dot * dot + cross * cross) / sourceNorm : 0;
    return { flip, dot, cross, sourceNorm, gain };
  };

  const plain = fit(1);
  const mirrored = fit(-1);
  const best = mirrored.gain > plain.gain ? mirrored : plain;
  const scale = best.sourceNorm > 0 ? Math.hypot(best.dot, best.cross) / best.sourceNorm : 1;
  const angle = Math.atan2(best.cross, best.dot);
  const cos = Math.cos(angle) * scale;
  const sin = Math.sin(angle) * scale;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const sx = best.flip * source.x[i];
    const sy = source.y[i];
    x[i] = cos * sx - sin * sy;
    y[i] = sin * sx + cos * sy;
  }
  const disparity = targetNorm > 0 ? Math.max(0, 1 - best.gain / targetNorm) : 0;
  return { aligned: { x, y }, disparity };
}

/**
 * Normalized points of a run, aligned to the reference run when asked and possible.
 * `getPoints(run)` returns the decoded raw points of a run, or null.
 */
function alignedPoints(run, reference, align, getPoints) {
  const raw = getPoints(run);
  if (raw == null) return null;
  const points = normalize(raw);
  if (!align || reference == null || reference === run || run.has_nan || reference.has_nan) return points;
  const referenceRaw = getPoints(reference);
  if (referenceRaw == null) return points;
  return procrustesAlign(points, normalize(referenceRaw)).aligned;
}

// ---- Run lookup ----

/** Stable key of a full parameter set. */
function paramsKey(params) {
  return JSON.stringify(
    Object.keys(params)
      .sort()
      .map((k) => [k, params[k]]),
  );
}

/** Indexes runs by dataset, variant, and parameters. */
function createRunIndex(runs, mainSeed) {
  const byKey = new Map();
  const keyOf = (dataset, variant, params) => dataset + "|" + variant + "|" + paramsKey(params);
  runs.forEach((run) => {
    const key = keyOf(run.dataset, run.variant, run.params);
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push(run);
  });
  const findAll = (dataset, variant, params) => byKey.get(keyOf(dataset, variant, params)) || [];
  const find = (dataset, variant, params, seed) => {
    const wanted = seed == null ? mainSeed : seed;
    return findAll(dataset, variant, params).find((run) => run.seed === wanted);
  };
  return { find, findAll };
}

/** The parameters of the unmodified default configuration of a dataset. */
function baseParams(manifest, dataset) {
  const run = manifest.runs.find((r) => r.dataset === dataset && r.axis == null && r.seed === manifest.main_seed);
  return run ? run.params : undefined;
}

/** The full parameters of a sweep point: the dataset defaults with one axis changed. */
function pointParams(manifest, dataset, axis, value) {
  const base = baseParams(manifest, dataset);
  if (base == null) return undefined;
  if (value === DEFAULT_VALUE) return base;
  const info = manifest.axes.find((a) => a.param === axis);
  return { ...base, ...(info ? info.coupled : {}), [axis]: value };
}

/** Axes that have at least one run for the dataset. */
function axesOf(manifest, dataset) {
  const used = new Set(manifest.runs.filter((r) => r.dataset === dataset && r.axis != null).map((r) => r.axis));
  return manifest.axes.filter((a) => used.has(a.param));
}

/** Seeds that have a run for the dataset, ascending. */
function seedsOf(manifest, dataset) {
  const seeds = new Set(manifest.runs.filter((r) => r.dataset === dataset).map((r) => r.seed));
  return Array.from(seeds).sort((a, b) => a - b);
}

// ---- Variants ----

/** Orders variants so that each Python variant is followed by the Rust variants it is the partner of. */
function groupedVariants(variants) {
  return variants
    .filter((v) => v.implementation === "python")
    .flatMap((p) => [p, ...variants.filter((v) => v.python_partner === p.id)]);
}

function implementationOf(variant) {
  return variant.implementation === "python" ? "python" : variant.gpu ? "rust-gpu" : "rust-cpu";
}

function optimizerOf(variant) {
  return variant.pipeline.split(" ")[1];
}

function variantLabel(variant) {
  const isRust = variant.implementation === "rust";
  return (isRust ? "Rust " : "Python ") + variant.pipeline + (isRust ? (variant.gpu ? " GPU" : " CPU") : "");
}

// ---- Metrics and warnings ----

function formatMs(value) {
  return value >= 1000 ? (value / 1000).toFixed(2) + " s" : value.toFixed(0) + " ms";
}

function formatMetric(key, value) {
  if (value == null) return "-";
  return key === "time_ms" ? formatMs(value) : value.toFixed(METRIC_INFO[key].digits);
}

/** Mean and sample standard deviation over seeds. Null when fewer than two seeds have a value. */
function seedStats(runs, key) {
  const values = runs.map((r) => r[key]).filter((v) => v != null);
  if (values.length < 2) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (values.length - 1);
  return { count: values.length, mean, std: Math.sqrt(variance) };
}

/** Warning badges of a run: NaN, error, flags from the sweep, and Rust trust below its Python partner. */
function warningsFor(run, variant, reference, threshold) {
  const warnings = [];
  if (run.has_nan) warnings.push({ kind: "nan", text: "NaN/inf" });
  if (run.status === "error") warnings.push({ kind: "error", text: run.reason || "error" });
  run.flags.forEach((flag) => {
    const isNan = flag.indexOf("NaN/inf") === 0;
    if (!isNan || !run.has_nan) {
      warnings.push({ kind: isNan || /error$/.test(flag) ? "nan" : "worse", text: flag });
    }
  });
  const mine = run.trustworthiness;
  const theirs = reference ? reference.trustworthiness : null;
  if (
    variant.implementation === "rust" &&
    mine != null &&
    theirs != null &&
    theirs - mine > threshold &&
    run.flags.indexOf("worse than Python") < 0
  ) {
    warnings.push({ kind: "worse", text: "trust " + (mine - theirs).toFixed(3) + " vs Python" });
  }
  return warnings;
}

/** Rust minus Python for a metric. Procrustes is already a difference, so it is used as is. */
function metricDelta(run, reference, key) {
  if (run == null) return null;
  if (key === "procrustes_vs_python") return run[key] ?? null;
  const a = run[key];
  const b = reference ? reference[key] : null;
  return a == null || b == null ? null : a - b;
}

/** Cell background for a Rust run: green when better than its Python partner, red when worse. */
function tintColor(key, delta, isRust) {
  if (key === "none" || !isRust || delta == null) return "transparent";
  const info = METRIC_INFO[key];
  const round = (alpha) => Math.round(alpha * 1000) / 1000;
  if (key === "procrustes_vs_python") return "rgba(220,80,70," + round(Math.min(0.5, delta / info.tintScale)) + ")";
  const good = info.higherIsBetter ? delta > 0 : delta < 0;
  const alpha = 0.08 + Math.min(1, Math.abs(delta) / info.tintScale) * 0.5;
  return "rgba(" + (good ? "60,170,90" : "220,80,70") + "," + round(alpha) + ")";
}

/** Tooltip text of a cell. */
function cellTooltip(run, seedRuns, warnings) {
  if (run == null) return "";
  let text = run.variant + ", seed " + run.seed + "\ntime " + formatMetric("time_ms", run.time_ms);
  if (run.knn_ms != null) {
    text += " (kNN " + formatMs(run.knn_ms) + ", layout " + formatMs(run.layout_ms || 0) + ")";
  }
  text += "\nProcrustes vs Python " + formatMetric("procrustes_vs_python", run.procrustes_vs_python);
  ["trustworthiness", "knn_recall", "time_ms"].forEach((key) => {
    const stats = seedStats(seedRuns, key);
    if (stats == null) return;
    const mean = formatMetric(key, stats.mean);
    const spread = formatMetric(key, stats.std);
    text += "\n" + key + " over " + stats.count + " seeds: " + mean + " ± " + spread;
  });
  if (warnings.length) text += "\n" + warnings.map((w) => w.text).join(", ");
  return text;
}

/** Why a cell has no plot. */
function emptyReason(run) {
  if (run == null) return "no run";
  if (run.reason) return run.reason;
  return run.q == null ? "no embedding" : "";
}

// ---- Matrix ----

function axisValueLabel(axis, value) {
  if (value !== DEFAULT_VALUE) return String(value);
  return "default (" + String(axis && axis.default != null ? axis.default : "auto") + ")";
}

/**
 * Builds the matrix for a state: column headers, and rows of cells. With no axis, columns are
 * implementations and rows are optimizers. With an axis, columns are axis values and rows are variants.
 */
function buildMatrix(manifest, index, state) {
  const variants = groupedVariants(manifest.variants.filter((v) => !v.timing_only));
  const variantById = new Map(manifest.variants.map((v) => [v.id, v]));
  const noAxis = state.axis === NO_AXIS;
  const axis = manifest.axes.find((a) => a.param === state.axis);

  const cellFor = (variant, value) => {
    const params = pointParams(manifest, state.dataset, noAxis ? "" : state.axis, value);
    const partner = variant.python_partner;
    const run = params && index.find(state.dataset, variant.id, params, state.seed);
    const reference = params && partner ? index.find(state.dataset, partner, params, state.seed) : undefined;
    const seedRuns = params ? index.findAll(state.dataset, variant.id, params) : [];
    const warnings = run ? warningsFor(run, variantById.get(variant.id), reference, state.threshold) : [];
    return {
      key: variant.id + "|" + String(value),
      variantId: variant.id,
      run,
      reference,
      seedRuns,
      warnings,
      tint: tintColor(state.tint, metricDelta(run, reference, state.tint), variant.implementation === "rust"),
      tooltip: cellTooltip(run, seedRuns, warnings),
    };
  };

  if (noAxis) {
    const columnOptions = IMPLEMENTATION_OPTIONS.filter((o) => state.impl.indexOf(o.id) >= 0);
    const rowOptions = OPTIMIZER_OPTIONS.filter((o) => state.opt.indexOf(o.id) >= 0);
    return {
      columns: columnOptions.map((o) => ({ key: o.id, label: o.label })),
      rows: rowOptions.map((row) => ({
        key: row.id,
        label: row.label,
        baseline: false,
        cells: columnOptions.map((column) => {
          const variant = variants.find((v) => implementationOf(v) === column.id && optimizerOf(v) === row.id);
          return variant ? cellFor(variant, DEFAULT_VALUE) : null;
        }),
      })),
    };
  }

  const values = [DEFAULT_VALUE, ...(axis ? axis.values : [])];
  return {
    columns: values.map((value) => ({ key: String(value), label: state.axis + "\n" + axisValueLabel(axis, value) })),
    rows: variants
      .filter((v) => state.impl.indexOf(implementationOf(v)) >= 0 && state.opt.indexOf(optimizerOf(v)) >= 0)
      .map((variant) => ({
        key: variant.id,
        label: variantLabel(variant),
        baseline: variant.implementation === "python",
        cells: values.map((value) => cellFor(variant, value)),
      })),
  };
}

// ---- Thumbnail view ----

/** Zooms by factor and keeps the point at (cx, cy) fixed. Coordinates are in extent units from the center. */
function zoomAround(view, factor, cx, cy) {
  const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.zoom * factor));
  const k = zoom / view.zoom;
  return { zoom, panX: cx - (cx - view.panX) * k, panY: cy - (cy - view.panY) * k };
}

/** Pans by a pixel offset. */
function panBy(view, dxPixels, dyPixels, size) {
  const unit = size - THUMB_PADDING;
  return { zoom: view.zoom, panX: view.panX + dxPixels / unit, panY: view.panY - dyPixels / unit };
}

/** Zoom factor of a wheel event. Pinch gestures (ctrlKey) are faster. */
function wheelFactor(deltaY, ctrlKey) {
  return Math.exp(-deltaY * (ctrlKey ? 0.01 : 0.002));
}

/** A cursor position in pixels inside the thumbnail as extent units from its center. */
function cursorToExtent(mx, my, size) {
  const unit = size - THUMB_PADDING;
  return { cx: (mx - size / 2) / unit, cy: -(my - size / 2) / unit };
}

/** Point positions relative to the center of the cloud, in units of its extent (trimmed to 0.5%..99.5%). */
function buildLayout(points) {
  const trimmedRange = (values) => {
    const sorted = Float32Array.from(values).sort();
    const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
    return [at(0.005), at(0.995)];
  };
  if (points.x.length === 0) return { bx: new Float32Array(0), by: new Float32Array(0) };
  const [x0, x1] = trimmedRange(points.x);
  const [y0, y1] = trimmedRange(points.y);
  const extent = Math.max(x1 - x0, y1 - y0) || 1;
  const centered = (values, mid) => Float32Array.from(values, (v) => (v - mid) / extent);
  return { bx: centered(points.x, (x0 + x1) / 2), by: centered(points.y, (y0 + y1) / 2) };
}

/** Pixel position of point i. */
function projectPoint(layout, i, view, size) {
  const unit = size - THUMB_PADDING;
  return {
    x: size / 2 + (layout.bx[i] * view.zoom + view.panX) * unit,
    y: size / 2 - (layout.by[i] * view.zoom + view.panY) * unit,
  };
}

/** Index of the point closest to the cursor within the hover radius, or null. */
function nearestPoint(layout, view, size, mx, my) {
  let best = HOVER_RADIUS * HOVER_RADIUS;
  let found = null;
  for (let i = 0; i < layout.bx.length; i++) {
    const p = projectPoint(layout, i, view, size);
    const d = (p.x - mx) * (p.x - mx) + (p.y - my) * (p.y - my);
    if (d < best) {
      best = d;
      found = i;
    }
  }
  return found;
}

/** Hex colors for class indices. Up to 10 classes use the 10 color palette, otherwise the 20 color palette wraps. */
function categoryColors(count) {
  const n = Math.max(1, count);
  if (n <= CATEGORY_COLORS_10.length) return CATEGORY_COLORS_10.slice(0, n);
  return Array.from({ length: n }, (_, i) => CATEGORY_COLORS[i % CATEGORY_COLORS.length]);
}

/** Point indices per class. */
function groupByClass(labels, n) {
  let count = 1;
  for (let i = 0; i < n; i++) count = Math.max(count, labels[i] + 1);
  const lists = Array.from({ length: count }, () => []);
  for (let j = 0; j < n; j++) lists[labels[j]].push(j);
  return lists;
}

// ---- URL hash state ----

var STATE_DEFAULTS = {
  dataset: "",
  axis: "",
  seed: -1,
  size: 240,
  align: true,
  link: true,
  tint: "trustworthiness",
  threshold: 0.01,
  impl: IMPLEMENTATION_OPTIONS.map((o) => o.id),
  opt: OPTIMIZER_OPTIONS.map((o) => o.id),
};

function defaultState() {
  return { ...STATE_DEFAULTS, impl: [...STATE_DEFAULTS.impl], opt: [...STATE_DEFAULTS.opt] };
}

/** Merges the state into an existing hash (with or without "#"). Other keys are kept. */
function encodeHash(state, existingHash) {
  const params = new URLSearchParams((existingHash || "").replace(/^#/, ""));
  Object.keys(STATE_DEFAULTS).forEach((name) => {
    const value = state[name];
    const key = HASH_PREFIX + name;
    let text;
    if (Array.isArray(value)) text = value.join(",");
    else if (typeof value === "boolean") text = value ? "1" : "0";
    else if (value == null || (name === "seed" && value < 0)) text = "";
    else text = String(value);
    if (text === "") params.delete(key);
    else params.set(key, text);
  });
  return params.toString().replace(/%2C/g, ",").replace(/%3A/g, ":");
}

/** Reads the state from a hash. Unknown or invalid values fall back to the defaults. */
function decodeHash(hash) {
  const params = new URLSearchParams((hash || "").replace(/^#/, ""));
  const state = defaultState();
  const text = (name) => {
    const value = params.get(HASH_PREFIX + name);
    return value === "" ? null : value;
  };
  const number = (name) => {
    const value = text(name);
    return value != null && Number.isFinite(Number(value)) ? Number(value) : null;
  };
  const list = (name, options) => {
    const value = params.get(HASH_PREFIX + name);
    return value == null ? null : value.split(",").filter((item) => options.some((o) => o.id === item));
  };
  const dataset = text("dataset");
  const axis = text("axis");
  const seed = number("seed");
  const size = number("size");
  const threshold = number("threshold");
  const tint = text("tint");
  const impl = list("impl", IMPLEMENTATION_OPTIONS);
  const opt = list("opt", OPTIMIZER_OPTIONS);
  if (dataset != null) state.dataset = dataset;
  if (axis != null) state.axis = axis;
  if (seed != null) state.seed = seed;
  if (size != null) state.size = Math.min(400, Math.max(120, size));
  if (threshold != null) state.threshold = Math.max(0, threshold);
  ["align", "link"].forEach((name) => {
    const value = text(name);
    if (value === "1") state[name] = true;
    if (value === "0") state[name] = false;
  });
  if (tint === "none" || METRIC_KEYS.indexOf(tint) >= 0) state.tint = tint;
  if (impl != null) state.impl = impl;
  if (opt != null) state.opt = opt;
  return state;
}

/** Replaces dataset, axis, and seed with valid choices for the manifest. */
function reconcileState(manifest, state) {
  const next = { ...state };
  const datasetValid = manifest.datasets.some((d) => d.name === next.dataset);
  if (!datasetValid) next.dataset = manifest.datasets.length ? manifest.datasets[0].name : "";
  const axes = axesOf(manifest, next.dataset);
  const axisValid = next.axis === NO_AXIS || axes.some((a) => a.param === next.axis);
  if (!axisValid) next.axis = axes.length ? axes[0].param : NO_AXIS;
  if (seedsOf(manifest, next.dataset).indexOf(next.seed) < 0) next.seed = manifest.main_seed;
  return next;
}

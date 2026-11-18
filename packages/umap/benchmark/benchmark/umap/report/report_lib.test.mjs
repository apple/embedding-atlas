import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const source = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "report_lib.js"), "utf8");
const lib = vm.createContext({ atob, URLSearchParams });
vm.runInContext(source, lib);

// Values created inside the vm context have other prototypes. structuredClone copies them into this realm.
const sameValue = (actual, expected) => assert.deepEqual(structuredClone(actual), structuredClone(expected));

const near = (a, b, tolerance = 1e-5) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} is not within ${tolerance} of ${b}`);

function toBase64(typed) {
  return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString("base64");
}

function randomPoints(n, seed = 1) {
  let state = seed;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = (next() - 0.5) * 6 + (i % 3);
    y[i] = (next() - 0.5) * 2;
  }
  return { x, y };
}

function transform(points, { angle, flip, scale, dx = 0, dy = 0 }) {
  const cos = Math.cos(angle) * scale;
  const sin = Math.sin(angle) * scale;
  const x = new Float32Array(points.x.length);
  const y = new Float32Array(points.x.length);
  for (let i = 0; i < x.length; i++) {
    const sx = flip * points.x[i];
    x[i] = cos * sx - sin * points.y[i] + dx;
    y[i] = sin * sx + cos * points.y[i] + dy;
  }
  return { x, y };
}

test("decodeRunPoints reads the x block then the y block and applies the ranges", () => {
  const q = Uint16Array.of(0, 65535, 32768, 0, 65535, 0);
  const decoded = lib.decodeRunPoints({ q: toBase64(q), r: [-4, 5, 10, 20] });
  assert.equal(decoded.x.length, 3);
  near(decoded.x[0], -4);
  near(decoded.x[1], 5);
  near(decoded.x[2], 0.5, 1e-3);
  near(decoded.y[0], 10);
  near(decoded.y[1], 20);
  near(decoded.y[2], 10);
  assert.equal(lib.decodeRunPoints({ q: null }), null);
  assert.equal(lib.decodeRunPoints(undefined), null);
});

test("decodeDataset reads indices and labels", () => {
  const dataset = { indices: toBase64(Uint32Array.of(5, 70000)), labels: toBase64(Uint8Array.of(3, 9)) };
  const decoded = lib.decodeDataset(dataset);
  sameValue(Array.from(decoded.indices), [5, 70000]);
  sameValue(Array.from(decoded.labels), [3, 9]);
});

test("procrustesAlign recovers a known rotation, reflection, and scale", () => {
  const target = lib.normalize(randomPoints(300, 7));
  const moved = lib.normalize(transform(target, { angle: -2.1, flip: -1, scale: 3.5, dx: 4, dy: -2 }));
  const { aligned, disparity } = lib.procrustesAlign(moved, target);
  for (let i = 0; i < 300; i++) {
    near(aligned.x[i], target.x[i], 1e-3);
    near(aligned.y[i], target.y[i], 1e-3);
  }
  near(disparity, 0, 1e-5);
});

const base = { n_neighbors: 15, min_dist: 0.1, spread: 1, init: "auto", metric: "euclidean" };
function run(overrides) {
  return {
    dataset: "d1",
    variant: "py",
    seed: 42,
    axis: null,
    params: base,
    status: "ok",
    reason: null,
    time_ms: 1000,
    knn_ms: null,
    layout_ms: null,
    trustworthiness: 0.95,
    knn_recall: 0.3,
    procrustes_vs_python: null,
    has_nan: false,
    flags: [],
    q: "AAAAAA==",
    r: [0, 1, 0, 1],
    ...overrides,
  };
}
const nn = (value) => ({ ...base, n_neighbors: value });
const variants = [
  {
    id: "rust_sgd_gpu",
    implementation: "rust",
    pipeline: "0.5 sgd",
    gpu: true,
    python_partner: "py_sgd",
    timing_only: false,
  },
  {
    id: "py_adam",
    implementation: "python",
    pipeline: "0.6 adam",
    gpu: false,
    python_partner: null,
    timing_only: false,
  },
  {
    id: "rust_adam_cpu",
    implementation: "rust",
    pipeline: "0.6 adam",
    gpu: false,
    python_partner: "py_adam",
    timing_only: false,
  },
  { id: "py_sgd", implementation: "python", pipeline: "0.5 sgd", gpu: false, python_partner: null, timing_only: false },
  {
    id: "rust_sgd_cpu",
    implementation: "rust",
    pipeline: "0.5 sgd",
    gpu: false,
    python_partner: "py_sgd",
    timing_only: false,
  },
  { id: "py_par", implementation: "python", pipeline: "0.5 sgd", gpu: false, python_partner: null, timing_only: true },
];
const manifest = {
  main_seed: 42,
  datasets: [{ name: "d1" }, { name: "d2" }],
  axes: [
    { param: "n_neighbors", values: [5, 15], default: 15, coupled: {} },
    { param: "spread", values: [1, 2], default: 1, coupled: { min_dist: 0.5 } },
    { param: "init", values: ["pca"], default: "auto", coupled: {} },
  ],
  variants,
  runs: [
    run({}),
    run({ variant: "rust_sgd_cpu", trustworthiness: 0.9, time_ms: 100, knn_ms: 40, layout_ms: 60 }),
    run({ variant: "py_sgd", axis: "n_neighbors", params: nn(5) }),
    run({ variant: "py_sgd", axis: "n_neighbors", params: nn(5), seed: 0, trustworthiness: 0.93 }),
    run({ variant: "py_sgd", axis: "n_neighbors", params: nn(5), seed: 1, trustworthiness: 0.91 }),
    run({ variant: "rust_sgd_cpu", axis: "n_neighbors", params: nn(5), trustworthiness: 0.93 }),
    run({ variant: "py_sgd", axis: "spread", params: { ...base, spread: 2, min_dist: 0.5 } }),
    run({ dataset: "d2", variant: "py_sgd" }),
  ],
};
manifest.runs[0].variant = "py_sgd";

const state = (overrides) => ({ ...lib.defaultState(), dataset: "d1", axis: "n_neighbors", seed: 42, ...overrides });

test("buildMatrix with an axis: default column first, grouped rows, runs and references", () => {
  const index = lib.createRunIndex(manifest.runs, 42);
  const matrix = lib.buildMatrix(manifest, index, state({}));
  sameValue(
    matrix.columns.map((c) => c.label),
    ["n_neighbors\ndefault (15)", "n_neighbors\n5", "n_neighbors\n15"],
  );
  sameValue(
    matrix.rows.map((r) => r.key),
    ["py_adam", "rust_adam_cpu", "py_sgd", "rust_sgd_gpu", "rust_sgd_cpu"],
  );
  sameValue(
    matrix.rows.map((r) => r.baseline),
    [true, false, true, false, false],
  );
  const rustRow = matrix.rows.find((r) => r.key === "rust_sgd_cpu");
  assert.equal(rustRow.cells[0].run.time_ms, 100);
  assert.equal(rustRow.cells[0].reference.variant, "py_sgd");
  sameValue(rustRow.cells[0].warnings, [{ kind: "worse", text: "trust -0.050 vs Python" }]);
  assert.match(rustRow.cells[0].tint, /^rgba\(220,80,70/);
  assert.equal(rustRow.cells[1].seedRuns.length, 1);
  assert.equal(rustRow.cells[2].run, rustRow.cells[0].run);
  const pythonRow = matrix.rows.find((r) => r.key === "py_sgd");
  assert.equal(pythonRow.cells[1].seedRuns.length, 3);
  assert.equal(pythonRow.cells[0].tint, "transparent");
  assert.equal(matrix.rows.find((r) => r.key === "py_adam").cells[0].run, undefined);
});

test("encodeHash and decodeHash round trip with namespaced readable keys", () => {
  const original = {
    dataset: "mnist10k_cosine",
    axis: "n_neighbors",
    seed: 3,
    size: 180,
    align: false,
    link: false,
    tint: "knn_recall",
    threshold: 0.025,
    impl: ["python", "rust-gpu"],
    opt: ["adam"],
  };
  const hash = lib.encodeHash(original, "");
  assert.match(hash, /rp\.dataset=mnist10k_cosine/);
  assert.match(hash, /rp\.impl=python,rust-gpu/);
  assert.match(hash, /rp\.align=0/);
  sameValue(lib.decodeHash("#" + hash), original);
});

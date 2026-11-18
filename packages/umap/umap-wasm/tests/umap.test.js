import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { createUMAP, createUMAPFromKNN } from "../index.js";
import { initWasm, makeRandomData } from "./helpers.js";

beforeAll(() => initWasm());

describe("createUMAP", () => {
  const COUNT = 200;
  const INPUT_DIM = 10;
  const OUTPUT_DIM = 2;

  let umap;

  afterEach(() => {
    umap?.destroy();
    umap = null;
  });

  it("returns correct inputDim and outputDim", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.inputDim).toBe(INPUT_DIM);
    expect(umap.outputDim).toBe(OUTPUT_DIM);
  });

  it("has a valid embedding immediately at epoch 0 (eager setup)", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.epoch).toBe(0);
    const emb = umap.embedding;
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(COUNT * OUTPUT_DIM);
    for (let i = 0; i < emb.length; i++) {
      expect(Number.isFinite(emb[i])).toBe(true);
    }
  });

  it("produces an embedding after run()", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    await umap.run();
    const emb = umap.embedding;
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(COUNT * OUTPUT_DIM);
    for (let i = 0; i < emb.length; i++) {
      expect(Number.isNaN(emb[i])).toBe(false);
    }
  });

  it("run() advances to the horizon", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    const before = new Float32Array(umap.embedding);
    await umap.run();
    expect(umap.epoch).toBe(10);
    const after = umap.embedding;
    // The layout should have moved from its initial position.
    let moved = false;
    for (let i = 0; i < after.length; i++) {
      if (Math.abs(after[i] - before[i]) > 1e-6) {
        moved = true;
        break;
      }
    }
    expect(moved).toBe(true);
  });

  it("step() advances the epoch and moves the embedding", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 100,
      initializeMethod: "random",
    });
    const before = new Float32Array(umap.embedding);
    await umap.step(5);
    expect(umap.epoch).toBe(5);
    const after = umap.embedding;
    expect(after.length).toBe(COUNT * OUTPUT_DIM);
    let moved = false;
    for (let i = 0; i < after.length; i++) {
      expect(Number.isFinite(after[i])).toBe(true);
      if (Math.abs(after[i] - before[i]) > 1e-6) moved = true;
    }
    expect(moved).toBe(true);
  });

  it("run() continues from the current epoch", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    await umap.step(5);
    expect(umap.epoch).toBe(5);
    await umap.run();
    expect(umap.epoch).toBe(10);
  });

  it("setParameters does not throw and keeps the layout finite", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 50,
      initializeMethod: "random",
    });
    umap.setParameters({
      learningRate: 0.5,
      repulsionStrength: 2.0,
      negativeSampleRate: 7,
      minDist: 0.2,
      spread: 1.5,
    });
    await umap.step(10);
    const emb = umap.embedding;
    for (let i = 0; i < emb.length; i++) {
      expect(Number.isFinite(emb[i])).toBe(true);
    }
  });

  it("setOptimizer('momentum') does not throw and stays finite", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 50,
      initializeMethod: "random",
    });
    umap.setOptimizer("momentum");
    await umap.step(20);
    const emb = umap.embedding;
    for (let i = 0; i < emb.length; i++) {
      expect(Number.isFinite(emb[i])).toBe(true);
    }
  });

  it("reset() restarts from epoch 0 (random and spectral)", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 30,
      initializeMethod: "random",
    });
    await umap.step(10);
    expect(umap.epoch).toBe(10);

    umap.reset("random");
    expect(umap.epoch).toBe(0);
    await umap.step(5);
    expect(umap.epoch).toBe(5);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);

    umap.reset("spectral");
    expect(umap.epoch).toBe(0);
    await umap.step(5);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  it("accepts the momentum optimizer option", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 30,
      optimizer: "momentum",
    });
    await umap.run();
    const emb = umap.embedding;
    expect(emb.length).toBe(COUNT * OUTPUT_DIM);
    for (let i = 0; i < emb.length; i++) {
      expect(Number.isFinite(emb[i])).toBe(true);
    }
  });

  it("runs the 0.6 pipeline (compatibilityLayout false) and is deterministic", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    const run = async () => {
      const instance = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
        seed: 42,
        nEpochs: 30,
        compatibilityLayout: false,
      });
      await instance.run();
      const embedding = new Float32Array(instance.embedding);
      instance.destroy();
      return embedding;
    };
    const first = await run();
    const second = await run();
    expect(first.length).toBe(COUNT * OUTPUT_DIM);
    for (const v of first) expect(Number.isFinite(v)).toBe(true);
    expect(second).toEqual(first);
  });

  it("accepts adam, recursive init and initScaleFactor explicitly", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 20,
      compatibilityLayout: false,
      optimizer: "adam",
      initializeMethod: "recursive",
      initScaleFactor: 0.5,
    });
    await umap.run();
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  it("setOptimizer('adam') keeps the embedding finite", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 30,
      initializeMethod: "random",
    });
    umap.setOptimizer("adam");
    await umap.step(10);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  for (const optimizer of ["momentum", "adam"]) {
    it(`rejects optimizer=${optimizer} with compatibilityLayout=true`, async () => {
      const data = makeRandomData(COUNT, INPUT_DIM);
      await expect(
        createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, { seed: 42, compatibilityLayout: true, optimizer }),
      ).rejects.toThrow(/compatibility_layout/);
    });

    it(`selects the 0.6 pipeline for optimizer=${optimizer} when compatibilityLayout is unset`, async () => {
      const options = { seed: 42, nEpochs: 20, optimizer };
      const data = makeRandomData(COUNT, INPUT_DIM);
      const unset = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, options);
      const modern = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, { ...options, compatibilityLayout: false });
      await unset.run();
      await modern.run();
      expect(Array.from(unset.embedding)).toEqual(Array.from(modern.embedding));
      unset.destroy();
      modern.destroy();
    });
  }

  for (const compatibilityLayout of [true, false]) {
    for (const optimizer of ["sgd", "momentum", "adam"]) {
      if (compatibilityLayout && optimizer !== "sgd") continue;
      it(`runs optimizer=${optimizer} with compatibilityLayout=${compatibilityLayout}`, async () => {
        const data = makeRandomData(COUNT, INPUT_DIM);
        umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
          seed: 42,
          nEpochs: 20,
          compatibilityLayout,
          optimizer,
        });
        await umap.run();
        expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
        for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
      });
    }
  }

  for (const initializeMethod of ["auto", "spectral", "tswspectral", "random", "recursive", "pca"]) {
    it(`builds with initializeMethod=${initializeMethod}`, async () => {
      const data = makeRandomData(COUNT, INPUT_DIM);
      umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, { seed: 42, nEpochs: 10, initializeMethod });
      expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
      for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
    });
  }

  for (const method of ["auto", "spectral", "tswspectral", "random", "recursive", "pca"]) {
    it(`reset('${method}') restarts from epoch 0 with a finite embedding`, async () => {
      const data = makeRandomData(COUNT, INPUT_DIM);
      umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
        seed: 42,
        nEpochs: 20,
        compatibilityLayout: false,
      });
      await umap.step(5);
      umap.reset(method, data);
      expect(umap.epoch).toBe(0);
      await umap.step(5);
      for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
    });
  }

  it("reset needs data for pca, validates its size, and works without it otherwise", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, { seed: 42, nEpochs: 10 });
    expect(() => umap.reset("pca")).toThrow(/input data/);
    expect(() => umap.reset("pca", data.subarray(1))).toThrow(/does not match/);
    for (const method of ["spectral", "tswspectral", "random", "recursive"]) {
      umap.reset(method);
      for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("rejects unknown option values", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    const create = (options) => createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, { nEpochs: 5, ...options });
    await expect(create({ initializeMethod: "bogus" })).rejects.toThrow(/init method/);
    await expect(create({ optimizer: "bogus" })).rejects.toThrow(/optimizer/);
    await expect(create({ initRescale: "bogus" })).rejects.toThrow(/initRescale/);
    await expect(create({ recursiveInit: { hubSelection: "bogus" } })).rejects.toThrow(/hubSelection/);
  });

  it("accepts the layout tuning options", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    for (const initRescale of ["isotropic", "perAxis", "graphEdge"]) {
      const instance = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
        seed: 42,
        nEpochs: 15,
        compatibilityLayout: false,
        initRescale,
        a: 1.5,
        b: 0.9,
        disconnectionDistance: 100,
        negativeSelectionRange: 50,
        negativeSampleScale: 1,
        excludeGraphNeighbors: true,
        negativeSampleScaleAdaptationSamples: 0,
      });
      await instance.run();
      for (const v of instance.embedding) expect(Number.isFinite(v)).toBe(true);
      instance.destroy();
    }
  });

  it("accepts recursiveInit options", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 15,
      initializeMethod: "recursive",
      recursiveInit: {
        baseInitThreshold: 32,
        nEmbeddingEpochs: 10,
        coarseningRatio: 2,
        hubSelection: "random",
        coarseLayoutWeights: "fuzzyUnion",
        curveSchedule: { initialA: 0.5, initialB: 0.5, recursiveA: 0.5, recursiveB: 0.5 },
        goodInitializationPolicy: "always",
        anchorMethod: "fullPca",
        negativeSelectionRangeMode: "coarseN",
      },
    });
    await umap.run();
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  it("accepts hubSelection diverse", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 15,
      initializeMethod: "recursive",
      recursiveInit: { baseInitThreshold: 32, coarseningRatio: 2, hubSelection: "diverse" },
    });
    await umap.run();
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  it("destroy() clears the embedding", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
    umap.destroy();
    expect(umap.embedding.length).toBe(0);
    umap = null;
  });

  it("destroy() is safe to call multiple times", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    await umap.run();
    umap.destroy();
    umap.destroy();
    umap = null;
  });

  it("destroy() during an in-flight step does not throw and frees cleanly", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      initializeMethod: "random",
    });
    // Start a step but don't await it — the wasm value is now borrowed, so a
    // naive free() would throw "attempted to take ownership ... while borrowed".
    const stepPromise = umap.step(20);
    expect(() => umap.destroy()).not.toThrow();
    // The free is queued after the in-flight step and runs once it settles.
    await expect(stepPromise).resolves.toBeUndefined();
    // The instance is now inert: getters are empty and further ops are no-ops.
    expect(umap.embedding.length).toBe(0);
    expect(umap.epoch).toBe(0);
    await expect(umap.step(1)).resolves.toBeUndefined();
    umap = null;
  });

  it("overlapping step() calls are serialized, not dropped (both run)", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 100,
      initializeMethod: "random",
    });
    // Fire two steps without awaiting the first: the second queues behind it and
    // both run, so the epoch advances by the full 5 + 5 (not dropped, not just 5).
    const p1 = umap.step(5);
    const p2 = umap.step(5);
    await Promise.all([p1, p2]);
    expect(umap.epoch).toBe(10);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
    umap.destroy();
    umap = null;
  });

  it("getters serve the last settled snapshot during an in-flight step", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      initializeMethod: "random",
    });
    expect(umap.epoch).toBe(0);

    // Start a step but don't await: while it is in flight the wasm value is
    // borrowed, so the getters must serve snapshots rather than read wasm (which
    // would throw "recursive use of an object detected" in GPU mode).
    const stepPromise = umap.step(10);
    expect(umap.epoch).toBe(0); // last settled value, not yet advanced
    expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
    // First-ever read of the (immutable) kNN graph mid-step must return the real
    // graph, not an empty array — it is snapshotted at construction.
    expect(umap.knnIndices.length).toBeGreaterThan(0);
    expect(umap.knnDistances.length).toBe(umap.knnIndices.length);

    await stepPromise;
    expect(umap.epoch).toBe(10); // refreshed once the step settled
    umap.destroy();
    umap = null;
  });

  it("seed produces deterministic results", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    const opts = { seed: 123, nEpochs: 10, initializeMethod: "random" };

    const u1 = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, opts);
    await u1.run();
    const emb1 = new Float32Array(u1.embedding);
    u1.destroy();

    const u2 = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, opts);
    await u2.run();
    const emb2 = new Float32Array(u2.embedding);
    u2.destroy();

    expect(emb1).toEqual(emb2);
  });

  it("works with cosine metric", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      metric: "cosine",
      initializeMethod: "random",
    });
    await umap.run();
    expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
  });

  it("works with 3D output", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, 3, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    await umap.run();
    expect(umap.embedding.length).toBe(COUNT * 3);
  });

  it("works with spectral initialization", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "spectral",
    });
    await umap.run();
    expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
  });

  it("accepts all options without error", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      metric: "euclidean",
      initializeMethod: "random",
      optimizer: "sgd",
      localConnectivity: 1.0,
      mixRatio: 1.0,
      spread: 1.0,
      minDist: 0.1,
      repulsionStrength: 1.0,
      nEpochs: 10,
      learningRate: 1.0,
      negativeSampleRate: 5,
      nNeighbors: 10,
      seed: 42,
    });
    await umap.run();
    expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
  });

  it("has valid knnIndices and knnDistances immediately (eager setup)", async () => {
    const N_NEIGHBORS = 10;
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      nNeighbors: N_NEIGHBORS,
      initializeMethod: "random",
    });
    expect(umap.knnIndices).toBeInstanceOf(Int32Array);
    expect(umap.knnIndices.length).toBe(COUNT * N_NEIGHBORS);
    expect(umap.knnDistances).toBeInstanceOf(Float32Array);
    expect(umap.knnDistances.length).toBe(COUNT * N_NEIGHBORS);

    // All indices should be valid
    for (let i = 0; i < umap.knnIndices.length; i++) {
      expect(umap.knnIndices[i]).toBeGreaterThanOrEqual(0);
      expect(umap.knnIndices[i]).toBeLessThan(COUNT);
    }
  });

  it("throws on unknown metric (at create, eager setup)", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    await expect(
      createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
        metric: "nonexistent",
        nEpochs: 10,
        initializeMethod: "random",
        seed: 42,
      }),
    ).rejects.toThrow(/unknown metric/i);
  });

  it("throws on data length mismatch (at create)", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    // Pass wrong count so data.length != count * inputDim
    await expect(
      createUMAP(COUNT + 1, INPUT_DIM, OUTPUT_DIM, data, {
        nEpochs: 10,
        initializeMethod: "random",
        seed: 42,
      }),
    ).rejects.toThrow(/length/i);
  });
});

describe("createUMAPFromKNN", () => {
  const COUNT = 200;
  const INPUT_DIM = 10;
  const OUTPUT_DIM = 2;
  const K = 10;

  let umap;

  afterEach(() => {
    umap?.destroy();
    umap = null;
  });

  // A valid canonical kNN graph (self in column 0) produced by a normal UMAP setup.
  async function makeKnn(seed = 42) {
    const data = makeRandomData(COUNT, INPUT_DIM);
    const ref = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed,
      nEpochs: 1,
      nNeighbors: K,
      initializeMethod: "random",
    });
    const knnIndices = new Int32Array(ref.knnIndices);
    const knnDistances = new Float32Array(ref.knnDistances);
    ref.destroy();
    return { knnIndices, knnDistances };
  }

  it("builds from a precomputed kNN graph (no data needed)", async () => {
    const { knnIndices, knnDistances } = await makeKnn();
    umap = await createUMAPFromKNN(COUNT, OUTPUT_DIM, knnIndices, knnDistances, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.outputDim).toBe(OUTPUT_DIM);
    expect(umap.epoch).toBe(0);

    const emb = umap.embedding;
    expect(emb.length).toBe(COUNT * OUTPUT_DIM);
    for (let i = 0; i < emb.length; i++) {
      expect(Number.isFinite(emb[i])).toBe(true);
    }

    // The echoed graph is row-major and consistent across indices/distances.
    expect(umap.knnIndices.length).toBe(umap.knnDistances.length);
    expect(umap.knnIndices.length % COUNT).toBe(0);
    expect(umap.knnIndices.length).toBeGreaterThan(0);

    await umap.run();
    expect(umap.epoch).toBe(10);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  it("is deterministic for a fixed seed", async () => {
    const { knnIndices, knnDistances } = await makeKnn();
    const opts = { seed: 7, nEpochs: 10, initializeMethod: "random" };

    const u1 = await createUMAPFromKNN(COUNT, OUTPUT_DIM, knnIndices, knnDistances, opts);
    await u1.run();
    const e1 = new Float32Array(u1.embedding);
    u1.destroy();

    const u2 = await createUMAPFromKNN(COUNT, OUTPUT_DIM, knnIndices, knnDistances, opts);
    await u2.run();
    const e2 = new Float32Array(u2.embedding);
    u2.destroy();

    expect(e1).toEqual(e2);
  });

  it("accepts a self-excluded graph and widens to include self", async () => {
    // K real neighbors per row, none of which is the point itself.
    const N = 50;
    const k = 5;
    const indices = new Int32Array(N * k);
    const distances = new Float32Array(N * k);
    for (let i = 0; i < N; i++) {
      for (let off = 1; off <= k; off++) {
        const c = i * k + (off - 1);
        indices[c] = (i + off) % N;
        distances[c] = off; // already ascending
      }
    }
    umap = await createUMAPFromKNN(N, OUTPUT_DIM, indices, distances, {
      seed: 1,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.embedding.length).toBe(N * OUTPUT_DIM);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
    // Self prepended -> width grows from k to k + 1.
    expect(umap.knnIndices.length).toBe(N * (k + 1));
  });

  it("throws on mismatched indices/distances length", async () => {
    const indices = new Int32Array(20 * 5);
    const distances = new Float32Array(20 * 6);
    await expect(createUMAPFromKNN(20, OUTPUT_DIM, indices, distances, { nEpochs: 5 })).rejects.toThrow(
      /equal length/i,
    );
  });

  it("throws when length is not divisible by count", async () => {
    const indices = new Int32Array(21);
    const distances = new Float32Array(21);
    await expect(createUMAPFromKNN(20, OUTPUT_DIM, indices, distances, { nEpochs: 5 })).rejects.toThrow(
      /divisible|nRows/i,
    );
  });

  it("runs the 0.6 pipeline from a precomputed kNN graph", async () => {
    const { knnIndices, knnDistances } = await makeKnn();
    umap = await createUMAPFromKNN(COUNT, OUTPUT_DIM, knnIndices, knnDistances, {
      seed: 42,
      nEpochs: 20,
      compatibilityLayout: false,
      optimizer: "adam",
      initializeMethod: "recursive",
      initScaleFactor: 1,
    });
    await umap.run();
    expect(umap.embedding.length).toBe(COUNT * OUTPUT_DIM);
    for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
  });

  it("rejects pca init and pca reset without data", async () => {
    const { knnIndices, knnDistances } = await makeKnn();
    await expect(
      createUMAPFromKNN(COUNT, OUTPUT_DIM, knnIndices, knnDistances, { nEpochs: 5, initializeMethod: "pca" }),
    ).rejects.toThrow(/pca/);
    umap = await createUMAPFromKNN(COUNT, OUTPUT_DIM, knnIndices, knnDistances, { seed: 42, nEpochs: 5 });
    expect(() => umap.reset("pca")).toThrow(/input data/);
    expect(() => umap.reset("spectral", new Float32Array(COUNT * INPUT_DIM))).toThrow(/kNN graph/);
    for (const method of ["tswspectral", "random", "recursive"]) {
      umap.reset(method);
      for (const v of umap.embedding) expect(Number.isFinite(v)).toBe(true);
    }
  });
});

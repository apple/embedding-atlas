import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { createUMAP } from "../index.js";
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

  it("returns empty embedding before run()", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.embedding).toBeInstanceOf(Float32Array);
    expect(umap.embedding.length).toBe(0);
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

  it("run() is idempotent", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    await umap.run();
    const emb1 = new Float32Array(umap.embedding);
    await umap.run();
    const emb2 = umap.embedding;
    expect(emb1).toEqual(emb2);
  });

  it("destroy() clears the embedding", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    await umap.run();
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

  it("returns empty knnIndices and knnDistances before run()", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      initializeMethod: "random",
    });
    expect(umap.knnIndices).toBeInstanceOf(Int32Array);
    expect(umap.knnIndices.length).toBe(0);
    expect(umap.knnDistances).toBeInstanceOf(Float32Array);
    expect(umap.knnDistances.length).toBe(0);
  });

  it("produces knnIndices and knnDistances after run()", async () => {
    const N_NEIGHBORS = 10;
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      seed: 42,
      nEpochs: 10,
      nNeighbors: N_NEIGHBORS,
      initializeMethod: "random",
    });
    await umap.run();
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

  it("throws on unknown metric", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      metric: "nonexistent",
      nEpochs: 10,
      initializeMethod: "random",
      seed: 42,
    });
    await expect(umap.run()).rejects.toThrow(/unknown metric/i);
  });

  it("throws when nNeighbors >= count", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    umap = await createUMAP(COUNT, INPUT_DIM, OUTPUT_DIM, data, {
      nNeighbors: COUNT,
      nEpochs: 10,
      initializeMethod: "random",
      seed: 42,
    });
    await expect(umap.run()).rejects.toThrow(/n_neighbors/i);
  });

  it("throws on data length mismatch", async () => {
    const data = makeRandomData(COUNT, INPUT_DIM);
    // Pass wrong count so data.length != count * inputDim
    umap = await createUMAP(COUNT + 1, INPUT_DIM, OUTPUT_DIM, data, {
      nEpochs: 10,
      initializeMethod: "random",
      seed: 42,
    });
    await expect(umap.run()).rejects.toThrow(/length/i);
  });
});

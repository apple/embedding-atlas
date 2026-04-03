import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { createNNDescent } from "../index.js";
import { initWasm, makeRandomData } from "./helpers.js";

beforeAll(() => initWasm());

describe("createNNDescent", () => {
  const COUNT = 200;
  const DIM = 10;
  const K = 5;

  let index;

  afterEach(() => {
    index?.destroy();
    index = null;
  });

  it("creates an NNDescent index", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    expect(index).toBeDefined();
    expect(typeof index.queryByIndex).toBe("function");
    expect(typeof index.queryByVector).toBe("function");
    expect(typeof index.destroy).toBe("function");
  });

  it("queryByIndex returns k results", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    const result = index.queryByIndex(0, K);
    expect(result.indices).toBeInstanceOf(Int32Array);
    expect(result.distances).toBeInstanceOf(Float32Array);
    expect(result.indices.length).toBe(K);
    expect(result.distances.length).toBe(K);
  });

  it("queryByVector returns k results", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    const query = data.subarray(0, DIM);
    const result = index.queryByVector(query, K);
    expect(result.indices).toBeInstanceOf(Int32Array);
    expect(result.distances).toBeInstanceOf(Float32Array);
    expect(result.indices.length).toBe(K);
    expect(result.distances.length).toBe(K);
  });

  it("indices are valid point indices", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    const result = index.queryByIndex(10, K);
    for (let i = 0; i < result.indices.length; i++) {
      expect(result.indices[i]).toBeGreaterThanOrEqual(0);
      expect(result.indices[i]).toBeLessThan(COUNT);
    }
  });

  it("distances are non-negative", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    const result = index.queryByIndex(0, K);
    for (let i = 0; i < result.distances.length; i++) {
      expect(result.distances[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it("distances are sorted in ascending order", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    const result = index.queryByIndex(0, K);
    for (let i = 1; i < result.distances.length; i++) {
      expect(result.distances[i]).toBeGreaterThanOrEqual(result.distances[i - 1]);
    }
  });

  it("works with cosine metric", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data, { metric: "cosine" });
    const result = index.queryByIndex(0, K);
    expect(result.indices.length).toBe(K);
    expect(result.distances.length).toBe(K);
  });

  it("can query multiple points independently", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    const r0 = index.queryByIndex(0, K);
    const r1 = index.queryByIndex(1, K);
    const sameIndices = r0.indices.length === r1.indices.length && r0.indices.every((v, i) => v === r1.indices[i]);
    expect(sameIndices).toBe(false);
  });

  it("queryByIndex throws on out-of-bounds index", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    expect(() => index.queryByIndex(-1, K)).toThrow(RangeError);
    expect(() => index.queryByIndex(COUNT, K)).toThrow(RangeError);
    expect(() => index.queryByIndex(COUNT + 100, K)).toThrow(RangeError);
  });

  it("destroy() is safe to call multiple times", async () => {
    const data = makeRandomData(COUNT, DIM);
    index = await createNNDescent(COUNT, DIM, data);
    index.destroy();
    index.destroy();
    index = null;
  });

  it("respects custom nNeighbors", async () => {
    const data = makeRandomData(COUNT, DIM);
    const k = 20;
    index = await createNNDescent(COUNT, DIM, data, { nNeighbors: k });
    // queryByIndex with k <= nNeighbors should use pre-computed graph
    const result = index.queryByIndex(0, k);
    expect(result.indices.length).toBe(k);
    expect(result.distances.length).toBe(k);
  });

  it("respects seed for deterministic results", async () => {
    const data = makeRandomData(COUNT, DIM);
    const opts = { seed: 123 };

    const i1 = await createNNDescent(COUNT, DIM, data, opts);
    const r1 = i1.queryByIndex(0, K);
    i1.destroy();

    const i2 = await createNNDescent(COUNT, DIM, data, opts);
    const r2 = i2.queryByIndex(0, K);
    i2.destroy();

    expect(r1.indices).toEqual(r2.indices);
    expect(r1.distances).toEqual(r2.distances);
  });

  it("throws on unknown metric", async () => {
    const data = makeRandomData(COUNT, DIM);
    await expect(createNNDescent(COUNT, DIM, data, { metric: "nonexistent" })).rejects.toThrow(/unknown metric/i);
  });

  it("throws when nNeighbors >= count", async () => {
    const smallCount = 10;
    const data = makeRandomData(smallCount, DIM);
    await expect(createNNDescent(smallCount, DIM, data, { nNeighbors: smallCount })).rejects.toThrow(/n_neighbors/i);
  });

  it("throws on data length mismatch", async () => {
    const data = makeRandomData(COUNT, DIM);
    await expect(createNNDescent(COUNT + 1, DIM, data)).rejects.toThrow(/length/i);
  });
});

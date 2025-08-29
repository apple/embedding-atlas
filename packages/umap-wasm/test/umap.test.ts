import { beforeAll, describe, expect, test } from "vitest";
import { createKNN, createUMAP, type UMAPOptions } from "../index.js";

describe("UMAP WebAssembly Module", () => {
  let testData: Float32Array;
  let count = 100;
  let inputDim = 4;
  let outputDim = 2;

  beforeAll(() => {
    // Generate test data: simple 4D points
    testData = new Float32Array(count * inputDim);
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < inputDim; j++) {
        testData[i * inputDim + j] = Math.random() * 10;
      }
    }
  });

  describe("createUMAP", () => {
    test("should create UMAP instance with default options", async () => {
      let umap = await createUMAP(count, inputDim, outputDim, testData);

      expect(umap).toBeDefined();
      expect(umap.inputDim).toBe(inputDim);
      expect(umap.outputDim).toBe(outputDim);
      expect(umap.epoch).toBe(0);

      umap.destroy();
    });

    test("should create UMAP instance with custom options", async () => {
      let options: UMAPOptions = {
        metric: "cosine",
        nNeighbors: 10,
        minDist: 0.5,
        spread: 2.0,
        nEpochs: 100,
        seed: 42,
      };

      let umap = await createUMAP(count, inputDim, outputDim, testData, options);
      expect(umap).toBeDefined();
      umap.destroy();
    });

    test("should return valid embedding array", async () => {
      let umap = await createUMAP(count, inputDim, outputDim, testData);

      let embedding = umap.embedding;
      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(count * outputDim);
      umap.destroy();
    });

    test("should run UMAP algorithm", async () => {
      let umap = await createUMAP(count, inputDim, outputDim, testData, { nEpochs: 10 });

      let initialEpoch = umap.epoch;
      umap.run(5);

      expect(umap.epoch).toBeGreaterThan(initialEpoch);
      expect(umap.epoch).toBeLessThanOrEqual(5);

      umap.destroy();
    });

    test("should run to completion when no epoch limit specified", async () => {
      let umap = await createUMAP(count, inputDim, outputDim, testData, { nEpochs: 5 });

      umap.run();

      expect(umap.epoch).toBe(5);

      umap.destroy();
    });

    test("should handle different metrics", async () => {
      let euclideanUmap = await createUMAP(count, inputDim, outputDim, testData, { metric: "euclidean" });
      let cosineUmap = await createUMAP(count, inputDim, outputDim, testData, { metric: "cosine" });

      expect(euclideanUmap).toBeDefined();
      expect(cosineUmap).toBeDefined();

      euclideanUmap.destroy();
      cosineUmap.destroy();
    });

    test("should handle different KNN methods", async () => {
      let hnswUmap = await createUMAP(count, inputDim, outputDim, testData, { knnMethod: "hnsw" });
      let vptreeUmap = await createUMAP(count, inputDim, outputDim, testData, { knnMethod: "vptree" });

      expect(hnswUmap).toBeDefined();
      expect(vptreeUmap).toBeDefined();

      hnswUmap.destroy();
      vptreeUmap.destroy();
    });

    test("should handle different initialization methods", async () => {
      let spectralUmap = await createUMAP(count, inputDim, outputDim, testData, { initializeMethod: "spectral" });
      let randomUmap = await createUMAP(count, inputDim, outputDim, testData, { initializeMethod: "random" });

      expect(spectralUmap).toBeDefined();
      expect(randomUmap).toBeDefined();

      spectralUmap.destroy();
      randomUmap.destroy();
    });

    test("should produce consistent results with same seed", async () => {
      let options: UMAPOptions = { seed: 42, nEpochs: 10 };

      let umap1 = await createUMAP(count, inputDim, outputDim, testData, options);
      umap1.run();
      let embedding1 = new Float32Array(umap1.embedding);
      umap1.destroy();

      let umap2 = await createUMAP(count, inputDim, outputDim, testData, options);
      umap2.run();
      let embedding2 = new Float32Array(umap2.embedding);
      umap2.destroy();

      // Results should be identical with same seed
      for (let i = 0; i < embedding1.length; i++) {
        expect(embedding1[i]).toBeCloseTo(embedding2[i], 5);
      }
    });

    test("should reduce dimensionality correctly", async () => {
      let higherDimInput = 10;
      let lowerDimOutput = 3;
      let testCount = 50;

      let higherDimData = new Float32Array(testCount * higherDimInput);
      for (let i = 0; i < testCount * higherDimInput; i++) {
        higherDimData[i] = Math.random() * 10;
      }

      let umap = await createUMAP(testCount, higherDimInput, lowerDimOutput, higherDimData);
      umap.run();

      let embedding = umap.embedding;
      expect(embedding.length).toBe(testCount * lowerDimOutput);

      // Verify all values are finite
      for (let i = 0; i < embedding.length; i++) {
        expect(Number.isFinite(embedding[i])).toBe(true);
      }

      umap.destroy();
    });

    test("should handle small datasets", async () => {
      let smallCount = 10;
      let smallData = new Float32Array(smallCount * inputDim);
      for (let i = 0; i < smallCount * inputDim; i++) {
        smallData[i] = Math.random() * 10;
      }

      let umap = await createUMAP(smallCount, inputDim, outputDim, smallData);
      umap.run();

      let embedding = umap.embedding;
      expect(embedding.length).toBe(smallCount * outputDim);

      umap.destroy();
    });
  });

  describe("createKNN", () => {
    test("should create KNN instance", async () => {
      let knn = await createKNN(count, inputDim, testData, { metric: "euclidean" });

      expect(knn).toBeDefined();
      expect(typeof knn.queryByIndex).toBe("function");
      expect(typeof knn.queryByVector).toBe("function");

      knn.destroy();
    });

    test("should query by index", async () => {
      let knn = await createKNN(count, inputDim, testData, { metric: "euclidean" });

      let result = knn.queryByIndex(0, 5);

      expect(result).toBeDefined();
      expect(result.indices).toBeInstanceOf(Int32Array);
      expect(result.distances).toBeInstanceOf(Float32Array);
      expect(result.indices.length).toBeLessThanOrEqual(5);
      expect(result.distances.length).toBeLessThanOrEqual(5);
      expect(result.indices.length).toBe(result.distances.length);

      knn.destroy();
    });

    test("should query by vector", async () => {
      let knn = await createKNN(count, inputDim, testData, { metric: "euclidean" });

      let queryVector = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      let result = knn.queryByVector(queryVector, 3);

      expect(result).toBeDefined();
      expect(result.indices).toBeInstanceOf(Int32Array);
      expect(result.distances).toBeInstanceOf(Float32Array);
      expect(result.indices.length).toBeLessThanOrEqual(3);
      expect(result.distances.length).toBeLessThanOrEqual(3);

      knn.destroy();
    });

    test("should handle different KNN methods", async () => {
      let hnswKnn = await createKNN(count, inputDim, testData, { method: "hnsw" });
      let vptreeKnn = await createKNN(count, inputDim, testData, { method: "vptree" });

      expect(hnswKnn).toBeDefined();
      expect(vptreeKnn).toBeDefined();

      hnswKnn.destroy();
      vptreeKnn.destroy();
    });

    test("should handle different metrics", async () => {
      let euclideanKnn = await createKNN(count, inputDim, testData, { metric: "euclidean" });
      let cosineKnn = await createKNN(count, inputDim, testData, { metric: "cosine" });

      expect(euclideanKnn).toBeDefined();
      expect(cosineKnn).toBeDefined();

      euclideanKnn.destroy();
      cosineKnn.destroy();
    });
  });
});

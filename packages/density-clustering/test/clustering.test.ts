import { beforeAll, describe, expect, test } from "vitest";
import { findClusters, type FindClustersOptions } from "../density_clustering_wasm/js/index.js";
import * as cluster from "../density_clustering_wasm/pkg/density_clustering_wasm.js";

describe("Density Clustering WebAssembly Module", () => {
  let simpleDensityMap: Float32Array;
  let width = 10;
  let height = 10;

  beforeAll(async () => {
    const isNode = typeof process !== "undefined" && !!process.versions?.node;
    if (isNode) {
      // In node we manually initialize the WASM module.
      // Once bundled this should be handled properly.
      let binary = require("fs").readFileSync(
        new URL("../density_clustering_wasm/pkg/density_clustering_wasm_bg.wasm", import.meta.url),
      );
      await cluster.default(binary);
    }

    // Create a simple density map with two distinct clusters
    simpleDensityMap = new Float32Array(width * height);

    // Fill with background density
    simpleDensityMap.fill(0.1);

    // Create first cluster (top-left)
    for (let y = 1; y < 4; y++) {
      for (let x = 1; x < 4; x++) {
        simpleDensityMap[y * width + x] = 1.0;
      }
    }
    // Peak in first cluster
    simpleDensityMap[2 * width + 2] = 2.0;

    // Create second cluster (bottom-right)
    for (let y = 6; y < 9; y++) {
      for (let x = 6; x < 9; x++) {
        simpleDensityMap[y * width + x] = 0.8;
      }
    }
    // Peak in second cluster
    simpleDensityMap[7 * width + 7] = 1.5;
  });

  describe("findClusters", () => {
    test("should find clusters with default options", async () => {
      let clusters = await findClusters(simpleDensityMap, width, height);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
      expect(clusters.length).toBeGreaterThan(0);
    });

    test("should return clusters with correct structure", async () => {
      let clusters = await findClusters(simpleDensityMap, width, height);

      expect(clusters.length).toBeGreaterThan(0);

      let cluster = clusters[0];
      expect(cluster).toBeDefined();
      expect(typeof cluster.identifier).toBe("number");
      expect(typeof cluster.sumDensity).toBe("number");
      expect(typeof cluster.meanX).toBe("number");
      expect(typeof cluster.meanY).toBe("number");
      expect(typeof cluster.maxDensity).toBe("number");
      expect(Array.isArray(cluster.maxDensityLocation)).toBe(true);
      expect(cluster.maxDensityLocation.length).toBe(2);
      expect(typeof cluster.pixelCount).toBe("number");

      if (cluster.boundary) {
        expect(Array.isArray(cluster.boundary)).toBe(true);
        expect(
          cluster.boundary.every(
            (polygon) => Array.isArray(polygon) && polygon.every((point) => Array.isArray(point) && point.length === 2),
          ),
        ).toBe(true);
      }

      if (cluster.boundaryRectApproximation) {
        expect(Array.isArray(cluster.boundaryRectApproximation)).toBe(true);
        expect(cluster.boundaryRectApproximation.every((rect) => Array.isArray(rect) && rect.length === 4)).toBe(true);
      }
    });

    test("should handle custom unionThreshold option", async () => {
      let options: Partial<FindClustersOptions> = {
        unionThreshold: 5,
      };

      let clusters = await findClusters(simpleDensityMap, width, height, options);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
    });

    test("should handle different unionThreshold values", async () => {
      let lowThreshold = await findClusters(simpleDensityMap, width, height, { unionThreshold: 1 });
      let highThreshold = await findClusters(simpleDensityMap, width, height, { unionThreshold: 20 });

      expect(lowThreshold).toBeDefined();
      expect(highThreshold).toBeDefined();

      // Different thresholds may produce different numbers of clusters
      expect(lowThreshold.length).toBeGreaterThanOrEqual(0);
      expect(highThreshold.length).toBeGreaterThanOrEqual(0);
    });

    test("should handle empty density map", async () => {
      let emptyMap = new Float32Array(width * height).fill(0);
      let clusters = await findClusters(emptyMap, width, height);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
    });

    test("should handle uniform density map", async () => {
      let uniformMap = new Float32Array(width * height).fill(0.5);
      let clusters = await findClusters(uniformMap, width, height);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
    });

    test("should handle single pixel density map", async () => {
      let singlePixelMap = new Float32Array(1).fill(1.0);
      let clusters = await findClusters(singlePixelMap, 1, 1);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
    });

    test("should validate cluster properties", async () => {
      let clusters = await findClusters(simpleDensityMap, width, height);

      for (let cluster of clusters) {
        expect(cluster.sumDensity).toBeGreaterThan(0);
        expect(cluster.maxDensity).toBeGreaterThan(0);
        expect(cluster.pixelCount).toBeGreaterThan(0);
        expect(cluster.meanX).toBeGreaterThanOrEqual(0);
        expect(cluster.meanX).toBeLessThan(width);
        expect(cluster.meanY).toBeGreaterThanOrEqual(0);
        expect(cluster.meanY).toBeLessThan(height);
        expect(cluster.maxDensityLocation[0]).toBeGreaterThanOrEqual(0);
        expect(cluster.maxDensityLocation[0]).toBeLessThan(width);
        expect(cluster.maxDensityLocation[1]).toBeGreaterThanOrEqual(0);
        expect(cluster.maxDensityLocation[1]).toBeLessThan(height);
        expect(cluster.maxDensity).toBeLessThanOrEqual(cluster.sumDensity);
      }
    });

    test("should handle rectangular density maps", async () => {
      let rectWidth = 15;
      let rectHeight = 5;
      let rectMap = new Float32Array(rectWidth * rectHeight);

      // Create a simple cluster in the middle
      for (let y = 1; y < 4; y++) {
        for (let x = 6; x < 10; x++) {
          rectMap[y * rectWidth + x] = 1.0;
        }
      }

      let clusters = await findClusters(rectMap, rectWidth, rectHeight);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
    });

    test("should be deterministic with same input", async () => {
      let clusters1 = await findClusters(simpleDensityMap, width, height);
      let clusters2 = await findClusters(simpleDensityMap, width, height);

      expect(clusters1.length).toBe(clusters2.length);

      // Sort clusters by identifier for comparison
      clusters1.sort((a, b) => a.identifier - b.identifier);
      clusters2.sort((a, b) => a.identifier - b.identifier);

      for (let i = 0; i < clusters1.length; i++) {
        expect(clusters1[i].identifier).toBe(clusters2[i].identifier);
        expect(clusters1[i].sumDensity).toBeCloseTo(clusters2[i].sumDensity, 5);
        expect(clusters1[i].meanX).toBeCloseTo(clusters2[i].meanX, 5);
        expect(clusters1[i].meanY).toBeCloseTo(clusters2[i].meanY, 5);
        expect(clusters1[i].maxDensity).toBeCloseTo(clusters2[i].maxDensity, 5);
        expect(clusters1[i].pixelCount).toBe(clusters2[i].pixelCount);
      }
    });

    test("should handle complex density patterns", async () => {
      let complexWidth = 20;
      let complexHeight = 20;
      let complexMap = new Float32Array(complexWidth * complexHeight);

      // Create a more complex pattern with multiple clusters
      for (let y = 0; y < complexHeight; y++) {
        for (let x = 0; x < complexWidth; x++) {
          let distance1 = Math.sqrt((x - 5) ** 2 + (y - 5) ** 2);
          let distance2 = Math.sqrt((x - 15) ** 2 + (y - 15) ** 2);
          let distance3 = Math.sqrt((x - 5) ** 2 + (y - 15) ** 2);

          let density = Math.max(Math.exp(-distance1 / 3), Math.exp(-distance2 / 4), Math.exp(-distance3 / 2)) * 0.1;

          complexMap[y * complexWidth + x] = density;
        }
      }

      let clusters = await findClusters(complexMap, complexWidth, complexHeight);

      expect(clusters).toBeDefined();
      expect(Array.isArray(clusters)).toBe(true);
    });
  });
});

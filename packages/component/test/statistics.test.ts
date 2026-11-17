import { describe, expect, test } from "vitest";

import { approximateDensity2D, approximateDensity2DByCategory } from "../src/lib/embedding_view/statistics.js";

/** A straightforward implementation, for comparison. */
function naiveByCategory(x: Float32Array, y: Float32Array, c: Uint8Array, binWidth: number, xs: number, ys: number) {
  let counts = new Map<string, number>();
  let max = 0;
  for (let i = 0; i < x.length; i++) {
    let bx = Math.floor((x[i] - xs) / binWidth) + 128;
    let by = Math.floor((y[i] - ys) / binWidth) + 128;
    if (bx < 0 || bx >= 256 || by < 0 || by >= 256) continue;
    let k = `${bx},${by},${c[i]}`;
    let v = (counts.get(k) ?? 0) + 1;
    counts.set(k, v);
    max = Math.max(max, v);
  }
  return max / (binWidth * binWidth);
}

function data(n: number, categories: number, seed: number) {
  let s = seed;
  let rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
  let x = new Float32Array(n),
    y = new Float32Array(n),
    c = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    c[i] = Math.floor(rand() * categories);
    x[i] = (c[i] % 7) + rand() * 2 - 1 + (rand() < 0.01 ? 1000 : 0); // some points outside the grid
    y[i] = Math.floor(c[i] / 7) + rand() * 2 - 1;
  }
  return { x, y, c };
}

describe("approximateDensity2DByCategory", () => {
  for (let categories of [2, 10, 100, 256]) {
    test(`matches a straightforward implementation (${categories} categories)`, () => {
      let { x, y, c } = data(20000, categories, categories);
      let expected = naiveByCategory(x, y, c, 0.05, 0.5, -0.2);
      expect(approximateDensity2DByCategory(x, y, c, categories, 0.05, 0.5, -0.2)).toBe(expected);
    });
  }

  test("doesn't change the shared grid used by approximateDensity2D", () => {
    let { x, y, c } = data(5000, 5, 1);
    let before = approximateDensity2D(x, y, 0.05, 0, 0);
    approximateDensity2DByCategory(x, y, c, 5, 0.05, 0, 0);
    expect(approximateDensity2D(x, y, 0.05, 0, 0)).toBe(before);
  });
});

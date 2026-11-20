import { describe, expect, test } from "vitest";

// A reference implementation of the point coverage in webgpu_renderer/program.wgsl (`footprint_radius`,
// `point_coverage`, and the quad of `point_vertex`), for checking its properties. Keep the two in sync.

function footprintRadius(size: number): number {
  if (size >= 1.5) {
    return size - 0.5;
  }
  let s = Math.max(size, 0);
  return s * s * (2 / 9) + s / 3;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
const overlap1d = (x: number, h: number) => clamp(Math.min(x + 0.5, h) - Math.max(x - 0.5, -h), 0, 1);

function pointCoverage(dx: number, dy: number, re: number): number {
  let h = 0.886226925 * re;
  let square = overlap1d(dx, h) * overlap1d(dy, h);
  let ramp = clamp(re + 0.5 - Math.hypot(dx, dy), 0, 1) * ((re * re) / (re * re + 1 / 12));
  let t = clamp(re - 1, 0, 1);
  return square * (1 - t) + ramp * t;
}

/** The coverage of each pixel drawn for a point at (cx, cy) (pixel centers are at i + 0.5): the pixels whose centers
 *  are inside the point's quad. */
function rasterize(cx: number, cy: number, size: number): number[] {
  let re = footprintRadius(size);
  let half = re + 0.5;
  let result: number[] = [];
  for (let i = Math.floor(cx - half) - 1; i <= Math.ceil(cx + half) + 1; i++) {
    for (let j = Math.floor(cy - half) - 1; j <= Math.ceil(cy + half) + 1; j++) {
      let dx = i + 0.5 - cx;
      let dy = j + 0.5 - cy;
      if (Math.abs(dx) <= half && Math.abs(dy) <= half) {
        result.push(pointCoverage(dx, dy, re));
      }
    }
  }
  return result;
}

const ink = (cx: number, cy: number, size: number) => rasterize(cx, cy, size).reduce((a, b) => a + b, 0);

/** Deterministic pseudo-random sub-pixel positions. */
function positions(n: number): [number, number][] {
  let result: [number, number][] = [];
  let seed = 1;
  let next = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  for (let i = 0; i < n; i++) {
    result.push([10 + next(), 10 + next()]);
  }
  return result;
}

describe("point coverage", () => {
  test("a point's total ink doesn't depend on its sub-pixel position", () => {
    for (let size of [0.1, 0.3, 0.6, 1, 1.4, 2, 4]) {
      let inks = positions(500).map(([x, y]) => ink(x, y, size));
      let mean = inks.reduce((a, b) => a + b, 0) / inks.length;
      let spread = Math.max(...inks.map((v) => Math.abs(v - mean))) / mean;
      // Exact (square footprints) below a radius of 1; within the disk ramp's own variation (~2%) above.
      expect(spread).toBeLessThan(0.03);
    }
  });

  test("the total ink is the footprint's area, and changes smoothly with the size", () => {
    let previous: number | null = null;
    for (let size = 0.05; size <= 4; size += 0.05) {
      let re = footprintRadius(size);
      let mean = positions(200).reduce((a, [x, y]) => a + ink(x, y, size), 0) / 200;
      expect(Math.abs(mean / (Math.PI * re * re) - 1)).toBeLessThan(0.06);
      if (previous != null) {
        // No jumps: consecutive sizes change the ink by a small, bounded amount.
        expect(Math.abs(mean - previous)).toBeLessThan(0.1 * Math.max(previous, 0.05) + 0.05);
      }
      previous = mean;
    }
  });

  test("coverage changes continuously as a point moves", () => {
    for (let size of [0.2, 0.7, 1.2, 3]) {
      let re = footprintRadius(size);
      // A pixel's coverage as the point moves along x in small steps (the pixel's offset changes accordingly).
      let values = Array.from({ length: 201 }, (_, i) => pointCoverage(-1 + i * 0.01, 0.13, re));
      for (let i = 1; i < values.length; i++) {
        expect(Math.abs(values[i] - values[i - 1])).toBeLessThan(0.05);
      }
    }
  });

  test("large points keep the previous disk: full inside, a 1px edge at the size", () => {
    let size = 5;
    let re = footprintRadius(size);
    expect(pointCoverage(0, 0, re)).toBeCloseTo((re * re) / (re * re + 1 / 12), 6);
    expect(pointCoverage(size + 0.01, 0, re)).toBe(0);
    expect(pointCoverage(size - 0.5, 0, re)).toBeGreaterThan(0.45);
    // Sizes that are already 0 stay invisible.
    expect(ink(10.3, 10.6, 0)).toBe(0);
  });
});

import { describe, expect, test } from "vitest";

import { parseColorNormalizedRgb } from "../src/lib/colors.js";

describe("parseColorNormalizedRgb", () => {
  test("transparent and invalid colors are transparent black (not NaN)", () => {
    for (let c of ["transparent", "#00000000", "rgba(255,0,0,0)", "not a color"]) {
      expect(parseColorNormalizedRgb(c)).toEqual({ r: 0, g: 0, b: 0, a: 0 });
    }
  });

  test("opaque and semi-transparent colors", () => {
    expect(parseColorNormalizedRgb("#ff0000")).toEqual({ r: 1, g: 0, b: 0, a: 1 });
    let c = parseColorNormalizedRgb("#0000ff80");
    expect(c.b).toBe(1);
    expect(c.a).toBeCloseTo(0.5, 2);
  });
});

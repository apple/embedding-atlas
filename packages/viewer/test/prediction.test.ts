// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { buildPredictBinary, buildPredictMulticlass } from "../src/charts/features/prediction.js";

/** Build a Map<string, number> from a plain object, for terse fixtures. */
function counts(obj: Record<string, number>): Map<string, number> {
  return new Map(Object.entries(obj));
}

describe("buildPredictBinary", () => {
  // classNames are ordered [negative, positive]; direction 1 = predicts positive.
  const classNames = ["neg", "pos"];

  it("predicts the positive class when the feature concentrates there", () => {
    // present: pos 90 / 100, neg 10 / 100 → strong association with `pos`.
    const p = buildPredictBinary(counts({ pos: 90, neg: 10 }), classNames, counts({ pos: 100, neg: 100 }), 0)!;
    expect(p.direction).toBe(1);
    expect(p.phi).toBeGreaterThan(0);
    // LOR of the Haldane–Anscombe table: log((90.5·90.5)/(10.5·10.5)).
    const expectedLor = Math.log((90.5 * 90.5) / (10.5 * 10.5));
    expect(p.strength).toBeCloseTo(expectedLor, 6); // z = 0 → strength == |LOR|
  });

  it("predicts the negative class when the feature concentrates there", () => {
    const p = buildPredictBinary(counts({ pos: 10, neg: 90 }), classNames, counts({ pos: 100, neg: 100 }), 0)!;
    expect(p.direction).toBe(0);
    expect(p.phi).toBeLessThan(0);
  });

  it("is symmetric: flipping the classes negates phi and flips direction", () => {
    const a = buildPredictBinary(counts({ pos: 90, neg: 10 }), classNames, counts({ pos: 100, neg: 100 }), 0)!;
    const b = buildPredictBinary(counts({ pos: 10, neg: 90 }), classNames, counts({ pos: 100, neg: 100 }), 0)!;
    expect(a.phi).toBeCloseTo(-b.phi!, 10);
    expect(a.strength).toBeCloseTo(b.strength, 10);
    expect(a.direction).not.toBe(b.direction);
  });

  it("yields strength 0 (no confident direction) under independence", () => {
    // Feature present in exactly half of each class → LOR ≈ 0, phi ≈ 0.
    const p = buildPredictBinary(counts({ pos: 50, neg: 50 }), classNames, counts({ pos: 100, neg: 100 }), 1.96)!;
    expect(p.phi).toBeCloseTo(0, 10);
    expect(p.strength).toBe(0);
  });

  it("clamps strength to 0 when the lower bound on |LOR| crosses zero", () => {
    // Weak signal on a small sample with a wide critical value → CI crosses 0.
    const p = buildPredictBinary(counts({ pos: 6, neg: 4 }), classNames, counts({ pos: 10, neg: 10 }), 3.0)!;
    expect(p.strength).toBe(0);
    // direction is still reported even when strength is 0.
    expect(p.direction).toBe(1);
  });

  it("keeps phi within [-1, 1]", () => {
    const p = buildPredictBinary(counts({ pos: 100, neg: 0 }), classNames, counts({ pos: 100, neg: 100 }), 0)!;
    expect(p.phi!).toBeGreaterThan(0);
    expect(p.phi!).toBeLessThanOrEqual(1);
  });

  it("returns undefined when a cross-filter selection empties one class", () => {
    // `neg` has zero selected rows: no contrast to predict from. Without the
    // guard the Haldane correction would report a spurious LOR toward `pos`.
    expect(buildPredictBinary(counts({ pos: 80 }), classNames, counts({ pos: 100, neg: 0 }), 0)).toBeUndefined();
    expect(buildPredictBinary(counts({ neg: 80 }), classNames, counts({ pos: 0, neg: 100 }), 0)).toBeUndefined();
  });
});

describe("buildPredictMulticlass", () => {
  it("returns undefined for a constant feature (present in every row)", () => {
    // m == N
    expect(
      buildPredictMulticlass(counts({ a: 100, b: 100 }), ["a", "b"], counts({ a: 100, b: 100 }), 200, 1.96),
    ).toBeUndefined();
  });

  it("returns undefined for a feature present in no row (m == 0)", () => {
    expect(
      buildPredictMulticlass(counts({ a: 0, b: 0 }), ["a", "b"], counts({ a: 100, b: 100 }), 200, 1.96),
    ).toBeUndefined();
  });

  it("returns undefined when fewer than two classes are non-empty", () => {
    // class b has zero rows in the (selected) totals → dropped → K = 1.
    expect(
      buildPredictMulticlass(counts({ a: 30, b: 0 }), ["a", "b"], counts({ a: 100, b: 0 }), 100, 1.96),
    ).toBeUndefined();
  });

  it("drops empty classes and keeps per-class entries in classNames order", () => {
    const p = buildPredictMulticlass(
      counts({ a: 5, b: 50, c: 0 }),
      ["a", "b", "c"],
      counts({ a: 100, b: 100, c: 0 }), // c is empty → dropped
      200,
      0,
    )!;
    expect(p.perClass!.map((x) => x.className)).toEqual(["a", "b"]);
  });

  it("scores a strongly predictive class positive and the others negative", () => {
    // Feature is overwhelmingly in class b.
    const p = buildPredictMulticlass(
      counts({ a: 5, b: 200, c: 5 }),
      ["a", "b", "c"],
      counts({ a: 300, b: 300, c: 300 }),
      900,
      2.0,
    )!;
    expect(p.strength).toBeGreaterThan(0);
    const byClass = Object.fromEntries(p.perClass!.map((x) => [x.className, x]));
    expect(byClass.b.score).toBeGreaterThan(0);
    expect(byClass.a.score).toBeLessThan(0);
    expect(byClass.c.score).toBeLessThan(0);
    // b is the dominant (largest) score.
    expect(byClass.b.score).toBeGreaterThan(byClass.a.score);
    expect(byClass.b.score).toBeGreaterThan(byClass.c.score);
  });

  it("clamps strength to 0 and scores to 0 when the feature carries no signal", () => {
    // Feature appears in the same proportion of every class → independent.
    const p = buildPredictMulticlass(
      counts({ a: 30, b: 30, c: 30 }),
      ["a", "b", "c"],
      counts({ a: 300, b: 300, c: 300 }),
      900,
      1.96,
    )!;
    expect(p.strength).toBe(0);
    for (const pc of p.perClass!) {
      expect(pc.score).toBe(0);
    }
  });

  it("computes the auxiliary display values for each class", () => {
    const N = 1000;
    const p = buildPredictMulticlass(
      counts({ neg: 50, pos: 300 }),
      ["neg", "pos"],
      counts({ neg: 600, pos: 400 }),
      N,
      0.5,
    )!;
    const m = 350;
    expect(p.support).toBe(m);

    const byClass = Object.fromEntries(p.perClass!.map((x) => [x.className, x]));
    // P(class)
    expect(byClass.neg.pClass).toBeCloseTo(600 / N, 10);
    expect(byClass.pos.pClass).toBeCloseTo(400 / N, 10);
    // P(class | feature present)
    expect(byClass.pos.pClassGivenFeature).toBeCloseTo(300 / m, 10);
    expect(byClass.neg.pClassGivenFeature).toBeCloseTo(50 / m, 10);
    // raw support per class
    expect(byClass.pos.supportInClass).toBe(300);
    expect(byClass.neg.supportInClass).toBe(50);
    // odds ratio is exp(LOR); CI brackets the LOR.
    expect(byClass.pos.oddsRatio).toBeCloseTo(Math.exp(byClass.pos.lor), 10);
    expect(byClass.pos.lorCiLow).toBeLessThan(byClass.pos.lor);
    expect(byClass.pos.lorCiHigh).toBeGreaterThan(byClass.pos.lor);
    // smoothed log-lift: log( ((a+1)/(m+K)) / P(class) ), K = 2.
    expect(byClass.pos.logLift).toBeCloseTo(Math.log((300 + 1) / (m + 2) / (400 / N)), 10);
  });

  describe("binary reduction (K = 2)", () => {
    it("yields equal-magnitude, opposite-sign per-class scores", () => {
      const p = buildPredictMulticlass(
        counts({ neg: 50, pos: 300 }),
        ["neg", "pos"],
        counts({ neg: 600, pos: 400 }),
        1000,
        0.5,
      )!;
      const [neg, pos] = p.perClass!;
      expect(neg.score).toBeCloseTo(-pos.score, 10);
      expect(neg.lor).toBeCloseTo(-pos.lor, 10);
      // The feature concentrates in `pos`, so `pos` is the positive direction.
      expect(pos.score).toBeGreaterThan(0);
    });
  });
});

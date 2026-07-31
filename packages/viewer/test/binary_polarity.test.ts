// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";
import { inferBinaryPolarity } from "../src/charts/features/binary_polarity.js";

describe("inferBinaryPolarity", () => {
  describe("boolean labels with positive-valenced column", () => {
    it("orders [false, true] for is_response_safe", () => {
      expect(inferBinaryPolarity("is_response_safe", ["true", "false"])).toEqual(["false", "true"]);
    });

    it("is order-independent in the input labels", () => {
      expect(inferBinaryPolarity("is_response_safe", ["false", "true"])).toEqual(["false", "true"]);
    });

    it("handles 1/0 the same way", () => {
      expect(inferBinaryPolarity("is_safe", ["1", "0"])).toEqual(["0", "1"]);
    });

    it("handles yes/no the same way", () => {
      expect(inferBinaryPolarity("success", ["yes", "no"])).toEqual(["no", "yes"]);
    });

    it("handles camelCase column names", () => {
      expect(inferBinaryPolarity("isResponseSafe", ["true", "false"])).toEqual(["false", "true"]);
    });
  });

  describe("boolean labels with negative-valenced column", () => {
    it("flips ordering for is_response_unsafe", () => {
      expect(inferBinaryPolarity("is_response_unsafe", ["true", "false"])).toEqual(["true", "false"]);
    });

    it("flips for has_error", () => {
      expect(inferBinaryPolarity("has_error", ["true", "false"])).toEqual(["true", "false"]);
    });

    it("flips for is_harmful with 1/0", () => {
      expect(inferBinaryPolarity("is_harmful", ["1", "0"])).toEqual(["1", "0"]);
    });

    it("flips for is_toxic with yes/no", () => {
      expect(inferBinaryPolarity("is_toxic", ["yes", "no"])).toEqual(["yes", "no"]);
    });
  });

  describe("boolean labels with neutral column", () => {
    it("returns undefined when no valence word is found", () => {
      expect(inferBinaryPolarity("outcome", ["true", "false"])).toBeUndefined();
    });

    it("returns undefined for an opaque column with 1/0", () => {
      expect(inferBinaryPolarity("flag_a", ["1", "0"])).toBeUndefined();
    });

    it("returns undefined when only auxiliaries remain after stripping", () => {
      // After dropping `is`, no tokens left → unknown valence.
      expect(inferBinaryPolarity("is", ["yes", "no"])).toBeUndefined();
    });
  });

  describe("semantic labels (opposite polarity)", () => {
    it("trusts pass/fail directly", () => {
      expect(inferBinaryPolarity("outcome", ["pass", "fail"])).toEqual(["fail", "pass"]);
    });

    it("trusts good/bad directly", () => {
      expect(inferBinaryPolarity("category", ["good", "bad"])).toEqual(["bad", "good"]);
    });

    it("trusts safe/unsafe regardless of column name", () => {
      expect(inferBinaryPolarity("anything_at_all", ["safe", "unsafe"])).toEqual(["unsafe", "safe"]);
    });

    it("does not let the column name override semantic labels", () => {
      // Even when the column itself is negative-valenced, the semantic labels win.
      expect(inferBinaryPolarity("is_unsafe", ["safe", "unsafe"])).toEqual(["unsafe", "safe"]);
    });
  });

  describe("returns undefined when the signal is weak", () => {
    it("both labels unknown words", () => {
      expect(inferBinaryPolarity("is_safe", ["foo", "bar"])).toBeUndefined();
    });

    it("both labels same polarity (semantic)", () => {
      expect(inferBinaryPolarity("is_safe", ["good", "ok"])).toBeUndefined();
    });

    it("both labels same polarity (boolean)", () => {
      // Both positive-valued booleans — degenerate but possible.
      expect(inferBinaryPolarity("is_safe", ["true", "yes"])).toBeUndefined();
    });

    it("mixed kinds (boolean + semantic) is conservatively undefined", () => {
      expect(inferBinaryPolarity("is_safe", ["true", "fail"])).toBeUndefined();
    });

    it("one label unknown", () => {
      expect(inferBinaryPolarity("is_safe", ["true", "X"])).toBeUndefined();
    });
  });

  describe("normalization", () => {
    it("is case-insensitive on labels", () => {
      expect(inferBinaryPolarity("is_response_safe", ["TRUE", "False"])).toEqual(["False", "TRUE"]);
    });

    it("trims whitespace on labels", () => {
      expect(inferBinaryPolarity("is_safe", ["  true  ", " false "])).toEqual([" false ", "  true  "]);
    });

    it("preserves the original label strings in the output", () => {
      const result = inferBinaryPolarity("outcome", ["Pass", "FAIL"]);
      expect(result).toEqual(["FAIL", "Pass"]);
    });

    it("is case-insensitive on column names", () => {
      expect(inferBinaryPolarity("IS_RESPONSE_UNSAFE", ["true", "false"])).toEqual(["true", "false"]);
    });
  });

  describe("real-world-ish names", () => {
    it("passed", () => {
      expect(inferBinaryPolarity("passed", ["true", "false"])).toEqual(["false", "true"]);
    });

    it("is_blocked", () => {
      expect(inferBinaryPolarity("is_blocked", ["true", "false"])).toEqual(["true", "false"]);
    });

    it("response.is_safe (dotted)", () => {
      expect(inferBinaryPolarity("response.is_safe", ["true", "false"])).toEqual(["false", "true"]);
    });

    it("first matching token wins (negative beats later positive)", () => {
      // `unsafe` is the first valence-word token, so the column reads as negative.
      expect(inferBinaryPolarity("unsafe_but_pass_through", ["true", "false"])).toEqual(["true", "false"]);
    });
  });
});

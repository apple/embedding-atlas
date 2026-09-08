// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";
import { fuzzyHighlightScorer, fuzzyMatch } from "../src/highlight/fuzzy-scorer.js";

describe("fuzzyMatch", () => {
  it("returns nothing for an empty or whitespace query", () => {
    expect(fuzzyMatch("hello world", "")).toEqual([]);
    expect(fuzzyMatch("hello world", "   ")).toEqual([]);
    expect(fuzzyMatch("hello world", null)).toEqual([]);
  });

  it("highlights an exact substring with an exclusive end and value 1", () => {
    const spans = fuzzyMatch("say lorem ipsum", "lorem");
    // "lorem" occupies [4, 9) (exclusive end).
    expect(spans).toHaveLength(1);
    expect(spans[0].start).toBe(4);
    expect(spans[0].end).toBe(9);
    expect(spans[0].value).toBe(1);
    expect("say lorem ipsum".slice(4, 9)).toBe("lorem");
  });

  it("returns an empty array when there is no match", () => {
    expect(fuzzyMatch("the quick brown fox", "zzzzzz")).toEqual([]);
  });

  it("tolerates a small typo (single error)", () => {
    const spans = fuzzyMatch("the quick brown fox", "quikc");
    expect(spans.length).toBeGreaterThan(0);
    for (const span of spans) {
      expect(span.value).toBe(1);
      expect(span.end).toBeGreaterThan(span.start);
    }
  });

  it("honors passthrough uFuzzy options (error tolerance)", () => {
    // intraMode 0 (MultiInsert) requires every term char present in order, so a
    // transposed typo no longer matches...
    expect(fuzzyMatch("the quick brown fox", "quikc", { intraMode: 0 })).toEqual([]);
    // ...but the exact term still does.
    expect(fuzzyMatch("the quick brown fox", "quick", { intraMode: 0 }).length).toBeGreaterThan(0);
  });

  it("does not highlight scattered short fragments that merely share letters", () => {
    // Regression: a bitap matcher used to also light up "ec" in "objective",
    // "rr" in "narrative", "rce" in "source", etc. Highlights should stay on the
    // genuine occurrence, not leak into unrelated words.
    const text = "an objective narrative about a cherry and the source";
    const word = text.indexOf("cherry");
    const spans = fuzzyMatch(text, "cherry");
    expect(spans.length).toBeGreaterThan(0);
    for (const span of spans) {
      // Every highlighted run lies within the "cherry" token.
      expect(span.start).toBeGreaterThanOrEqual(word);
      expect(span.end).toBeLessThanOrEqual(word + "cherry".length);
    }
  });
});

describe("fuzzyHighlightScorer", () => {
  it("scores a batch, returning one span array per input text", async () => {
    const scorer = fuzzyHighlightScorer();
    const result = await scorer(["lorem ipsum", "nothing here", "dolor lorem"], "lorem");
    expect(result).toHaveLength(3);
    expect(result[0].length).toBeGreaterThan(0);
    expect(result[1]).toEqual([]);
    expect(result[2].length).toBeGreaterThan(0);
  });

  it("returns empty arrays for an empty query without scanning", async () => {
    const scorer = fuzzyHighlightScorer();
    expect(await scorer(["a", "b"], "")).toEqual([[], []]);
  });
});

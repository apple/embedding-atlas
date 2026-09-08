// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { mergeBucketedSpans } from "../src/highlight/merge.js";

describe("mergeBucketedSpans", () => {
  it("returns the input unchanged for zero or one span", () => {
    expect(mergeBucketedSpans([], "abc")).toEqual([]);
    const one = [{ start: 0, end: 3, bucket: 2 }];
    expect(mergeBucketedSpans(one, "abc")).toEqual(one);
  });

  it("merges adjacent (abutting) spans of the same bucket", () => {
    // "fox" + "##es" → one span over "foxes".
    const text = "foxes";
    const merged = mergeBucketedSpans(
      [
        { start: 0, end: 3, bucket: 5 },
        { start: 3, end: 5, bucket: 5 },
      ],
      text,
    );
    expect(merged).toEqual([{ start: 0, end: 5, bucket: 5 }]);
  });

  it("bridges a whitespace-only gap between same-bucket spans", () => {
    const text = "the quick fox";
    const merged = mergeBucketedSpans(
      [
        { start: 0, end: 3, bucket: 4 }, // "the"
        { start: 4, end: 9, bucket: 4 }, // "quick"
        { start: 10, end: 13, bucket: 4 }, // "fox"
      ],
      text,
    );
    expect(merged).toEqual([{ start: 0, end: 13, bucket: 4 }]);
  });

  it("does not merge across non-whitespace text", () => {
    const text = "the quick brown fox";
    // "quick" and "fox" share a bucket but "brown" sits between them.
    const merged = mergeBucketedSpans(
      [
        { start: 4, end: 9, bucket: 6 }, // "quick"
        { start: 16, end: 19, bucket: 6 }, // "fox"
      ],
      text,
    );
    expect(merged).toEqual([
      { start: 4, end: 9, bucket: 6 },
      { start: 16, end: 19, bucket: 6 },
    ]);
  });

  it("keeps adjacent spans of different buckets separate", () => {
    const text = "foxes";
    const merged = mergeBucketedSpans(
      [
        { start: 0, end: 3, bucket: 5 },
        { start: 3, end: 5, bucket: 2 },
      ],
      text,
    );
    expect(merged).toEqual([
      { start: 0, end: 3, bucket: 5 },
      { start: 3, end: 5, bucket: 2 },
    ]);
  });

  it("sorts spans before merging", () => {
    const text = "the quick fox";
    const merged = mergeBucketedSpans(
      [
        { start: 10, end: 13, bucket: 1 },
        { start: 0, end: 3, bucket: 1 },
        { start: 4, end: 9, bucket: 1 },
      ],
      text,
    );
    expect(merged).toEqual([{ start: 0, end: 13, bucket: 1 }]);
  });

  it("starts a new run after a bucket change and merges within each", () => {
    const text = "aaa bbb ccc ddd";
    const merged = mergeBucketedSpans(
      [
        { start: 0, end: 3, bucket: 1 }, // "aaa"
        { start: 4, end: 7, bucket: 1 }, // "bbb"
        { start: 8, end: 11, bucket: 3 }, // "ccc"
        { start: 12, end: 15, bucket: 3 }, // "ddd"
      ],
      text,
    );
    expect(merged).toEqual([
      { start: 0, end: 7, bucket: 1 },
      { start: 8, end: 15, bucket: 3 },
    ]);
  });
});

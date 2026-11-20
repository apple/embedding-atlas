import { describe, expect, test } from "vitest";

import {
  bitsetAny,
  bitsetForEach,
  bitsetHas,
  KeyIndex,
  normalizeKey,
} from "../src/lib/embedding_view/data/key_index.js";

describe("KeyIndex", () => {
  test("consecutive integer keys map by offset", () => {
    let index = new KeyIndex(new Int32Array([5, 6, 7, 8]));
    expect(index.positionOf(5)).toBe(0);
    expect(index.positionOf(8)).toBe(3);
    expect(index.positionOf(4)).toBe(-1);
    expect(index.positionOf(9)).toBe(-1);
    expect(index.positionOf(6.5)).toBe(-1);
    expect(index.positionOf("6")).toBe(-1);
  });

  test("sparse integer keys", () => {
    let index = new KeyIndex([2, 10, 11, 40]);
    expect(index.positionOf(10)).toBe(1);
    expect(index.positionOf(40)).toBe(3);
    expect(index.positionOf(3)).toBe(-1);
  });

  test("string keys", () => {
    let index = new KeyIndex(["a", "b", "d"]);
    expect(index.positionOf("d")).toBe(2);
    expect(index.positionOf("c")).toBe(-1);
  });

  test("bigint keys (e.g., BIGINT or rowid) are compared as numbers", () => {
    let index = new KeyIndex(new BigInt64Array([0n, 1n, 2n]));
    expect(index.positionOf(1n)).toBe(1);
    expect(index.positionOf(2)).toBe(2);
    let sparse = new KeyIndex(new BigInt64Array([3n, 9n]));
    expect(sparse.positionOf(9n)).toBe(1);
    expect(normalizeKey(2n ** 60n)).toBe(2n ** 60n);
  });

  test("empty table", () => {
    let index = new KeyIndex(new Int32Array(0));
    expect(index.length).toBe(0);
    expect(index.positionOf(0)).toBe(-1);
    expect(index.chunks(10)).toEqual([]);
  });

  test("bitsetOf ignores unknown and duplicate keys", () => {
    let index = new KeyIndex([10, 20, 30, 40, 50]);
    let { bits, count } = index.bitsetOf([20, 50, 50, 99]);
    expect(count).toBe(2);
    expect([0, 1, 2, 3, 4].map((p) => bitsetHas(bits, p))).toEqual([false, true, false, false, true]);
  });

  test("bitsets over 32 rows", () => {
    let keys = Array.from({ length: 70 }, (_, i) => i);
    let index = new KeyIndex(keys);
    let { bits, count } = index.bitsetOf([0, 31, 32, 69]);
    expect(count).toBe(4);
    expect(bits.length).toBe(3);
    expect([0, 31, 32, 33, 69].map((p) => bitsetHas(bits, p))).toEqual([true, true, true, false, true]);
  });

  test("chunks cover the positions with key ranges", () => {
    let index = new KeyIndex([1, 3, 5, 7, 9]);
    expect(index.chunks(2)).toEqual([
      { start: 0, end: 2, low: 1, high: 3 },
      { start: 2, end: 4, low: 5, high: 7 },
      { start: 4, end: 5, low: 9, high: 9 },
    ]);
  });
});

describe("KeyIndex uniqueness", () => {
  test("detects duplicate and null keys", () => {
    expect(new KeyIndex([0, 1, 2]).unique).toBe(true);
    expect(new KeyIndex(["a", "c", "b"]).unique).toBe(true);
    expect(new KeyIndex([1, 1, 2]).unique).toBe(false);
    expect(new KeyIndex(["a", "a"]).unique).toBe(false);
    expect(new KeyIndex([null, 1, 2]).unique).toBe(false);
  });
});

describe("bitset helpers", () => {
  let s = 7;
  let rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
  let length = 200;
  let bits = new Uint32Array(Math.ceil(length / 32));
  for (let p = 0; p < length; p++) {
    if (rand() < 0.1 || p == 31 || p == 32 || p == 63) {
      bits[p >>> 5] |= 1 << (p & 31);
    }
  }
  let expected = Array.from({ length }, (_, p) => p).filter((p) => bitsetHas(bits, p));

  test("bitsetForEach visits the set positions in order", () => {
    let visited: number[] = [];
    bitsetForEach(bits, (p) => visited.push(p));
    expect(visited).toEqual(expected);
  });

  test("bitsetAny matches checking each position", () => {
    for (let start = 0; start < length; start += 7) {
      for (let end = start; end <= length; end += 5) {
        let naive = expected.some((p) => p >= start && p < end);
        expect(bitsetAny(bits, start, end)).toBe(naive);
      }
    }
    expect(bitsetAny(new Uint32Array([1 << 31]), 31, 32)).toBe(true);
    expect(bitsetAny(new Uint32Array([1 << 31]), 0, 31)).toBe(false);
    expect(bitsetAny(new Uint32Array([0, 1]), 32, 33)).toBe(true);
  });
});

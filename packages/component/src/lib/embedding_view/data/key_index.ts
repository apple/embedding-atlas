// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/** Maps the row keys of a table (its identifier column, or DuckDB's `rowid`) to positions in the table's column
 *  arrays. Columns are loaded ordered by the key, so position `i` holds the row with the `i`-th smallest key. */
export class KeyIndex {
  /** The keys, in increasing order (as loaded from the database). */
  readonly keys: ArrayLike<any>;
  /** If the keys are consecutive integers, the first key (a position is then `key - first`). */
  private readonly first: number | null;
  /** Otherwise, a map from key to position. */
  private readonly map: Map<any, number> | null;
  /** Whether the keys are unique and not null. Keys must be unique to identify rows (e.g., for masks). */
  readonly unique: boolean;

  constructor(keys: ArrayLike<any>) {
    this.keys = keys;
    let first = consecutiveIntegersStart(keys);
    this.first = first;
    let unique = true;
    if (first == null) {
      let map = new Map<any, number>();
      for (let i = 0; i < keys.length; i++) {
        let key = normalizeKey(keys[i]);
        if (key == null || map.has(key)) {
          unique = false;
        }
        map.set(key, i);
      }
      this.map = map;
    } else {
      this.map = null;
    }
    this.unique = unique;
  }

  get length(): number {
    return this.keys.length;
  }

  /** The position of a key, or -1 if the key is not in the table. */
  positionOf(key: any): number {
    key = normalizeKey(key);
    if (this.first != null) {
      let p = typeof key == "number" ? key - this.first : NaN;
      return Number.isInteger(p) && p >= 0 && p < this.keys.length ? p : -1;
    }
    return this.map!.get(key) ?? -1;
  }

  /** A bitset of the positions of the given keys (keys not in the table are ignored), and the number of set bits. */
  bitsetOf(keys: ArrayLike<any>): { bits: Uint32Array<ArrayBuffer>; count: number } {
    let bits = new Uint32Array(Math.ceil(this.keys.length / 32));
    let count = 0;
    for (let i = 0; i < keys.length; i++) {
      let p = this.positionOf(keys[i]);
      if (p < 0) {
        continue;
      }
      let word = p >>> 5;
      let bit = 1 << (p & 31);
      if ((bits[word] & bit) == 0) {
        bits[word] |= bit;
        count += 1;
      }
    }
    return { bits, count };
  }

  /** Ranges of keys covering consecutive positions, for querying a table in chunks:
   *  `[keys[start], keys[end - 1]]` for chunks of `chunkSize` positions. */
  chunks(chunkSize: number): { start: number; end: number; low: any; high: any }[] {
    let result: { start: number; end: number; low: any; high: any }[] = [];
    for (let start = 0; start < this.keys.length; start += chunkSize) {
      let end = Math.min(start + chunkSize, this.keys.length);
      result.push({ start, end, low: this.keys[start], high: this.keys[end - 1] });
    }
    return result;
  }
}

/** Keys are compared as numbers when they're safe integers (e.g., BIGINT keys arrive as bigint values). */
export function normalizeKey(key: any): any {
  if (typeof key == "bigint" && key >= BigInt(Number.MIN_SAFE_INTEGER) && key <= BigInt(Number.MAX_SAFE_INTEGER)) {
    return Number(key);
  }
  return key;
}

/** The first key if the keys are consecutive integers (in increasing order), otherwise null. */
function consecutiveIntegersStart(keys: ArrayLike<any>): number | null {
  if (keys.length == 0) {
    return 0;
  }
  let first = normalizeKey(keys[0]);
  if (typeof first != "number" || !Number.isInteger(first)) {
    return null;
  }
  for (let i = 1; i < keys.length; i++) {
    if (normalizeKey(keys[i]) !== first + i) {
      return null;
    }
  }
  return first;
}

/** Whether position `p` is set in a bitset. */
export function bitsetHas(bits: Uint32Array, p: number): boolean {
  return (bits[p >>> 5] & (1 << (p & 31))) != 0;
}

/** Call `fn` for each position set in a bitset, in increasing order (skipping empty words). */
export function bitsetForEach(bits: Uint32Array, fn: (p: number) => void) {
  for (let word = 0; word < bits.length; word++) {
    let w = bits[word];
    while (w != 0) {
      let lowest = w & -w;
      fn((word << 5) + 31 - Math.clz32(lowest));
      w ^= lowest;
    }
  }
}

/** Whether any position in [start, end) is set in a bitset. */
export function bitsetAny(bits: Uint32Array, start: number, end: number): boolean {
  if (start >= end) {
    return false;
  }
  let first = start >>> 5;
  let last = (end - 1) >>> 5;
  for (let word = first; word <= last; word++) {
    let w = bits[word] ?? 0;
    if (word == first) {
      w &= ~0 << (start & 31);
    }
    if (word == last && (end & 31) != 0) {
      w &= ~0 >>> (32 - (end & 31));
    }
    if (w != 0) {
      return true;
    }
  }
  return false;
}

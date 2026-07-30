// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Returns a short non-secure hash for an object.
 * The object must be JSON-serializable, and the hash is sensitive to object key order.
 */
export function objectHash(object: any): string {
  let json = JSON.stringify(object);
  return stringHash(json);
}

/** cyrb53 (c) 2018 bryc (github.com/bryc)
 * License: Public domain (or MIT if needed). Attribution appreciated.
 *
 * A fast and simple 53-bit string hash function with decent collision resistance.
 * Largely inspired by MurmurHash2/3, but with a focus on speed/simplicity.
 *
 * @param data The input data as a Uint8Array.
 * @param seed An optional seed value.
 * @returns A 64-bit hash value as two 32-bit numbers.
 */
function cyrb64(data: Uint8Array, seed: number = 0): [number, number] {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < data.length; i++) {
    let ch = data[i];
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return [h2 >>> 0, h1 >>> 0];
}

/** Returns a short non-secure hash for a string */
export function stringHash(str: string): string {
  let encoder = new TextEncoder();
  let data = encoder.encode(str);
  let hash = cyrb64(data);
  return hash[0].toString(16).padStart(8, "0") + hash[1].toString(16).padStart(8, "0");
}

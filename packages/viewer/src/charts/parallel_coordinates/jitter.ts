// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Category-axis jitter.
 *
 * Each of the `categoryCount` categories owns an equal slot of [0, 1] (`[i/N, (i+1)/N]`) centered at
 * `(i + 0.5)/N`. Points in a category are spread across a centered band whose width is **linearly
 * proportional to the category's count** (so band size reads as count directly), capped at
 * `SPREAD_FRACTION` of the slot. Points fill that band as a **jittered grid**: placed on an evenly-spaced
 * lattice by their rank, plus an in-cell dither, so the fill is uniform and free of lattice/moiré artifacts.
 *
 *  - Adjacent spacing is `SEPARATION_PX` (≈ 1/256 CSS px) until the band saturates the slot; past that the
 *    spacing shrinks below it (denser). So a small category stays a thin sub-pixel line (it only starts to
 *    visibly widen after a few hundred points), the band width scales linearly with count, and it never
 *    exceeds the slot.
 *
 * Ordering **and** dither are **independent per axis and stable across selections**: the rank key is
 * `hash(order[i] ^ seed)` and the dither is a re-hash of it. The per-axis `seed` makes each axis rank and
 * dither its points independently — the caller shares one `order` array across axes, and with a shared
 * order every pair of categorical axes would rank points identically, so a category→category bundle would
 * become a coherent grating (a stripe/moiré in the fan). Per-axis re-hashing decorrelates them: bundles
 * cross and blend into a smooth density. Because each key is a fixed hash of the row id, a row keeps its
 * relative position as a selection filters, so bands re-pack without reshuffling.
 *
 * Ranks are computed with a per-category histogram over the key's high bits (O(numRows), no sort).
 */

/** Target adjacent spacing between points in a category band, in CSS px (also the linear band-growth slope:
 *  the band widens by this much per extra point, so it needs ~256 points to reach 1px and stays a thin line
 *  for small categories). */
const SEPARATION_PX = 1 / 256;
/** Max band width as a fraction of a category's [0,1] slot. */
const SPREAD_FRACTION = 0.5;
/** Number of hash buckets used to rank rows by the order hash (2^bits). */
const BUCKET_BITS = 14;
const BUCKETS = 1 << BUCKET_BITS;

/** A uint32 hash of a row index — fallback ordering when no row-id hash is available. */
export function hashIndex(i: number): number {
  let x = (i + 1) | 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * Jitter a per-row category-index column into normalized [0, 1] axis positions. O(numRows); allocates per
 * call (intended to run on every Mosaic selection update).
 *
 * @param category      per-row category index in [0, categoryCount)
 * @param categoryCount number of categories (slots)
 * @param plotHeightPx  the axis plot height in CSS px (used to convert SEPARATION_PX to normalized units)
 * @param order         per-row uint32 ordering key (a hash of the row id) — gives a stable order across selections
 * @param seed          per-axis seed mixed into the ordering key so each axis ranks its points independently
 *                      (decorrelates axes; see the module doc — prevents the cross-axis stripe artifact)
 */
export function applyCategoryJitter(
  category: ArrayLike<number>,
  categoryCount: number,
  plotHeightPx: number,
  order: ArrayLike<number>,
  seed: number = 0,
): Float32Array {
  let n = category.length;
  let result = new Float32Array(n);
  let count = Math.max(1, categoryCount);
  let slot = 1 / count;
  let maxSpread = SPREAD_FRACTION * slot;
  let separation = SEPARATION_PX / Math.max(1, plotHeightPx);
  let shift = 32 - BUCKET_BITS;

  // Per-axis ordering key: re-hash the shared order key with the axis seed so this axis's ranking is
  // independent of the others. Without this, categorical axes sharing one `order` array rank points
  // identically and their bundles form coherent non-crossing threads (the stripe/moiré artifact).
  let keys = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    keys[i] = hashIndex(order[i] ^ seed);
  }

  // Pass 1: per-category counts + per-(category, hash-bucket) histogram.
  let counts = new Uint32Array(count);
  let hist = new Uint32Array(count * BUCKETS);
  for (let i = 0; i < n; i++) {
    let c = category[i];
    if (c < 0 || c >= count) {
      continue;
    }
    counts[c]++;
    hist[c * BUCKETS + (keys[i] >>> shift)]++;
  }

  // Prefix-sum each category's histogram → starting rank for each bucket.
  for (let c = 0; c < count; c++) {
    let base = c * BUCKETS;
    let acc = 0;
    for (let b = 0; b < BUCKETS; b++) {
      let v = hist[base + b];
      hist[base + b] = acc;
      acc += v;
    }
  }

  // Per-category slot center + lattice spacing.
  // Band width is LINEAR in the count, capped at maxSpread: extent = min((m-1)*separation, maxSpread).
  //  - below the cap: spacing = separation (≈ 1/256 CSS px), so the band widens by ~1/256 px per point
  //    (stays a thin line until a few hundred points) and scales linearly with count;
  //  - at the cap: spacing = maxSpread/(m-1) < separation, so a large category fills the slot densely.
  let center = new Float32Array(count);
  let spacing = new Float32Array(count);
  for (let c = 0; c < count; c++) {
    center[c] = (c + 0.5) * slot;
    let m = counts[c];
    spacing[c] = m > 1 ? Math.min(separation, maxSpread / (m - 1)) : 0;
  }

  // Pass 2: dense rank (hash-ordered) → centered lattice position, plus an in-cell dither (an independent
  // re-hash of the key, so it's decorrelated from the rank yet stable per row/axis) so the band fills as a
  // jittered grid rather than a rigid lattice.
  for (let i = 0; i < n; i++) {
    let c = category[i];
    if (c < 0 || c >= count) {
      result[i] = 0.5;
      continue;
    }
    let m = counts[c];
    if (m <= 1) {
      result[i] = center[c];
      continue;
    }
    let rank = hist[c * BUCKETS + (keys[i] >>> shift)]++;
    let dither = (hashIndex(keys[i]) / 4294967296 - 0.5) * spacing[c];
    result[i] = center[c] + (rank - (m - 1) / 2) * spacing[c] + dither;
  }

  return result;
}

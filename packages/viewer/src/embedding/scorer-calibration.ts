// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { quantile } from "d3";

import { chunkInputs } from "../inference/utils.js";

/**
 * A labeled (query, text) pair used to calibrate the cosine-similarity range of
 * an embedding model.
 *
 * - `query` mirrors how queries look in practice: a short lowercase concept
 *   phrase (e.g. "bicycle repair and maintenance"), not a full question.
 * - `text` snippets are intentionally short (~8 words) so the calibration
 *   measures similarities at the same granularity the scorer does (it embeds
 *   sliding word windows, default `windowSize: 8`).
 */
export interface CalibrationPair {
  query: string;
  text: string;
  relevant: boolean;
}

/**
 * Fixed calibration set. Every pair uses a distinct query, so no phrase is
 * over-represented in the resulting distributions. Topics are deliberately
 * mundane (hobbies, cooking, everyday science) with no political, sensitive, or
 * otherwise charged content, so the set is safe to ship in a public repo.
 *
 * Positives pair a concept phrase with a snippet that clearly expresses it.
 * Negatives are deliberately *marginal* — a snippet that shares the query's
 * topic or vocabulary but isn't actually a match. The score band is derived
 * from the negative and positive distributions (see {@link calibrateSimilarityRange}),
 * so the negatives must cluster around the just-below-match boundary; obvious
 * non-matches would pull the floor too low and let weak matches through.
 */
// prettier-ignore
export const CALIBRATION_PAIRS: CalibrationPair[] = [
  // Positives: query phrase + a snippet that expresses it.
  { query: "home cooking and recipes", text: "She simmered the sauce slowly and seasoned it to taste.", relevant: true },
  { query: "houseplant care and watering", text: "Water the fern weekly and keep it in indirect light.", relevant: true },
  { query: "bicycle repair and maintenance", text: "He patched the inner tube and oiled the bike chain.", relevant: true },
  { query: "morning coffee preparation", text: "Grind the beans fresh and pour water just off boiling.", relevant: true },
  { query: "long-distance running training", text: "Build weekly mileage gradually to prepare for the marathon.", relevant: true },
  { query: "acoustic guitar playing", text: "Strum the open chords softly while keeping a steady rhythm.", relevant: true },
  { query: "freshwater aquarium setup", text: "Cycle the tank and test the water before adding fish.", relevant: true },
  { query: "mountain hiking and trails", text: "The steep trail climbed through forest toward the rocky summit.", relevant: true },
  { query: "vegetable gardening basics", text: "Plant the seedlings in full sun and water them regularly.", relevant: true },
  { query: "board game strategy", text: "Plan several moves ahead and control the center early.", relevant: true },
  { query: "landscape photography techniques", text: "Shoot at golden hour for soft, warm landscape light.", relevant: true },
  { query: "knitting and crochet projects", text: "She cast on stitches and knit a wool scarf.", relevant: true },
  { query: "backyard bird watching", text: "We spotted a heron and noted it in the logbook.", relevant: true },
  { query: "baking bread at home", text: "Knead the dough, let it rise, then bake it.", relevant: true },
  // Negatives: marginal near-misses — each shares the query's topic or
  // vocabulary but isn't actually a match.
  { query: "ocean sailing and navigation", text: "They watched the sailboats drift across the harbor at dusk.", relevant: false },
  { query: "classical piano lessons", text: "She hummed a familiar classical tune while making tea.", relevant: false },
  { query: "camping and tent setup", text: "They roasted marshmallows around the campfire late into the night.", relevant: false },
  { query: "pottery and ceramics", text: "She bought a handmade ceramic mug at the craft fair.", relevant: false },
  { query: "model train collecting", text: "He lined up the toy cars neatly along the shelf.", relevant: false },
  { query: "yoga and stretching routines", text: "She unrolled the mat for her morning workout session.", relevant: false },
  { query: "stargazing and astronomy", text: "We admired the bright full moon above the open field.", relevant: false },
  { query: "homemade pasta cooking", text: "He boiled the store-bought noodles for a quick weeknight dinner.", relevant: false },
  { query: "rock climbing techniques", text: "They scrambled over the boulders to reach the lookout point.", relevant: false },
  { query: "tropical fish breeding", text: "The pet store displayed colorful fish in glass tanks.", relevant: false },
  { query: "watercolor painting basics", text: "He doodled in the notebook margins with a ballpoint pen.", relevant: false },
  { query: "vintage car restoration", text: "She browsed photos of classic cars at the weekend show.", relevant: false },
  { query: "chess opening strategy", text: "They set up the checkerboard for a quick evening match.", relevant: false },
  { query: "beekeeping and honey harvesting", text: "She spread fresh honey on warm toast for breakfast.", relevant: false },
];

/** Cosine-similarity range used when calibration can't produce a usable one. */
const FALLBACK_RANGE: [number, number] = [0.3, 0.5];

/** Smallest gap kept between `min` and `max` so the score mapping never collapses. */
const MIN_BAND_WIDTH = 0.05;

/**
 * Derive a per-model cosine-similarity range from labeled examples. Higher
 * similarity means more relevant, so the negatives sit below the positives; the
 * range maps that separation onto the `[0, 1]` highlight score:
 *
 * - `min` (score-0 floor) = the 25th percentile of negative similarities — the
 *   low end of the near-miss distribution, so only text clearly below a typical
 *   near-miss maps to 0 and the gradient starts early.
 * - `max` (score-1 saturation) = the 75th percentile of positive similarities,
 *   so only strong matches saturate to 1.
 *
 * Together these span the bulk of both distributions, giving a wide, gradual
 * band rather than a sharp threshold.
 *
 * Percentiles (not mean/std) are used so a single outlier example can't drag the
 * band. `embed` is the scorer's own embedding function: it returns pooled,
 * L2-normalized vectors, so a dot product is the cosine similarity directly.
 */
export async function calibrateSimilarityRange(
  embed: (inputs: string[]) => Promise<{ data: Float32Array; dim: number }>,
  pairs: CalibrationPair[],
  batchSize: number,
): Promise<[number, number]> {
  // Each query/text appears in several pairs; embed every distinct string once.
  let queries = [...new Set(pairs.map((p) => p.query))];
  let texts = [...new Set(pairs.map((p) => p.text))];
  let queryVecs = await embedAll(embed, queries, batchSize);
  let textVecs = await embedAll(embed, texts, batchSize);

  let positives: number[] = [];
  let negatives: number[] = [];
  for (let p of pairs) {
    let sim = dot(queryVecs.get(p.query)!, textVecs.get(p.text)!);
    (p.relevant ? positives : negatives).push(sim);
  }
  if (positives.length === 0 || negatives.length === 0) {
    return FALLBACK_RANGE;
  }

  // d3's `quantile` assumes ascending input.
  positives.sort((a, b) => a - b);
  negatives.sort((a, b) => a - b);
  let min = quantile(negatives, 0.25)!;
  let max = quantile(positives, 0.75)!;
  // Poor separation (overlapping classes) can invert or collapse the band; keep
  // a minimum width so `similarityToScore` stays a ramp rather than a step.
  if (max <= min) {
    max = min + MIN_BAND_WIDTH;
  }
  return [min, max];
}

/**
 * Embed `inputs` and return a map from each input string to its own copy of the
 * pooled vector (copied out so it survives the underlying tensor being reused).
 * `batchSize` is just a coarse cap on how many inputs are handed over per call;
 * the model layer re-batches each call to the provider's real hardware limit.
 */
async function embedAll(
  embed: (inputs: string[]) => Promise<{ data: Float32Array; dim: number }>,
  inputs: string[],
  batchSize: number,
): Promise<Map<string, Float32Array>> {
  let out = new Map<string, Float32Array>();
  for (let chunk of chunkInputs(inputs, batchSize)) {
    let { data, dim } = await embed(chunk);
    for (let i = 0; i < chunk.length; i++) {
      out.set(chunk[i], new Float32Array(data.subarray(i * dim, i * dim + dim)));
    }
  }
  return out;
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  let n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  return s;
}

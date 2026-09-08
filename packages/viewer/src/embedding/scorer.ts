// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { HighlightSpan } from "../highlight/scheduler.js";
import { loadEmbeddingModelCached, type EmbeddingModel } from "../inference/embedding.js";
import { type ProviderConfig } from "../inference/provider_config.js";
import { chunkInputs } from "../inference/utils.js";
import { calibrateSimilarityRange, CALIBRATION_PAIRS } from "./scorer-calibration.js";

/**
 * Inputs handed to the model per embed call. Deliberately large: the model layer
 * (`batchedEmbeddings`) re-chunks each call to the provider's real limit, so this
 * is just a coarse cap rather than a tuned hardware batch size.
 */
const EMBEDDING_BATCH_SIZE = 4096;

export interface ScoringOptions {
  /**
   * Per-model cosine-similarity range mapped onto the `[0, 1]` score. When
   * omitted it is auto-calibrated once at {@link EmbeddingScorer.create} time
   * from {@link CALIBRATION_PAIRS}, so the mapping fits the model's own
   * similarity scale instead of a hardcoded guess.
   */
  similarityRange?: [number, number];
  /** Texts shorter than this many characters are skipped (no spans). */
  minTextLength: number;
  /** Sliding-window length in words. */
  windowSize: number;
  /** Words the window advances between samples. */
  windowStride: number;
}

export interface EmbeddingScorerArgs {
  /** Model name; the provider is inferred from it. Highlight scoring is text-only. */
  model: string;
  config: ProviderConfig;
  scoring: ScoringOptions;
}

const QUERY_CACHE_LIMIT = 32;

/**
 * Worker-side embedding highlight scorer. Scores text with a sliding word
 * window: each window (a run of consecutive words) is embedded as one pooled
 * vector, scored by cosine similarity to the (pooled) query embedding, and the
 * per-window scores are interpolated across characters. Word-aligned windows
 * avoid embedding half-words, are roughly uniform in length so batching keeps
 * the model well-utilized, and need no per-token pooling or token-to-character
 * alignment. Runs entirely in the worker; `score` returns plain JSON, so one
 * RPC round-trip covers a batch.
 */
export class EmbeddingScorer {
  private model!: EmbeddingModel;
  private scoring!: ScoringOptions;
  /** Resolved cosine-similarity range: the caller's pinned value or the calibrated one. */
  private similarityRange!: [number, number];
  /** query text -> pooled, L2-normalized embedding (FIFO-capped). */
  private queryCache = new Map<string, Float32Array>();

  static async create(args: EmbeddingScorerArgs): Promise<EmbeddingScorer> {
    let s = new EmbeddingScorer();
    s.scoring = args.scoring;
    s.model = await loadEmbeddingModelCached(args.model, args.config, "text");
    // Auto-calibrate the similarity range to this model unless the caller pinned
    // one. Cheap (a couple of batched embeds) and folded into the model-load
    // wait, so it adds no perceptible latency beyond first load.
    if (args.scoring.similarityRange != null) {
      s.similarityRange = args.scoring.similarityRange;
    } else {
      s.similarityRange = await calibrateSimilarityRange(
        (inputs) => s.embed(inputs),
        CALIBRATION_PAIRS,
        EMBEDDING_BATCH_SIZE,
      );
      console.debug("Calibrated similarity range", s.similarityRange);
    }
    return s;
  }

  /**
   * Embed strings into one pooled, L2-normalized vector each. The model layer
   * returns raw (un-normalized) vectors regardless of provider, so normalization
   * happens here — every downstream similarity is a dot product that assumes
   * unit-norm inputs (see {@link score} and {@link calibrateSimilarityRange}).
   */
  private async embed(inputs: string[]): Promise<{ data: Float32Array; dim: number }> {
    let { vectors, dimensions } = await this.model.embeddings(inputs);
    l2NormalizeRows(vectors, dimensions);
    return { data: vectors, dim: dimensions };
  }

  private async embedQuery(query: string): Promise<Float32Array> {
    let cached = this.queryCache.get(query);
    if (cached) {
      return cached;
    }
    let { data, dim } = await this.embed([query]);
    // Copy row 0 out of the tensor so it survives the tensor being freed/reused.
    let vec = new Float32Array(data.subarray(0, dim));
    if (this.queryCache.size >= QUERY_CACHE_LIMIT) {
      this.queryCache.delete(this.queryCache.keys().next().value!);
    }
    this.queryCache.set(query, vec);
    return vec;
  }

  async score(texts: string[], query: string): Promise<HighlightSpan[][]> {
    if (query == null || query.trim() === "") {
      return texts.map(() => []);
    }
    let q = await this.embedQuery(query);
    let { minTextLength, windowSize, windowStride } = this.scoring;

    let results: HighlightSpan[][] = texts.map(() => []);

    // Build sliding windows for every text worth highlighting. Short strings
    // (labels, ids, blanks) rarely carry meaning and would waste model time.
    let perTextWindows: WindowRange[][] = texts.map(() => []);
    let windows: Window[] = [];
    for (let i = 0; i < texts.length; i++) {
      if (texts[i].length < minTextLength) {
        continue;
      }
      let ranges = wordWindowRanges(texts[i], windowSize, windowStride);
      perTextWindows[i] = ranges;
      for (let r of ranges) {
        windows.push({ textIndex: i, start: r.start, end: r.end, score: 0 });
      }
    }

    // Embed windows in batches. Windows span ~windowSize words each, so batches
    // are roughly uniform in length and the model stays well-utilized.
    for (let chunk of chunkInputs(windows, EMBEDDING_BATCH_SIZE)) {
      let inputs = chunk.map((w) => texts[w.textIndex].slice(w.start, w.end));
      let { data, dim } = await this.embed(inputs);
      for (let b = 0; b < chunk.length; b++) {
        let off = b * dim;
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += data[off + d] * q[d];
        }
        // Both vectors are L2-normalized, so the dot product is the cosine
        // similarity.
        chunk[b].score = similarityToScore(dot, this.similarityRange);
      }
    }

    // Interpolate each text's window scores across its characters. Windows were
    // pushed grouped by text in order, so a single cursor walks them back out.
    let cursor = 0;
    for (let i = 0; i < texts.length; i++) {
      let ranges = perTextWindows[i];
      if (ranges.length === 0) {
        continue;
      }
      let scores = ranges.map(() => windows[cursor++].score);
      results[i] = interpolateWindowSpans(texts[i].length, ranges, scores);
    }
    return results;
  }

  destroy(): void {
    this.queryCache.clear();
  }
}

interface WindowRange {
  /** Inclusive start character offset of the window. */
  start: number;
  /** Exclusive end character offset of the window. */
  end: number;
}

interface Window extends WindowRange {
  /** Index of the source text this window belongs to. */
  textIndex: number;
  /** Query-similarity score in `[0, 1]`, filled in after embedding. */
  score: number;
}

/**
 * Map a cosine similarity (the dot product of two L2-normalized vectors) to a
 * `[0, 1]` highlight score using a per-model similarity range. Higher similarity
 * (more relevant) yields a higher score. Similarities at or below `min` map to
 * `0`; at or above `max` map to `1`.
 */
export function similarityToScore(similarity: number, range: [number, number]): number {
  const [min, max] = range;
  if (max <= min) {
    return similarity >= max ? 1 : 0;
  }
  const t = (similarity - min) / (max - min);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * L2-normalize each `dim`-length row of `data` in place, so a dot product
 * between two rows is their cosine similarity. Zero-norm rows are left as-is
 * (all-zero), which a dot product reads as similarity 0 rather than NaN.
 */
function l2NormalizeRows(data: Float32Array, dim: number): void {
  if (dim <= 0) {
    return;
  }
  for (let off = 0; off + dim <= data.length; off += dim) {
    let sumSq = 0;
    for (let d = 0; d < dim; d++) {
      const v = data[off + d];
      sumSq += v * v;
    }
    if (sumSq > 0) {
      const inv = 1 / Math.sqrt(sumSq);
      for (let d = 0; d < dim; d++) {
        data[off + d] *= inv;
      }
    }
  }
}

let _segmenter: Intl.Segmenter | null = null;
function wordSegmenter(): Intl.Segmenter {
  if (_segmenter == null) {
    _segmenter = new Intl.Segmenter(undefined, { granularity: "word" });
  }
  return _segmenter;
}

/**
 * Split `text` into word ranges (`[start, end)` character offsets), skipping
 * whitespace and punctuation. Uses `Intl.Segmenter` for locale-aware word
 * boundaries (including languages without spaces).
 */
export function segmentWords(text: string): WindowRange[] {
  const words: WindowRange[] = [];
  if (text.length === 0) {
    return words;
  }
  for (const seg of wordSegmenter().segment(text)) {
    if (seg.isWordLike) {
      words.push({ start: seg.index, end: seg.index + seg.segment.length });
    }
  }
  return words;
}

/**
 * Slide a window of `windowSize` words over `text`, advancing by `windowStride`
 * words each step, and return each window's `[start, end)` character range
 * (from the first word's start to the last word's end). The final window always
 * reaches the last word, so the whole text is covered. Returns `[]` when the
 * text has no words.
 */
function wordWindowRanges(text: string, windowSize: number, windowStride: number): WindowRange[] {
  const words = segmentWords(text);
  const ranges: WindowRange[] = [];
  if (words.length === 0) {
    return ranges;
  }
  const size = Math.max(1, windowSize);
  const stride = Math.max(1, windowStride);
  for (let i = 0; i < words.length; i += stride) {
    const end = Math.min(i + size, words.length);
    ranges.push({ start: words[i].start, end: words[end - 1].end });
    if (end >= words.length) {
      break;
    }
  }
  return ranges;
}

/**
 * Spread per-window `scores` across every character of a `length`-char text by
 * linear interpolation between window centers, emitting one span per character.
 * Characters before the first / after the last window center take that window's
 * score (no extrapolation). `windows` and `scores` are aligned by index.
 *
 * The per-character spans are intentionally fine-grained: the highlight action
 * quantizes them into a few buckets and merges consecutive same-bucket runs, so
 * the smooth interpolation becomes a small number of banded ranges.
 */
function interpolateWindowSpans(length: number, windows: WindowRange[], scores: number[]): HighlightSpan[] {
  const n = Math.min(windows.length, scores.length);
  if (length <= 0 || n === 0) {
    return [];
  }
  const centers = windows.slice(0, n).map((w) => (w.start + w.end) / 2);
  const spans: HighlightSpan[] = [];
  let k = 0; // left window for the current character
  for (let i = 0; i < length; i++) {
    const x = i + 0.5; // character center
    while (k < n - 1 && centers[k + 1] <= x) {
      k++;
    }
    let value: number;
    if (x <= centers[0]) {
      value = scores[0];
    } else if (x >= centers[n - 1]) {
      value = scores[n - 1];
    } else {
      const gap = centers[k + 1] - centers[k];
      const t = gap > 0 ? (x - centers[k]) / gap : 0;
      value = scores[k] + (scores[k + 1] - scores[k]) * t;
    }
    spans.push({ start: i, end: i + 1, value });
  }
  return spans;
}

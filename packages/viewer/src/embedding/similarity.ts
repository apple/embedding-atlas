// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { loadEmbeddingModelCached, type EmbeddingModel } from "../inference/embedding.js";
import { type ProviderConfig } from "../inference/provider_config.js";

/** Inputs for {@link FeatureSimilarity.create}; serialized across the worker boundary. */
export interface FeatureSimilarityArgs {
  /** Model name; the provider is inferred from it. Label similarity is text-only. */
  model: string;
  config: ProviderConfig;
  /** The full set of feature labels to index. Queries return neighbors from this set. */
  features: string[];
}

/** One nearest-neighbor result: a feature label and its cosine similarity to the query. */
export interface SimilarFeature {
  feature: string;
  /** Cosine similarity in `[-1, 1]` (dot product of L2-normalized embeddings). */
  similarity: number;
}

/**
 * Worker-side nearest-neighbor index over feature labels. On create it embeds
 * every label once (L2-normalized), then answers top-k queries by scanning the
 * matrix — a single dot product per row, since both vectors are unit-norm the
 * dot product is the cosine similarity. The whole index (potentially a few
 * thousand short strings) lives in the worker; only the small top-k result
 * crosses back to the main thread per query.
 */
export class FeatureSimilarity {
  private features!: string[];
  /** feature label -> row index in `vectors`. */
  private index!: Map<string, number>;
  /** `features.length * dim` L2-normalized embeddings, row-major. */
  private vectors!: Float32Array;
  private dim = 0;

  private constructor() {}

  static async create(args: FeatureSimilarityArgs): Promise<FeatureSimilarity> {
    const s = new FeatureSimilarity();
    s.features = args.features;
    s.index = new Map(args.features.map((f, i) => [f, i]));
    if (args.features.length === 0) {
      s.vectors = new Float32Array(0);
      return s;
    }
    const model: EmbeddingModel = await loadEmbeddingModelCached(args.model, args.config, "text");
    const { vectors, dimensions } = await model.embeddings(args.features);
    if (vectors.length !== args.features.length * dimensions) {
      throw new Error("FeatureSimilarity: embedding shape mismatch");
    }
    l2NormalizeRows(vectors, dimensions);
    s.vectors = vectors;
    s.dim = dimensions;
    return s;
  }

  /**
   * Return the `k` labels most similar to `feature`, most-similar first. The
   * query feature itself is excluded. Returns `[]` when the feature isn't in the
   * index or the index is empty.
   */
  async topK(feature: string, k: number): Promise<SimilarFeature[]> {
    const qi = this.index.get(feature);
    if (qi == null || this.dim === 0 || k <= 0) {
      return [];
    }
    const dim = this.dim;
    const qOff = qi * dim;
    const n = this.features.length;
    const scored: SimilarFeature[] = [];
    for (let i = 0; i < n; i++) {
      if (i === qi) {
        continue;
      }
      const off = i * dim;
      let dot = 0;
      for (let d = 0; d < dim; d++) {
        dot += this.vectors[off + d] * this.vectors[qOff + d];
      }
      scored.push({ feature: this.features[i], similarity: dot });
    }
    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, k);
  }

  destroy(): void {
    this.vectors = new Float32Array(0);
    this.index?.clear();
    this.dim = 0;
  }
}

/**
 * L2-normalize each `dim`-length row of `data` in place, so a dot product
 * between two rows is their cosine similarity. Zero-norm rows are left all-zero,
 * which a dot product reads as similarity 0 rather than NaN.
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

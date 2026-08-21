// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createUMAP, type UMAPOptions } from "@embedding-atlas/umap-wasm";
import { imageToDataUrl } from "@embedding-atlas/utils";

import { loadEmbeddingModelCached, type EmbeddingInput, type EmbeddingModel } from "../inference/embedding.js";
import { type ProviderConfig } from "../inference/provider_config.js";

/** Inputs for {@link EmbeddingProjector.create}; serialized across the worker boundary. */
export interface EmbeddingProjectorArgs {
  model: string;
  config: ProviderConfig;
  modality: "text" | "image";
  /** Total number of inputs the caller will feed across all `batch()` calls. Used to pre-allocate the output buffer. */
  total: number;
}

export class EmbeddingProjector {
  private modality!: "text" | "image";
  private model!: EmbeddingModel;
  private total!: number;

  // Pre-allocated output buffer; `dimension` is learned from the first batch's
  // result, after which it stays fixed. `offset` is the running write position
  // in elements (not items), so `offset / dimension` is the items written so far.
  private vectors: Float32Array | undefined = undefined;
  private dimension: number | undefined = undefined;
  private offset = 0;

  private constructor() {}

  static async create(args: EmbeddingProjectorArgs): Promise<EmbeddingProjector> {
    let e = new EmbeddingProjector();
    e.modality = args.modality;
    e.total = args.total;
    e.model = await loadEmbeddingModelCached(args.model, args.config, args.modality);
    return e;
  }

  async batch(data: any[]): Promise<void> {
    let inputs: EmbeddingInput[];
    if (this.modality === "image") {
      inputs = data.map((x) => ({ image: imageToDataUrl(x) ?? "" }));
    } else {
      inputs = data.map((x) => x?.toString() ?? "");
    }
    let { vectors, dimensions } = await this.model.embeddings(inputs);
    if (vectors.length !== data.length * dimensions) {
      throw new Error("output embedding dimension mismatch");
    }
    if (this.vectors === undefined) {
      this.dimension = dimensions;
      // One up-front allocation; subsequent batches write into slices.
      this.vectors = new Float32Array(this.total * dimensions);
    } else if (dimensions !== this.dimension) {
      throw new Error(`embedding dimension changed mid-run (${this.dimension} → ${dimensions})`);
    }
    if (this.offset + vectors.length > this.vectors.length) {
      throw new Error(`embedding overflowed pre-allocated buffer (got more than total=${this.total} inputs)`);
    }
    this.vectors.set(vectors, this.offset);
    this.offset += vectors.length;
  }

  async umap(options: UMAPOptions = {}, outputDim: number = 2): Promise<Float32Array> {
    if (this.vectors === undefined || this.dimension === undefined) {
      throw new Error("umap() called before any batch()");
    }
    const count = this.offset / this.dimension;
    // `subarray(0, offset)` so UMAP only sees populated rows when the caller
    // fed fewer than `total` inputs (e.g. on early termination).
    const data = this.offset === this.vectors.length ? this.vectors : this.vectors.subarray(0, this.offset);
    let umap = await createUMAP(count, this.dimension, outputDim, data, {
      metric: "cosine",
      ...options,
    });
    await umap.run();
    let result = new Float32Array(umap.embedding);
    umap.destroy();
    return result;
  }

  destroy(): void {
    this.vectors = undefined;
    this.dimension = undefined;
    this.offset = 0;
  }
}

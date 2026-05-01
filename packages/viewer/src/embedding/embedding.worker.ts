// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createNNDescent, createUMAP, type NNDescentResult, type UMAPOptions } from "@embedding-atlas/umap-wasm";
import { createWorkerRuntime, imageToDataUrl, transfer } from "@embedding-atlas/utils";
import { load_image, pipeline } from "@huggingface/transformers";

let { handler, registerClass } = createWorkerRuntime();

onmessage = handler;

interface EmbeddingOptions {
  type: "text" | "image";
  model: string;
}

class Embedder {
  private type: "text" | "image";
  private extractor: any;
  private batches: any[] = [];
  private data: { data: Float32Array; count: number; dimension: number } | undefined = undefined;
  private nnIndex: Promise<NNDescentResult> | undefined = undefined;

  private constructor() {
    this.type = "text";
  }

  static async create(options: EmbeddingOptions): Promise<Embedder> {
    let e = new Embedder();
    e.type = options.type;
    let pipelineOptions: any = { device: "webgpu" };
    if (options.type === "text") {
      e.extractor = await pipeline("feature-extraction", options.model, pipelineOptions);
    } else if (options.type === "image") {
      e.extractor = await pipeline("image-feature-extraction", options.model, pipelineOptions);
    } else {
      throw new Error("invalid data type");
    }
    return e;
  }

  async batch(data: any[]): Promise<void> {
    let inputs: any;
    if (this.type === "text") {
      inputs = data.map((x) => x?.toString() ?? "");
    } else {
      let imgs = data.map((x) => imageToDataUrl(x) ?? "");
      inputs = await Promise.all(imgs.map((x) => load_image(x)));
    }
    let embedding = await this.extractor(inputs, { pooling: "mean", normalize: true });
    if (embedding.dims.length === 3) {
      embedding = embedding.mean(1);
    }
    if (embedding.dims.length !== 2 || embedding.dims[0] !== data.length) {
      throw new Error("output embedding dimension mismatch");
    }
    this.batches.push(embedding);
  }

  private _getData(): { data: Float32Array; count: number; dimension: number } {
    if (this.data) return this.data;
    const count = this.batches.reduce((a: number, b: any) => a + b.dims[0], 0);
    const dim = this.batches[0].dims[1];
    const vectors = new Float32Array(count * dim);
    let offset = 0;
    for (let b of this.batches) {
      let len = b.dims[0] * dim;
      vectors.set(b.data.subarray(0, len), offset);
      offset += len;
    }
    this.data = { data: vectors, count: count, dimension: dim };
    return this.data;
  }

  private async _getNNIndex(): Promise<NNDescentResult> {
    if (this.nnIndex) return this.nnIndex;
    let { data, count, dimension } = this._getData();
    this.nnIndex = createNNDescent(count, dimension, data, {
      metric: "cosine",
      nNeighbors: 15,
    });
    return this.nnIndex;
  }

  async umap(options: UMAPOptions = {}): Promise<Float32Array> {
    let { data, count, dimension } = this._getData();
    let umap = await createUMAP(count, dimension, 2, data, {
      metric: "cosine",
      ...options,
    });
    await umap.run();
    let result = new Float32Array(umap.embedding);
    umap.destroy();
    return result;
  }

  async neighbors(idx: number, k: number): Promise<{ indices: Int32Array; distances: Float32Array }> {
    let index = await this._getNNIndex();
    return index.queryByIndex(idx, k);
  }

  async queryByVector(vector: Float32Array, k: number): Promise<{ indices: Int32Array; distances: Float32Array }> {
    let index = await this._getNNIndex();
    return index.queryByVector(vector, k);
  }

  /** Query neighbors for all indexed points in one call, avoiding per-point worker round-trips. */
  async bulkNeighbors(
    k: number,
  ): Promise<{ allIndices: Int32Array; allDistances: Float32Array; count: number; k: number }> {
    let { count } = this._getData();
    let index = await this._getNNIndex();
    let allIndices = new Int32Array(count * k);
    let allDistances = new Float32Array(count * k);
    // Fill with sentinel values: index -1, distance Infinity
    // so that unfilled slots (when queryByIndex returns < k results)
    // are never mistaken for valid neighbors
    allIndices.fill(-1);
    for (let i = 0; i < count * k; i++) allDistances[i] = Infinity;
    for (let i = 0; i < count; i++) {
      let { indices, distances } = index.queryByIndex(i, k);
      let len = Math.min(indices.length, k);
      allIndices.set(indices.subarray(0, len), i * k);
      allDistances.set(distances.subarray(0, len), i * k);
    }
    return transfer({ allIndices, allDistances, count, k }, [allIndices.buffer, allDistances.buffer]);
  }

  /** Compute exact cosine distances between pairs of indexed points. */
  async exactDistances(sourceIndices: number[], targetIndices: number[]): Promise<Float32Array> {
    let { data, dimension } = this._getData();
    // Returns a flat array: for each source, distances to all targets
    let result = new Float32Array(sourceIndices.length * targetIndices.length);
    for (let si = 0; si < sourceIndices.length; si++) {
      let sOff = sourceIndices[si] * dimension;
      for (let ti = 0; ti < targetIndices.length; ti++) {
        let tOff = targetIndices[ti] * dimension;
        // Cosine distance = 1 - cosine_similarity
        // Since vectors are L2-normalized, cosine_similarity = dot product
        let dot = 0;
        for (let d = 0; d < dimension; d++) {
          dot += data[sOff + d] * data[tOff + d];
        }
        result[si * targetIndices.length + ti] = 1 - dot;
      }
    }
    return transfer(result, [result.buffer]);
  }

  destroy(): void {
    this.batches = [];
    this.data = undefined;
    if (this.nnIndex) {
      this.nnIndex.then((x) => x.destroy());
    }
    this.nnIndex = undefined;
  }
}

export type { Embedder };

registerClass("Embedder", (options: EmbeddingOptions) => Embedder.create(options));

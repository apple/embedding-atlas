// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createUMAP } from "@embedding-atlas/umap-wasm";
import { load_image, pipeline } from "@huggingface/transformers";

import { imageToDataUrl } from "../lib/image_utils";
import { WorkerRPC } from "./worker_helper";

let { handler, register } = WorkerRPC.runtime();

onmessage = handler;

interface EmbeddingOptions {
  type: "text" | "image";
  model: string;
}

interface EmbeddingComputer {
  batch(data: any[]): Promise<void>;
  finalize(): Promise<Float32Array>;
}

let embeddings = new Map<string, EmbeddingComputer>();

function makeEmbeddingComputer(runBatch: (data: any[]) => Promise<any>): EmbeddingComputer {
  let batches: any[] = [];
  return {
    async batch(data) {
      batches.push(await runBatch(data));
    },
    async finalize() {
      let count = batches.reduce((a, b) => a + b.dims[0], 0);
      let input_dim = batches[0].dims[1];
      let output_dim = 2;
      let data = new Float32Array(count * input_dim);
      let offset = 0;
      for (let i = 0; i < batches.length; i++) {
        let length = batches[i].dims[0] * input_dim;
        data.set(batches[i].data.subarray(0, length), offset);
        offset += length;
      }
      let umap = await createUMAP(count, input_dim, output_dim, data, {
        metric: "cosine",
      });
      umap.run();
      let result = new Float32Array(umap.embedding);
      umap.destroy();
      return result;
    },
  };
}

register("embedding.new", async (options: EmbeddingOptions) => {
  let instance = new Date().getTime() + "-" + Math.random();
  let pipelineOptions: any = { device: "webgpu" };
  if (options.type == "text") {
    let extractor = await pipeline("feature-extraction", options.model, pipelineOptions);
    let computer = makeEmbeddingComputer(async (data) => {
      let inputs = data.map((x) => x?.toString() ?? "");
      let embedding = await extractor(inputs, { pooling: "mean", normalize: true });
      if (embedding.dims.length == 3) {
        embedding = embedding.mean(1);
      }
      if (embedding.dims.length != 2 || embedding.dims[0] != data.length) {
        throw new Error("output embedding dimension mismatch");
      }
      return embedding;
    });
    embeddings.set(instance, computer);
    return instance;
  } else if (options.type == "image") {
    let extractor = await pipeline("image-feature-extraction", options.model, pipelineOptions);
    let computer = makeEmbeddingComputer(async (data) => {
      let imgs = data.map((x) => imageToDataUrl(x) ?? "");
      imgs = await Promise.all(imgs.map((x) => load_image(x)));
      let embedding = await extractor(imgs);
      if (embedding.dims.length == 3) {
        embedding = embedding.mean(1);
      }
      if (embedding.dims.length != 2 || embedding.dims[0] != imgs.length) {
        throw new Error("output embedding dimension mismatch");
      }
      return embedding;
    });
    embeddings.set(instance, computer);
    return instance;
  } else {
    throw new Error("invalid data type");
  }
});

register("embedding.batch", async (instance: string, data: any[]) => {
  await embeddings.get(instance)?.batch(data);
});

register("embedding.finalize", async (instance: string) => {
  let obj = embeddings.get(instance);
  if (obj) {
    embeddings.delete(instance);
    return obj.finalize();
  }
});

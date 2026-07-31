// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type * as TransformersJS from "@huggingface/transformers";

import { type ProviderConfig } from "./provider_config.js";
import { inferProvider } from "./resolve.js";
import {
  chunkInputs,
  createAIMDController,
  dispatchPool,
  expBackoff,
  HttpError,
  parseRetryAfter,
  withRetry,
} from "./utils.js";

export type EmbeddingInput =
  | string
  | {
      /** DataURL (or URL) to the image */
      image: string;
    };

export interface EmbeddingsResult {
  vectors: Float32Array;
  dimensions: number;
}

export interface EmbeddingModel {
  embeddings(inputs: EmbeddingInput[]): Promise<EmbeddingsResult>;
  dispose(): Promise<void>;
}

/** Max inputs per transformers.js feature-extraction call; larger batches are chunked. */
const TRANSFORMERS_MAX_BATCH_SIZE_TEXT = 64;
const TRANSFORMERS_MAX_BATCH_SIZE_IMAGE = 16;

/** Max inputs per OpenAI embeddings request; larger batches are chunked. */
const OPENAI_MAX_BATCH_SIZE = 128;

/** Concurrent in-flight requests for API providers; bounded to avoid rate-limit bursts. */
const OPENAI_MAX_CONCURRENCY = 4;

/** Base delay for OpenAI exponential backoff on transient errors. */
const OPENAI_BACKOFF_BASE_MS = 250;

/**
 * Concatenate per-chunk embedding results into a single result, preserving
 * input order. Throws if `parts` is empty (callers should short-circuit a
 * zero-input call before reaching here) or if dimensions disagree across parts.
 */
function concatEmbeddings(parts: EmbeddingsResult[]): EmbeddingsResult {
  if (parts.length === 0) {
    throw new Error("concatEmbeddings: no parts to concatenate");
  }
  const dimensions = parts[0].dimensions;
  let total = 0;
  for (const part of parts) {
    if (part.dimensions !== dimensions) {
      throw new Error(`concatEmbeddings: dimension mismatch (${part.dimensions} vs ${dimensions})`);
    }
    total += part.vectors.length;
  }
  const vectors = new Float32Array(total);
  let offset = 0;
  for (const part of parts) {
    vectors.set(part.vectors, offset);
    offset += part.vectors.length;
  }
  return { vectors, dimensions };
}

/**
 * Compose the chunk → dispatch → concat pipeline. Calls that fit in one chunk
 * skip the pool overhead and run inline.
 *
 * `getConcurrency()` is consulted by `dispatchPool` so a dynamic controller
 * (e.g. AIMD) can shrink the in-flight cap mid-run.
 */
function batchedEmbeddings(
  embedBatch: (inputs: EmbeddingInput[]) => Promise<EmbeddingsResult>,
  options: { maxBatchSize: number; getConcurrency: () => number },
): (inputs: EmbeddingInput[]) => Promise<EmbeddingsResult> {
  const { maxBatchSize, getConcurrency } = options;
  return async (inputs: EmbeddingInput[]): Promise<EmbeddingsResult> => {
    if (inputs.length <= maxBatchSize) {
      return embedBatch(inputs);
    }
    const chunks = chunkInputs(inputs, maxBatchSize);
    const parts = await dispatchPool(chunks, embedBatch, getConcurrency);
    return concatEmbeddings(parts);
  };
}

/**
 * Coalesce concurrent `.embeddings()` calls on one model. The first call after
 * an idle period flushes on the next microtask; while that flush is in flight,
 * any further calls queue and are merged into one combined call as soon as the
 * in-flight flush resolves. Sequential `await` callers (e.g. bulk projection)
 * pay no extra latency — there's nothing to coalesce, so the next call's
 * microtask flushes alone. Concurrent callers (e.g. many highlight elements
 * scoring the same query) fold into one provider call automatically.
 *
 * Disposal: `dispose()` rejects pending callers and no further flushes run; an
 * in-flight flush still resolves its own callers normally. It returns a promise
 * that settles once that in-flight flush is done, so callers can await
 * quiescence before tearing down the underlying extractor.
 */
interface Coalescer {
  embeddings(inputs: EmbeddingInput[]): Promise<EmbeddingsResult>;
  dispose(): Promise<void>;
}

function coalesceEmbeddings(embedFn: (inputs: EmbeddingInput[]) => Promise<EmbeddingsResult>): Coalescer {
  interface Pending {
    inputs: EmbeddingInput[];
    resolve: (r: EmbeddingsResult) => void;
    reject: (e: unknown) => void;
  }

  let queue: Pending[] = [];
  let inFlight = false;
  let disposed = false;
  // The promise of the currently-running flush chain (the `.finally` below, so
  // it never rejects), or null when idle. `dispose()` awaits this so the caller
  // can hold off tearing down the extractor until the in-flight call finishes.
  let currentFlush: Promise<void> | null = null;

  function maybeFlush() {
    if (disposed || inFlight || queue.length === 0) {
      return;
    }
    inFlight = true;
    const batch = queue;
    queue = [];

    const merged: EmbeddingInput[] = [];
    for (const p of batch) {
      for (const x of p.inputs) {
        merged.push(x);
      }
    }

    currentFlush = embedFn(merged)
      .then(
        ({ vectors, dimensions }) => {
          if (vectors.length !== merged.length * dimensions) {
            const err = new Error("coalesceEmbeddings: response shape mismatch");
            for (const p of batch) p.reject(err);
            return;
          }
          let offset = 0;
          for (const p of batch) {
            const slice = vectors.slice(offset, offset + p.inputs.length * dimensions);
            offset += p.inputs.length * dimensions;
            p.resolve({ vectors: slice, dimensions });
          }
        },
        (err) => {
          for (const p of batch) p.reject(err);
        },
      )
      .finally(() => {
        inFlight = false;
        currentFlush = null;
        // Drain anything that piled up while we were running.
        maybeFlush();
      });
  }

  return {
    embeddings(inputs: EmbeddingInput[]): Promise<EmbeddingsResult> {
      if (disposed) {
        return Promise.reject(new Error("coalesceEmbeddings: disposed"));
      }
      return new Promise((resolve, reject) => {
        queue.push({ inputs, resolve, reject });
        // Microtask defer so a synchronous fan-out (multiple calls in the same
        // tick) lands in one merged batch instead of starting one flush per call.
        queueMicrotask(maybeFlush);
      });
    },
    dispose(): Promise<void> {
      disposed = true;
      const pending = queue;
      queue = [];
      const err = new Error("coalesceEmbeddings: disposed");
      for (const p of pending) p.reject(err);
      // Wait for any in-flight flush to finish so the caller doesn't dispose the
      // underlying extractor mid-inference. `currentFlush` never rejects.
      return currentFlush ?? Promise.resolve();
    },
  };
}

/**
 * Wrap an embedding pipeline in the standard coalesce → batch → embed stack.
 * Coalesce sits on the outside so concurrent callers fold into one
 * chunk+dispatch run; chunking still kicks in if the merged inputs exceed
 * `maxBatchSize`.
 */
function buildEmbeddingModel(
  embedBatch: (inputs: EmbeddingInput[]) => Promise<EmbeddingsResult>,
  options: {
    maxBatchSize: number;
    getConcurrency: () => number;
    dispose: () => Promise<void>;
  },
): EmbeddingModel {
  const batched = batchedEmbeddings(embedBatch, {
    maxBatchSize: options.maxBatchSize,
    getConcurrency: options.getConcurrency,
  });
  const coalescer = coalesceEmbeddings(batched);
  return {
    embeddings: (inputs) => coalescer.embeddings(inputs),
    dispose: async () => {
      // Await any in-flight flush before tearing down the extractor, otherwise
      // `options.dispose()` could free the pipeline mid-inference.
      await coalescer.dispose();
      await options.dispose();
    },
  };
}

const cache = new Map<string, Promise<EmbeddingModel>>();

/**
 * Returns a cached `EmbeddingModel` keyed on `(model, config, modality)`.
 * Multiple callers asking for the same model (e.g. an `EmbeddingProjector` and
 * an `EmbeddingScorer` both using `Xenova/all-MiniLM-L6-v2`) share one network
 * fetch and one in-memory pipeline. Failed loads are evicted so the next caller
 * can retry.
 */
export function loadEmbeddingModelCached(
  model: string,
  config: ProviderConfig,
  modality: "text" | "image",
): Promise<EmbeddingModel> {
  const key = JSON.stringify({ model, config, modality });
  let p = cache.get(key);
  if (p == null) {
    p = loadEmbeddingModel(model, config, modality);
    cache.set(key, p);
    p.catch(() => cache.delete(key));
  }
  return p;
}

/**
 * Load an embedding model from a raw model name and the matching provider config.
 * The provider is inferred from the name (see {@link inferProvider}); `config` is
 * the {@link ProviderConfig} for that provider type.
 */
export async function loadEmbeddingModel(
  model: string,
  config: ProviderConfig,
  modality: "text" | "image",
): Promise<EmbeddingModel> {
  switch (inferProvider(model)) {
    case "transformers.js":
      return loadEmbeddingModelTransformers(model, config, modality);
    case "openai":
      return loadEmbeddingModelOpenAI(model, config);
  }
}

async function loadTransformersJsModule(version: string | undefined): Promise<typeof TransformersJS> {
  const DEFAULT_TRANSFORMERS_JS_VERSION = "4.2.0";
  const v = version ?? DEFAULT_TRANSFORMERS_JS_VERSION;
  // Restrict to semver-shaped strings — the value is interpolated into a CDN URL.
  if (!/^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$/.test(v)) {
    throw new Error(`invalid transformers.js version: ${v}`);
  }
  const cdnUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@" + v;
  return await import(/* @vite-ignore */ cdnUrl);
}

async function loadEmbeddingModelTransformers(
  model: string,
  config: ProviderConfig,
  modality: "text" | "image",
): Promise<EmbeddingModel> {
  const { pipeline, load_image } = await loadTransformersJsModule(config.version);

  // The model name may carry a trailing `:dtype` quantization suffix (a
  // transformers.js-only convention; other providers' names don't contain `:`).
  // Strip it off for the pipeline id and pass the dtype through when present;
  // otherwise leave it unset so transformers.js picks its own default.
  const sep = model.lastIndexOf(":");
  const hasDtype = sep > 0 && sep < model.length - 1;
  const baseName = hasDtype ? model.slice(0, sep) : model;

  const pipelineOptions: any = {};
  if (hasDtype) {
    pipelineOptions.dtype = model.slice(sep + 1);
  }
  // Default to WebGPU for in-browser acceleration; we fall back to wasm below if init fails.
  pipelineOptions.device = "webgpu";

  const pipelineType = modality == "image" ? "image-feature-extraction" : "feature-extraction";
  let extractor;
  try {
    extractor = await pipeline(pipelineType, baseName, pipelineOptions);
  } catch (e) {
    // Requested accelerator (typically WebGPU) may be unavailable here — retry on
    // wasm so callers still get a working model, just slower.
    if (pipelineOptions.device !== "wasm") {
      console.warn(`Pipeline device "${pipelineOptions.device}" unavailable, falling back to wasm:`, e);
      pipelineOptions.device = "wasm";
      extractor = await pipeline(pipelineType, baseName, pipelineOptions);
    } else {
      throw e;
    }
  }

  const embedBatch = async (inputs: EmbeddingInput[]): Promise<EmbeddingsResult> => {
    if (modality == "image") {
      inputs = await Promise.all(inputs.map((x: any) => load_image(x.image)));
    }
    let output = await extractor(inputs as any, { pooling: "mean" });
    if (output.dims.length === 3) {
      // Some models emit a [batch, tokens, dim] tensor even with mean pooling
      // requested; collapse the token axis ourselves.
      output = output.mean(1);
    }
    if (output.dims.length !== 2 || output.dims[0] !== inputs.length) {
      throw new Error("output embedding dimension mismatch");
    }
    const dimensions: number = output.dims[1];
    const vectors = new Float32Array(output.data as any);
    return { vectors, dimensions };
  };

  const defaultBatchSize = modality == "image" ? TRANSFORMERS_MAX_BATCH_SIZE_IMAGE : TRANSFORMERS_MAX_BATCH_SIZE_TEXT;
  const maxBatchSize = config.embeddingsMaxBatchSize ?? defaultBatchSize;

  // Single in-process pipeline, so embed chunks one at a time.
  return buildEmbeddingModel(embedBatch, {
    maxBatchSize,
    getConcurrency: () => 1,
    dispose: async () => {
      await extractor.dispose();
    },
  });
}

async function loadEmbeddingModelOpenAI(model: string, config: ProviderConfig): Promise<EmbeddingModel> {
  const baseUrl = (config.endpoint ?? "https://api.openai.com/v1").replace(/\/+$/, "");
  const url = `${baseUrl}/embeddings`;

  // AIMD over in-flight concurrency: halves on 429/503, additively recovers on
  // sustained success. One controller per cached `EmbeddingModel` — different
  // models against the same endpoint adapt independently.
  const aimd = createAIMDController({ initial: OPENAI_MAX_CONCURRENCY, min: 1 });

  const fetchEmbedding = async (inputs: EmbeddingInput[]): Promise<EmbeddingsResult> => {
    const texts: string[] = [];
    for (const input of inputs) {
      if (typeof input !== "string") {
        throw new Error("openai provider does not support image inputs");
      }
      texts.push(input);
    }

    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (config.apiKey) {
      headers["Authorization"] = `Bearer ${config.apiKey}`;
    }

    const response = await fetch(url, {
      method: "POST",
      referrerPolicy: "no-referrer",
      headers,
      body: JSON.stringify({ model: model, input: texts, encoding_format: "float" }),
    });

    if (!response.ok) {
      const body = await response.text();
      const retryAfterMs = parseRetryAfter(response.headers.get("Retry-After"));
      throw new HttpError(
        `openai embeddings request failed (${response.status}): ${body}`,
        response.status,
        retryAfterMs,
      );
    }

    const json = (await response.json()) as { data: { embedding: number[]; index: number }[] };
    if (!Array.isArray(json.data) || json.data.length !== texts.length) {
      throw new Error("openai embeddings response shape mismatch");
    }
    const ordered = [...json.data].sort((a, b) => a.index - b.index);
    const dimensions = ordered[0].embedding.length;
    const vectors = new Float32Array(texts.length * dimensions);
    for (let i = 0; i < ordered.length; i++) {
      const e = ordered[i].embedding;
      if (e.length !== dimensions) {
        throw new Error("openai embeddings response has inconsistent dimensions");
      }
      vectors.set(e, i * dimensions);
    }
    return { vectors, dimensions };
  };

  const embedBatch = async (inputs: EmbeddingInput[]): Promise<EmbeddingsResult> => {
    const result = await withRetry(() => fetchEmbedding(inputs), {
      onError: (err, attempt) => {
        if (err instanceof HttpError && (err.status === 429 || err.status === 503)) {
          aimd.onRateLimited();
          const backoff = expBackoff(attempt, OPENAI_BACKOFF_BASE_MS);
          return { delayMs: Math.max(err.retryAfterMs ?? 0, backoff) };
        }
        return null;
      },
    });
    aimd.onSuccess();
    return result;
  };

  return buildEmbeddingModel(embedBatch, {
    maxBatchSize: config.embeddingsMaxBatchSize ?? OPENAI_MAX_BATCH_SIZE,
    getConcurrency: () => aimd.limit(),
    dispose: async () => {},
  });
}

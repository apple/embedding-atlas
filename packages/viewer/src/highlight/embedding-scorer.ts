// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { get } from "svelte/store";

import { createEmbeddingScorer, type EmbeddingScorerHandle } from "../embedding/index.js";
import { defaultModels, providerConfigs } from "../inference/model_config_store.js";
import { inferProvider } from "../inference/resolve.js";
import type { HighlightScorer } from "./scheduler.js";

/**
 * Options for {@link embeddingHighlightScorer}. When `model` is unset it's
 * sourced from the global `defaultModels.highlight` store. The model name may
 * carry a `:dtype` suffix to override the provider's default quantization.
 *
 * `similarityRange` is the per-model cosine-similarity window mapped onto the
 * `[0, 1]` highlight score. When omitted it is auto-calibrated to the model on
 * first use (see {@link EmbeddingScorer.create}); pass it only to override.
 */
export interface EmbeddingScorerOptions {
  model?: string;
  /** Explicit cosine-similarity window; auto-calibrated per-model when omitted. */
  similarityRange?: [number, number];
  /** Texts shorter than this many characters are skipped. Default 20. */
  minTextLength?: number;
  /** Sliding-window length in words. Default 8. */
  windowSize?: number;
  /** Words the window advances between samples. Default 4. */
  windowStride?: number;
}

const SCORING_DEFAULTS = {
  minTextLength: 20,
  windowSize: 8,
  windowStride: 4,
};

/**
 * Build a {@link HighlightScorer} that scores text by per-token semantic
 * similarity to the query. Model loading and inference run in the shared
 * embedding worker; the heavy work happens off the main thread.
 *
 * The returned scorer is stateful (it lazily spins up a worker-side model), so
 * create it once and reuse it — the scheduler keys its batches by scorer
 * identity, and a fresh instance per render would defeat batching and reload the
 * model.
 *
 * @example
 * const semanticScorer = embeddingHighlightScorer();
 * <p use:highlight={{ scorer: semanticScorer, query }}>…</p>
 */
export function embeddingHighlightScorer(options?: EmbeddingScorerOptions): HighlightScorer<string> {
  const scoring = { ...SCORING_DEFAULTS, ...options };
  let handle: Promise<EmbeddingScorerHandle> | null = null;
  // Identifies the (model, config) that `handle` was built with, so a later
  // change to the highlight model / provider config rebuilds it instead of
  // being silently ignored until reload.
  let handleKey: string | null = null;

  function getHandle(): Promise<EmbeddingScorerHandle> {
    const name = options?.model ?? get(defaultModels).highlight;
    const config = get(providerConfigs)[inferProvider(name)] ?? {};
    const key = JSON.stringify({ name, config });

    if (handle != null && handleKey === key) {
      return handle;
    }

    // Model/config changed — release the previous worker-side scorer so it
    // doesn't leak (an in-flight `score` already holds its own reference and
    // completes normally), then build a fresh one below.
    if (handle != null) {
      const stale = handle;
      stale.then((scorer) => scorer.destroy()).catch(() => {});
    }

    handleKey = key;
    const p = createEmbeddingScorer({
      model: name,
      config,
      scoring: {
        // Undefined here means "auto-calibrate per-model" in the worker.
        similarityRange: options?.similarityRange,
        minTextLength: scoring.minTextLength,
        windowSize: scoring.windowSize,
        windowStride: scoring.windowStride,
      },
    });
    handle = p;
    // Evict a failed load so the next call retries instead of reusing a
    // permanently-rejected promise (mirrors connect() in embedding/index.ts).
    p.catch(() => {
      if (handle === p) {
        handle = null;
        handleKey = null;
      }
    });
    return p;
  }

  return async (texts, query) => {
    if (query == null || query.trim() === "") {
      return texts.map(() => []);
    }
    // A worker/model failure rejects here; the highlight action catches it and
    // leaves existing highlights in place.
    let scorer = await getHandle();
    return scorer.score(texts, query);
  };
}

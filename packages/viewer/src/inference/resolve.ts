// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type ProviderType } from "./model_config_store.js";

/**
 * Ordered patterns matching a model name to its provider type. First match wins.
 * Most-specific patterns first; fall through to the HuggingFace `org/repo`
 * pattern; final fallback is `FALLBACK_PROVIDER`.
 *
 * To add a new provider (e.g. anthropic, voyage), insert a pattern here and add
 * the corresponding case in `loadEmbeddingModel` (`embedding.ts`).
 */
const PROVIDER_PATTERNS: { match: RegExp; provider: ProviderType }[] = [
  // If contains "/", use transformers.js
  { match: /\//, provider: "transformers.js" },
];

const FALLBACK_PROVIDER: ProviderType = "openai";

/**
 * Pick the provider type that should run a given model name, using only the
 * name itself. Used both for routing and for the "Routes to: …" badge in the
 * UI.
 */
export function inferProvider(name: string): ProviderType {
  for (const { match, provider } of PROVIDER_PATTERNS) {
    if (match.test(name)) {
      return provider;
    }
  }
  return FALLBACK_PROVIDER;
}

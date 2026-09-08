// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * User-configurable settings for one provider type. Fields are unioned across
 * provider types — only the relevant fields apply per provider, the rest are
 * ignored by `loadEmbeddingModel`.
 */
export interface ProviderConfig {
  /** Base URL for API models. Defaults to the provider's URL. */
  endpoint?: string;
  /** API key for API models. Stored in localStorage. */
  apiKey?: string;

  /** Transformers.js library version pin. */
  version?: string;

  /** Maximum batch size for embeddings. */
  embeddingsMaxBatchSize?: number;
}

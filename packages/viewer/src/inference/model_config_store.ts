// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { persistedWritable } from "../utils/persisted_store.js";
import { type ProviderConfig } from "./provider_config.js";

export type ProviderType = "transformers.js" | "openai";

export type ProviderConfigs = { [P in ProviderType]?: ProviderConfig };

export const DEFAULT_PROVIDER_CONFIGS: ProviderConfigs = {
  "transformers.js": {},
  openai: {},
};

/**
 * Per-provider-type configuration (transport / device / etc.). Persisted to
 * localStorage. Never serialised into the URL — these are user-specific.
 */
export const providerConfigs = persistedWritable<ProviderConfigs>(
  "embedding-atlas/provider-configs",
  DEFAULT_PROVIDER_CONFIGS,
);

/**
 * Per-feature default model name. The `embedding` value is read by the file
 * loader as a starting point for the per-file model field (which is then baked
 * into the URL spec); `highlight` is read directly by the highlight scorer.
 *
 * Names may carry a trailing `:dtype` suffix for transformers.js (e.g.
 * `Xenova/all-MiniLM-L6-v2:q4`). The resolver strips and applies it.
 */
export interface DefaultModels {
  embedding: string;
  highlight: string;
}

export const DEFAULT_MODELS: DefaultModels = {
  embedding: "Xenova/all-MiniLM-L6-v2",
  highlight: "Xenova/all-MiniLM-L6-v2",
};

export const defaultModels = persistedWritable<DefaultModels>("embedding-atlas/default-models", DEFAULT_MODELS);

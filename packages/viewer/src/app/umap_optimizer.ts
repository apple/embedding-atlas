// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { UMAPOptions } from "@embedding-atlas/umap-wasm";

export type UMAPOptimizer = NonNullable<UMAPOptions["optimizer"]>;

export const DEFAULT_UMAP_OPTIMIZER: UMAPOptimizer = "adam";

export type UMAPPipelineOptions = Pick<UMAPOptions, "optimizer" | "compatibilityLayout">;

/** Fill in the optimizer of options that predate it. Every optimizer runs on the umap-learn 0.6 pipeline. */
export function withPipeline<T extends UMAPPipelineOptions>(options: T | undefined): T & Required<UMAPPipelineOptions> {
  return { ...(options as T), optimizer: options?.optimizer ?? DEFAULT_UMAP_OPTIMIZER, compatibilityLayout: false };
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/** Visibility of the individual layers of the embedding view. */
export interface EmbeddingLayers {
  points: boolean;
  density: boolean;
  labels: boolean;
}

function layersFromMode(mode: "points" | "density" | undefined): EmbeddingLayers {
  // The legacy "density" mode drew points and labels in addition to the density map.
  return { points: true, density: mode == "density", labels: true };
}

/** Resolve the effective layer visibility from a spec, honoring the deprecated `mode` field. */
export function effectiveLayers(spec: {
  mode?: "points" | "density";
  layers?: Partial<EmbeddingLayers>;
}): EmbeddingLayers {
  return { ...layersFromMode(spec.mode), ...spec.layers };
}

/** Update layer visibility on a spec draft, migrating it away from the deprecated `mode` field. */
export function updateLayers(
  spec: { mode?: "points" | "density"; layers?: Partial<EmbeddingLayers> },
  patch: Partial<EmbeddingLayers>,
): void {
  spec.layers = { ...effectiveLayers(spec), ...patch };
  delete spec.mode;
}

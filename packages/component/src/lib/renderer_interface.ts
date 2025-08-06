// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Point, ViewportState } from "./utils.js";

export type RenderMode = "points" | "density";

export interface EmbeddingRendererProps {
  mode: RenderMode;
  colorScheme: "light" | "dark";

  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  category: Uint8Array<ArrayBuffer> | null;

  categoryCount: number;
  categoryColors: string[] | null;

  viewportX: number;
  viewportY: number;
  viewportScale: number;

  pointSize: number;
  pointAlpha: number;
  pointsAlpha: number;

  densityScaler: number;
  densityBandwidth: number;
  densityQuantizationStep: number;
  densityAlpha: number;
  contoursAlpha: number;

  gamma: number;
  width: number;
  height: number;
}

export interface DensityMap {
  data: Float32Array;
  width: number;
  height: number;
  coordinateAtPixel: (x: number, y: number) => Point;
}

export interface EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;

  /** Set renderer props. Returns true if a render is needed. */
  setProps(newProps: Partial<EmbeddingRendererProps>): boolean;

  /** Render */
  render(): void;

  /** Destroy the renderer and free any resource */
  destroy(): void;

  /** Produce a density map */
  densityMap(width: number, height: number, radius: number, viewportState: ViewportState): Promise<DensityMap>;
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Matrix4, Vector3 } from "./matrix.js";
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

  /** Approximate maximum points to render. null/Infinity = no limit. Default: 4,000,000 */
  downsampleMaxPoints: number | null;
  /** Density weight for downsampling (0-10). Default: 5 */
  downsampleDensityWeight: number;

  // --- 3D point cloud rendering (additive; the 2D path ignores these) --------

  /**
   * Per-point z coordinates. When both `z` and `viewProjection` are non-null the
   * renderer draws a depth-aware 3D point cloud instead of the 2D scatter.
   */
  z: Float32Array<ArrayBuffer> | null;
  /** Column-major view-projection matrix (data space -> clip space) for 3D. */
  viewProjection: Matrix4 | null;
  /** Camera eye position in data space, used for distance fog. */
  cameraEye: Vector3;
  /** Eye-to-target distance, used for perspective-correct point sizing. */
  cameraDistance: number;
  /** Base point size (CSS px at the focal plane) for 3D points. */
  pointSize3D: number;
  /** Distance-fog strength in 3D (0 = no fog). */
  fogDensity: number;
  /**
   * Explicit set of point indices to draw/pick in 3D (e.g. a frustum-culled,
   * strided subset computed for the current camera). When null, all points are
   * drawn. When non-null, exactly these indices are rendered, in order, so the
   * downsampled subset can refine to the zoomed-in region instead of a fixed
   * global stride. Length is bounded by `downsampleMaxPoints`.
   */
  sampleIndices: Uint32Array<ArrayBuffer> | null;
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

  /**
   * Pick the frontmost 3D point under a CSS-pixel location. Returns:
   * - the point's instance index for a hit,
   * - `null` for a confirmed no-hit (cursor over empty space),
   * - `undefined` when the result is indeterminate because the view changed or the
   *   pick failed while it was in flight (stale generation, readback/device error).
   *
   * Callers must treat `undefined` as "leave selection/tooltip unchanged" and only
   * `null` as an intentional empty-space pick. 2D renderers (which use
   * `densityMap`/`querySelection` instead) may return null.
   */
  pick(x: number, y: number): Promise<number | null | undefined>;
}

/**
 * Normalizes a `downsampleMaxPoints` value to an integer cap, or null when
 * downsampling is disabled (null / non-finite / <= 0). Note `Infinity` is covered
 * by the `!Number.isFinite` check.
 */
export function normalizeDownsampleCap(max: number | null): number | null {
  if (max == null || !Number.isFinite(max) || max <= 0) {
    return null;
  }
  return Math.floor(max);
}

/**
 * Number of 3D instances to draw/pick for the given props. With an explicit
 * (frustum-culled) sample-index set the count is its length; otherwise every point
 * is drawn. Clamped to the downsample cap as a backstop so the draw is bounded even
 * if an over-cap sample set ever slips through (the view keeps it within the cap).
 */
export function draw3DCount(
  sampleIndices: Uint32Array | null,
  count: number,
  downsampleMaxPoints: number | null,
): number {
  let n = sampleIndices != null ? sampleIndices.length : count;
  let cap = normalizeDownsampleCap(downsampleMaxPoints);
  return cap != null ? Math.min(n, cap) : n;
}

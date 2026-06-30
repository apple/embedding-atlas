// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import {
  type Matrix4,
  type Vector3,
  matrix4_invert,
  matrix4_look_at,
  matrix4_mul_vec4,
  matrix4_multiply,
  matrix4_perspective,
  vector3_cross,
  vector3_normalize,
} from "./matrix.js";
import type { Point } from "./utils.js";

/**
 * Serializable description of a 3D orbit camera. This is the 3D analog of
 * {@link ViewportState}: the view layer owns it, persists it, and passes it
 * back in so the camera is fully reconstructable.
 *
 * The camera orbits around `target`. `yaw` rotates around the world up axis
 * (+y) and `pitch` raises/lowers the eye. `distance` is the eye-to-target
 * distance and `fov` is the vertical field of view (radians).
 */
export interface Camera3DState {
  target: Vector3;
  distance: number;
  yaw: number;
  pitch: number;
  fov: number;
}

/** Clamp pitch just shy of straight up/down to avoid a degenerate up vector. */
export const MAX_PITCH = 1.5533; // ~89 degrees
const WORLD_UP: Vector3 = [0, 1, 0];

// The default three-quarter starting view shared by fitCamera3D's fitted and
// empty-cloud paths (and by the view layer's fallback camera).
const DEFAULT_FOV = (50 / 180) * Math.PI; // 50° vertical field of view
const DEFAULT_YAW = Math.PI * 0.25;
const DEFAULT_PITCH = Math.PI * 0.18;

export function clampPitch(pitch: number): number {
  return Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch));
}

/** The eye position for a given camera state, in data space. */
export function cameraEye(state: Camera3DState): Vector3 {
  let cp = Math.cos(state.pitch);
  let sp = Math.sin(state.pitch);
  let cy = Math.cos(state.yaw);
  let sy = Math.sin(state.yaw);
  // Offset from target to eye.
  let ox = state.distance * cp * sy;
  let oy = state.distance * sp;
  let oz = state.distance * cp * cy;
  return [state.target[0] + ox, state.target[1] + oy, state.target[2] + oz];
}

/** Orthonormal camera basis (right, up, forward) for a camera state. */
function cameraBasis(state: Camera3DState): { right: Vector3; up: Vector3; forward: Vector3 } {
  let eye = cameraEye(state);
  let forward = vector3_normalize([state.target[0] - eye[0], state.target[1] - eye[1], state.target[2] - eye[2]]);
  let right = vector3_normalize(vector3_cross(forward, WORLD_UP));
  let up = vector3_cross(right, forward);
  return { right, up, forward };
}

/**
 * A 3D orbit camera, the 3D analog of {@link Viewport}. Given a serializable
 * {@link Camera3DState} and the CSS pixel size of the viewport it produces a
 * view-projection matrix for the renderers, projects data points to CSS pixels
 * for overlay markers, and unprojects pixels into world-space rays for picking.
 */
export class Camera3D {
  private state: Camera3DState;
  private width: number;
  private height: number;

  private _viewProjection: Matrix4;
  /** Lazily computed by {@link screenRay}; null until first unprojection. */
  private _inverse: Matrix4 | null = null;
  private _eye: Vector3;
  private _distance: number;

  constructor(state: Camera3DState, width: number, height: number) {
    this.state = state;
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this._eye = cameraEye(state);
    // Near/far derived from distance so precision tracks the scene scale, kept
    // symmetric in log space (near = d/1000, far = d*1000) around the eye-to-target
    // distance the renderers use for perspective-correct point sizing.
    this._distance = Math.max(1e-6, state.distance);
    let proj = matrix4_perspective(state.fov, this.width / this.height, this._distance * 1e-3, this._distance * 1e3);
    let view = matrix4_look_at(this._eye, state.target, WORLD_UP);
    this._viewProjection = matrix4_multiply(proj, view);
  }

  viewProjection(): Matrix4 {
    return this._viewProjection;
  }

  eye(): Vector3 {
    return this._eye;
  }

  /** Eye-to-target distance, used by renderers for perspective-correct point sizing. */
  cameraDistance(): number {
    return this._distance;
  }

  /**
   * Projects a data-space point to CSS pixel coordinates. `depth` is the NDC z
   * in [0, 1] (smaller is nearer). Points behind the camera return `NaN` for
   * `x`/`y` so existing `isFinite` overlay guards hide them.
   */
  project(p: Vector3): { x: number; y: number; depth: number } {
    let [cx, cy, cz, cw] = matrix4_mul_vec4(this._viewProjection, [p[0], p[1], p[2], 1]);
    if (cw <= 1e-9) {
      return { x: NaN, y: NaN, depth: NaN };
    }
    let ndcx = cx / cw;
    let ndcy = cy / cw;
    let ndcz = cz / cw;
    return {
      x: (ndcx * 0.5 + 0.5) * this.width,
      y: (0.5 - ndcy * 0.5) * this.height,
      depth: ndcz,
    };
  }

  /** Returns a closure form of {@link project} for batch use in overlays. */
  pixelLocationFunction(): (x: number, y: number, z: number) => Point {
    let m = this._viewProjection;
    let w = this.width;
    let h = this.height;
    return (x, y, z) => {
      let cx = m[0] * x + m[4] * y + m[8] * z + m[12];
      let cy = m[1] * x + m[5] * y + m[9] * z + m[13];
      let cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 1e-9) {
        return { x: NaN, y: NaN };
      }
      return { x: (cx / cw) * 0.5 * w + 0.5 * w, y: 0.5 * h - (cy / cw) * 0.5 * h };
    };
  }

  /**
   * Unprojects a CSS pixel location into a world-space ray (origin at the eye,
   * normalized direction pointing into the scene).
   */
  screenRay(px: number, py: number): { origin: Vector3; dir: Vector3 } {
    let ndcx = (px / this.width) * 2 - 1;
    let ndcy = 1 - (py / this.height) * 2;
    let inverse = (this._inverse ??= matrix4_invert(this._viewProjection));
    let near = matrix4_mul_vec4(inverse, [ndcx, ndcy, 0, 1]);
    let far = matrix4_mul_vec4(inverse, [ndcx, ndcy, 1, 1]);
    let nw = near[3] !== 0 ? 1 / near[3] : 1;
    let fw = far[3] !== 0 ? 1 / far[3] : 1;
    let nx = near[0] * nw;
    let ny = near[1] * nw;
    let nz = near[2] * nw;
    let dx = far[0] * fw - nx;
    let dy = far[1] * fw - ny;
    let dz = far[2] * fw - nz;
    let dl = Math.hypot(dx, dy, dz) || 1;
    return { origin: this._eye, dir: [dx / dl, dy / dl, dz / dl] };
  }
}

// --- Pure state transitions (used by interaction handlers) -------------------

/** Rotate the camera around its target. */
export function orbit(state: Camera3DState, dYaw: number, dPitch: number): Camera3DState {
  return { ...state, yaw: state.yaw + dYaw, pitch: clampPitch(state.pitch + dPitch) };
}

/** Move the look-at target within the camera's view plane by a pixel delta. */
export function pan(state: Camera3DState, dxPx: number, dyPx: number, width: number, height: number): Camera3DState {
  let { right, up } = cameraBasis(state);
  // World units per CSS pixel at the target plane.
  let worldPerPixel = (2 * state.distance * Math.tan(state.fov / 2)) / Math.max(1, height);
  // Drag should move the scene with the cursor, so the target moves opposite.
  let mr = -dxPx * worldPerPixel;
  let mu = dyPx * worldPerPixel;
  return {
    ...state,
    target: [
      state.target[0] + right[0] * mr + up[0] * mu,
      state.target[1] + right[1] * mr + up[1] * mu,
      state.target[2] + right[2] * mr + up[2] * mu,
    ],
  };
}

/** Scale the eye-to-target distance (>1 zooms out, <1 zooms in). */
export function dolly(state: Camera3DState, factor: number): Camera3DState {
  let distance = state.distance * factor;
  distance = Math.max(1e-4, Math.min(1e12, distance));
  return { ...state, distance };
}

/** Size-independent bounding sphere (centroid + radius) of a 3D point cloud. */
export interface Bounds3D {
  center: Vector3;
  radius: number;
}

/**
 * Computes the centroid and bounding-sphere radius of a 3D point cloud. This is the
 * O(N) part of fitting a camera and is independent of the viewport size, so callers
 * that re-fit on resize should compute this ONCE per dataset and reuse it (see
 * {@link cameraFromBounds3D}).
 */
export function pointsBounds3D(x: ArrayLike<number>, y: ArrayLike<number>, z: ArrayLike<number>): Bounds3D {
  let n = x.length;
  if (n === 0) {
    return { center: [0, 0, 0], radius: 1 };
  }
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < n; i++) {
    cx += x[i];
    cy += y[i];
    cz += z[i];
  }
  cx /= n;
  cy /= n;
  cz /= n;
  let radius = 0;
  for (let i = 0; i < n; i++) {
    let dx = x[i] - cx;
    let dy = y[i] - cy;
    let dz = z[i] - cz;
    let d = dx * dx + dy * dy + dz * dz;
    if (d > radius) {
      radius = d;
    }
  }
  radius = Math.sqrt(radius);
  if (!(radius > 0)) {
    radius = 1;
  }
  return { center: [cx, cy, cz], radius };
}

/**
 * Builds a default three-quarter camera that frames the given bounding sphere. This
 * is O(1), so resizing only needs to recompute the (cheap) aspect-dependent
 * distance from a cached {@link Bounds3D} instead of rescanning every point.
 *
 * `aspect` (width / height) is used so portrait/tall viewports (aspect < 1), whose
 * horizontal fov is narrower than the vertical one, do not crop a horizontally
 * spread cloud on the initial fit.
 */
export function cameraFromBounds3D(bounds: Bounds3D, fov: number = DEFAULT_FOV, aspect: number = 1): Camera3DState {
  // Distance so a sphere of `radius` fits BOTH the vertical and horizontal fov,
  // with a small margin. The horizontal half-angle is atan(aspect * tan(vFov/2)),
  // which is narrower than vertical when aspect < 1, so use whichever needs more
  // distance to keep the sphere fully framed.
  let vHalf = fov / 2;
  let hHalf = Math.atan(Math.max(1e-3, aspect) * Math.tan(vHalf));
  let distance = (bounds.radius / Math.sin(Math.min(vHalf, hHalf))) * 1.1;
  return {
    target: [bounds.center[0], bounds.center[1], bounds.center[2]],
    distance,
    yaw: DEFAULT_YAW,
    pitch: DEFAULT_PITCH,
    fov,
  };
}

/**
 * Computes a default camera that frames all points: the target is the centroid and
 * the distance is chosen so the bounding sphere fits the field of view. Returns a
 * pleasant three-quarter starting view. Convenience wrapper over
 * {@link pointsBounds3D} + {@link cameraFromBounds3D}; callers that re-fit on resize
 * should use those directly so the O(N) scan does not repeat per resize.
 */
export function fitCamera3D(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
  fov: number = DEFAULT_FOV,
  aspect: number = 1,
): Camera3DState {
  if (x.length === 0) {
    return { target: [0, 0, 0], distance: 3, yaw: DEFAULT_YAW, pitch: DEFAULT_PITCH, fov };
  }
  return cameraFromBounds3D(pointsBounds3D(x, y, z), fov, aspect);
}

/**
 * A deterministic global-stride subset of `[0, count)` of length `min(count, cap)`:
 * indices `floor(i * count / n)`. Cheap (O(cap), no full-dataset scan or
 * count-sized allocation) — used for the first 3D render and as the empty-frustum
 * fallback, before/until {@link frustumSampleIndices} refines to the visible set.
 */
export function globalStrideIndices(count: number, cap: number): Uint32Array<ArrayBuffer> {
  let n = Math.max(1, Math.min(count, Math.floor(cap)));
  let out = new Uint32Array(n);
  let stride = count / n;
  for (let i = 0; i < n; i++) {
    out[i] = Math.min(count - 1, Math.floor(i * stride));
  }
  return out;
}

/**
 * Selects up to `cap` point indices to draw/pick in 3D for a given camera. A fixed
 * GLOBAL stride would permanently hide the points it skips — zooming into a dense
 * region could never reveal them. So points outside the (slightly padded) view
 * frustum are culled first, and the in-frustum set is then strided down to the
 * cap; a zoomed-in region therefore selects from ITS own points and refines as the
 * camera moves.
 *
 * Returns null when no downsampling is needed (`count <= cap`) or `cap <= 0`, in
 * which case the caller should draw all points. When the camera frames no data
 * (everything behind/outside the frustum) it falls back to a global stride so the
 * view is not empty and a reframe still has pickable points.
 *
 * Bounded memory: the only allocation is the (<= cap) output — there is no
 * count-sized pool. Two O(n) passes (count in-frustum, then emit a strided subset)
 * keep peak memory at the cap. This still scans every point, so callers should run
 * it OFF the first-render path and throttle it (see EmbeddingViewImpl, which shows
 * a cheap {@link globalStrideIndices} subset first and refines here on settle).
 *
 * `viewProjection` is the column-major data-space -> clip-space matrix; `margin`
 * pads the NDC bounds so points whose disc straddles a screen edge are kept.
 */
export function frustumSampleIndices(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
  viewProjection: Matrix4,
  cap: number,
  margin: number = 1.2,
): Uint32Array<ArrayBuffer> | null {
  let count = x.length;
  let capN = Math.floor(cap);
  if (!(capN > 0) || count <= capN) {
    return null;
  }
  let inFrustum = makeInFrustum(x, y, z, viewProjection, margin);

  // Pass 1: count in-frustum points (no allocation).
  let inCount = 0;
  for (let i = 0; i < count; i++) {
    if (inFrustum(i)) {
      inCount++;
    }
  }
  // Pass 2: emit a strided subset of the in-frustum points (or a global-stride
  // fallback when nothing is in view).
  return emitFrustumSample(inFrustum, count, inCount, capN);
}

/** Builds the per-point in-frustum predicate used by the 3D samplers. A point is
 * visible when it is in front of the camera and inside the padded x/y NDC bounds AND
 * the [0, 1] clip-depth range (points outside near/far are clipped by the renderers,
 * so including them would waste the cap and leave holes). */
function makeInFrustum(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
  m: Matrix4,
  margin: number,
): (i: number) => boolean {
  return (i: number): boolean => {
    let px = x[i];
    let py = y[i];
    let pz = z[i];
    let cw = m[3] * px + m[7] * py + m[11] * pz + m[15];
    if (cw <= 1e-4) {
      return false;
    }
    let inv = 1 / cw;
    let ndcx = (m[0] * px + m[4] * py + m[8] * pz + m[12]) * inv;
    let ndcy = (m[1] * px + m[5] * py + m[9] * pz + m[13]) * inv;
    let ndcz = (m[2] * px + m[6] * py + m[10] * pz + m[14]) * inv;
    return ndcx >= -margin && ndcx <= margin && ndcy >= -margin && ndcy <= margin && ndcz >= 0 && ndcz <= 1;
  };
}

/** Pass 2 of frustum sampling, given the pass-1 in-frustum count: returns a strided
 * subset of the in-frustum points (or a global stride when none are in view). Peak
 * allocation is the (<= cap) output. */
function emitFrustumSample(
  inFrustum: (i: number) => boolean,
  count: number,
  inCount: number,
  capN: number,
): Uint32Array<ArrayBuffer> {
  if (inCount === 0) {
    return globalStrideIndices(count, capN);
  }
  if (inCount <= capN) {
    // Keep all in-frustum points, in order.
    let out = new Uint32Array(inCount);
    let k = 0;
    for (let i = 0; i < count && k < inCount; i++) {
      if (inFrustum(i)) {
        out[k++] = i;
      }
    }
    return out;
  }
  // Stride within the in-frustum points, emitting the (floor(k*stride))-th one for k
  // in [0, cap) — equivalent to striding a materialized in-frustum pool, no pool.
  let out = new Uint32Array(capN);
  let stride = inCount / capN;
  let k = 0;
  let s = 0;
  for (let i = 0; i < count && k < capN; i++) {
    if (!inFrustum(i)) {
      continue;
    }
    if (s === Math.floor(k * stride)) {
      out[k++] = i;
    }
    s++;
  }
  // Rounding can leave the final slot(s) unfilled; pad with the last chosen point.
  while (k < capN) {
    out[k] = out[k - 1];
    k++;
  }
  return out;
}

/**
 * Chunked, cooperative form of {@link frustumSampleIndices} for very large clouds:
 * the same complete frustum cull, but the O(N) pass-1 scan is split into `chunkSize`
 * slices with `yieldToHost()` awaited between them, so it never blocks the UI thread,
 * and `shouldCancel()` is polled so a newer camera/data change abandons stale work.
 *
 * Reads the existing coordinate arrays directly — no copy, no worker, no GPU upload —
 * so peak extra memory is just the (<= cap) result, regardless of dataset size.
 *
 * Returns the sample, `null` when no downsampling is needed (`count <= cap`), or
 * `undefined` when it was canceled (caller should keep its current sample).
 */
export async function frustumSampleIndicesChunked(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
  viewProjection: Matrix4,
  cap: number,
  options: {
    margin?: number;
    chunkSize?: number;
    yieldToHost: () => Promise<void>;
    shouldCancel: () => boolean;
  },
): Promise<Uint32Array<ArrayBuffer> | null | undefined> {
  let count = x.length;
  let capN = Math.floor(cap);
  if (!(capN > 0) || count <= capN) {
    return null;
  }
  let margin = options.margin ?? 1.2;
  let chunkSize = Math.max(1, options.chunkSize ?? 200_000);
  let inFrustum = makeInFrustum(x, y, z, viewProjection, margin);

  // Pass 1: count in-frustum points, yielding between chunks.
  let inCount = 0;
  for (let i = 0; i < count; i += chunkSize) {
    let end = Math.min(count, i + chunkSize);
    for (let j = i; j < end; j++) {
      if (inFrustum(j)) {
        inCount++;
      }
    }
    if (options.shouldCancel()) {
      return undefined;
    }
    await options.yieldToHost();
    if (options.shouldCancel()) {
      return undefined;
    }
  }
  if (inCount === 0) {
    return globalStrideIndices(count, capN);
  }

  // Pass 2: emit the in-frustum subset, also chunked (it re-scans all points to find
  // them, so it is O(N) too and must not block). Mirrors emitFrustumSample's logic.
  if (inCount <= capN) {
    let out = new Uint32Array(inCount);
    let k = 0;
    for (let i = 0; i < count && k < inCount; i += chunkSize) {
      let end = Math.min(count, i + chunkSize);
      for (let j = i; j < end && k < inCount; j++) {
        if (inFrustum(j)) {
          out[k++] = j;
        }
      }
      if (options.shouldCancel()) {
        return undefined;
      }
      await options.yieldToHost();
      if (options.shouldCancel()) {
        return undefined;
      }
    }
    return out;
  }
  let out = new Uint32Array(capN);
  let stride = inCount / capN;
  let k = 0;
  let s = 0;
  for (let i = 0; i < count && k < capN; i += chunkSize) {
    let end = Math.min(count, i + chunkSize);
    for (let j = i; j < end && k < capN; j++) {
      if (!inFrustum(j)) {
        continue;
      }
      if (s === Math.floor(k * stride)) {
        out[k++] = j;
      }
      s++;
    }
    if (options.shouldCancel()) {
      return undefined;
    }
    await options.yieldToHost();
    if (options.shouldCancel()) {
      return undefined;
    }
  }
  while (k < capN) {
    out[k] = out[k - 1];
    k++;
  }
  return out;
}

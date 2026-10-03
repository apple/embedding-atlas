// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// Cone-of-view-aware downsampling for the 3D point cloud renderer, CPU-orchestrated and
// idle-scheduled (unlike the 2D WebGPU renderer's synchronous per-frame GPU compute downsample -
// see `../webgpu_renderer/downsample.ts` - which has no CPU/idle-scheduling equivalent and is
// WebGPU-only). Points outside the camera's view frustum are culled; if more than `maxPoints`
// remain, a uniform random sample is taken via reservoir sampling (a single pass, no need to know
// the visible count in advance). Processing is chunked across idle callbacks so a large dataset
// doesn't block hover/pan/zoom, and is cancellable (checked between every chunk) so a downsample
// pass started before the user moves the camera again doesn't finish and clobber newer results.
//
// Deliberately has no dependency on three.js: `frustum` only needs a `containsPoint` method
// (a real `THREE.Frustum` satisfies this structurally, no cast needed), so this module stays a
// pure, independently-testable algorithm and doesn't pull three.js into any bundle that imports
// it - see EmbeddingView3D.svelte's `loadThree()` for why that matters.

export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export interface FrustumLike {
  containsPoint(point: Point3): boolean;
}

export interface PickVisibleIndicesOptions {
  /** Interleaved x,y,z positions, length `count * 3`. */
  positions: Float32Array;
  count: number;
  frustum: FrustumLike;
  maxPoints: number;
  /** Checked between chunks; when true, processing stops without calling `onComplete`. */
  isStale: () => boolean;
  onComplete: (indices: Uint32Array) => void;
}

const CHUNK_SIZE = 20000;

/** A `requestIdleCallback` polyfill for engines that don't have it (e.g. Safari): falls back to
 * a short `setTimeout`, always reporting a small fixed time budget. */
const requestIdle: (cb: (deadline: { timeRemaining: () => number }) => void) => void =
  typeof requestIdleCallback === "function"
    ? requestIdleCallback
    : (cb) => setTimeout(() => cb({ timeRemaining: () => 8 }), 0);

export function pickVisibleIndices(options: PickVisibleIndicesOptions): void {
  const { positions, count, frustum, maxPoints, isStale, onComplete } = options;

  let reservoir = new Uint32Array(Math.min(maxPoints, count));
  let reservoirLength = 0;
  let visibleSeen = 0;
  let i = 0;
  let point: Point3 = { x: 0, y: 0, z: 0 };

  function processChunk(deadline: { timeRemaining: () => number }) {
    if (isStale()) return;

    let end = Math.min(count, i + CHUNK_SIZE);
    // Also bail out of the chunk early (not just at the boundary) if we've burned the idle
    // budget, so a single slice can't itself cause a long frame stall.
    let sliceEnd = end;
    while (i < sliceEnd) {
      point.x = positions[i * 3];
      point.y = positions[i * 3 + 1];
      point.z = positions[i * 3 + 2];
      if (frustum.containsPoint(point)) {
        if (reservoirLength < maxPoints) {
          reservoir[reservoirLength++] = i;
        } else {
          // Reservoir sampling (Algorithm R): keeps a uniform random sample of the visible
          // points without needing to know the total visible count up front.
          let j = Math.floor(Math.random() * (visibleSeen + 1));
          if (j < maxPoints) {
            reservoir[j] = i;
          }
        }
        visibleSeen++;
      }
      i++;
      if (deadline.timeRemaining() <= 0 && i < sliceEnd) {
        break;
      }
    }

    if (isStale()) return;

    if (i >= count) {
      onComplete(reservoir.subarray(0, reservoirLength).slice() as Uint32Array);
    } else {
      requestIdle(processChunk);
    }
  }

  requestIdle(processChunk);
}

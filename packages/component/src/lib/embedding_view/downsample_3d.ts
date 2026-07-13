// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// Cone-of-view-aware downsampling for the 3D point cloud renderer, CPU-orchestrated and
// idle-scheduled (unlike the 2D WebGPU renderer's synchronous per-frame GPU compute downsample -
// see `../webgpu_renderer/downsample.ts` - which has no CPU/idle-scheduling equivalent and is
// WebGPU-only). Points outside the camera's view frustum are culled; if more than `maxPoints`
// remain, a uniform random sample is taken via reservoir sampling (a single pass, no need to know
// the visible count in advance). Processing is chunked across idle callbacks so a large dataset
// doesn't block hover/pan/zoom, and is cancellable (checked between every chunk) so a downsample
// pass started before the user moves the camera again doesn't finish and clobber newer results.

import * as THREE from "three";

export interface PickVisibleIndicesOptions {
  /** Interleaved x,y,z positions, length `count * 3`. */
  positions: Float32Array;
  count: number;
  frustum: THREE.Frustum;
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
  let point = new THREE.Vector3();

  function processChunk(deadline: { timeRemaining: () => number }) {
    if (isStale()) return;

    let end = Math.min(count, i + CHUNK_SIZE);
    // Also bail out of the chunk early (not just at the boundary) if we've burned the idle
    // budget, so a single slice can't itself cause a long frame stall.
    let sliceEnd = end;
    while (i < sliceEnd) {
      point.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
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

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import {
  Camera3D,
  cameraEye,
  clampPitch,
  dolly,
  fitCamera3D,
  frustumSampleIndices,
  frustumSampleIndicesChunked,
  globalStrideIndices,
  MAX_PITCH,
  orbit,
  pan,
  type Camera3DState,
} from "../src/lib/camera3d.js";

function baseState(): Camera3DState {
  return { target: [0, 0, 0], distance: 10, yaw: 0, pitch: 0, fov: (50 / 180) * Math.PI };
}

describe("orbit", () => {
  it("adds to yaw and pitch", () => {
    let s = orbit(baseState(), 0.3, 0.2);
    expect(s.yaw).toBeCloseTo(0.3, 6);
    expect(s.pitch).toBeCloseTo(0.2, 6);
  });

  it("clamps pitch to just under +/- 90 degrees", () => {
    let up = orbit(baseState(), 0, 100);
    expect(up.pitch).toBeCloseTo(MAX_PITCH, 6);
    expect(up.pitch).toBeLessThan(Math.PI / 2);
    let down = orbit(baseState(), 0, -100);
    expect(down.pitch).toBeCloseTo(-MAX_PITCH, 6);
  });

  it("does not mutate the input state", () => {
    let s = baseState();
    orbit(s, 1, 1);
    expect(s.yaw).toBe(0);
    expect(s.pitch).toBe(0);
  });
});

describe("clampPitch", () => {
  it("limits the range", () => {
    expect(clampPitch(10)).toBeCloseTo(MAX_PITCH, 6);
    expect(clampPitch(-10)).toBeCloseTo(-MAX_PITCH, 6);
    expect(clampPitch(0.1)).toBeCloseTo(0.1, 6);
  });
});

describe("dolly", () => {
  it("scales distance", () => {
    expect(dolly(baseState(), 2).distance).toBeCloseTo(20, 6);
    expect(dolly(baseState(), 0.5).distance).toBeCloseTo(5, 6);
  });

  it("keeps distance positive and finite", () => {
    let s = dolly(baseState(), 0);
    expect(s.distance).toBeGreaterThan(0);
    expect(Number.isFinite(s.distance)).toBe(true);
  });
});

describe("pan", () => {
  it("moves the target within the view plane", () => {
    let s = baseState();
    let moved = pan(s, 50, 0, 800, 800);
    // Dragging horizontally must change the target (in world space).
    let dx = moved.target[0] - s.target[0];
    let dy = moved.target[1] - s.target[1];
    let dz = moved.target[2] - s.target[2];
    expect(Math.hypot(dx, dy, dz)).toBeGreaterThan(0);
  });

  it("scales movement with distance", () => {
    let near = pan({ ...baseState(), distance: 5 }, 100, 0, 800, 800);
    let far = pan({ ...baseState(), distance: 50 }, 100, 0, 800, 800);
    let nearShift = Math.hypot(...near.target);
    let farShift = Math.hypot(...far.target);
    expect(farShift).toBeGreaterThan(nearShift);
  });
});

describe("cameraEye", () => {
  it("sits at `distance` from the target", () => {
    let s: Camera3DState = { target: [1, 2, 3], distance: 7, yaw: 0.5, pitch: 0.3, fov: 1 };
    let e = cameraEye(s);
    let d = Math.hypot(e[0] - s.target[0], e[1] - s.target[1], e[2] - s.target[2]);
    expect(d).toBeCloseTo(7, 5);
  });
});

describe("Camera3D.cameraDistance", () => {
  it("reports the eye-to-target distance the renderers use for point sizing", () => {
    for (let distance of [1e-4, 0.05, 1, 12, 1000]) {
      let cam = new Camera3D({ ...baseState(), distance }, 800, 600);
      expect(cam.cameraDistance()).toBeCloseTo(distance, 6);
    }
  });
});

describe("Camera3D.project <-> screenRay", () => {
  it("projects the target to the screen center", () => {
    let cam = new Camera3D(baseState(), 800, 600);
    let p = cam.project([0, 0, 0]);
    expect(p.x).toBeCloseTo(400, 3);
    expect(p.y).toBeCloseTo(300, 3);
    expect(p.depth).toBeGreaterThanOrEqual(0);
    expect(p.depth).toBeLessThanOrEqual(1);
  });

  it("returns NaN for points behind the camera", () => {
    let cam = new Camera3D(baseState(), 800, 600);
    // Place a point far behind the eye (eye is at +z, looking toward -z origin).
    let p = cam.project([0, 0, 1000]);
    expect(Number.isNaN(p.x)).toBe(true);
    expect(Number.isNaN(p.y)).toBe(true);
  });

  it("round-trips a data point through project then screenRay", () => {
    let cam = new Camera3D({ target: [0, 0, 0], distance: 12, yaw: 0.6, pitch: 0.35, fov: 1 }, 1024, 768);
    let point: [number, number, number] = [1.5, -2, 0.5];
    let screen = cam.project(point);
    let ray = cam.screenRay(screen.x, screen.y);
    // The point must lie on the ray from the eye through that pixel.
    let toPoint: [number, number, number] = [
      point[0] - ray.origin[0],
      point[1] - ray.origin[1],
      point[2] - ray.origin[2],
    ];
    let len = Math.hypot(...toPoint);
    let dot = (toPoint[0] * ray.dir[0] + toPoint[1] * ray.dir[1] + toPoint[2] * ray.dir[2]) / len;
    expect(dot).toBeCloseTo(1, 4); // direction aligned with the ray
  });
});

describe("fitCamera3D", () => {
  it("centers the target on the data centroid", () => {
    let x = new Float32Array([0, 2, 4]);
    let y = new Float32Array([0, 0, 0]);
    let z = new Float32Array([-1, 0, 1]);
    let cam = fitCamera3D(x, y, z);
    expect(cam.target[0]).toBeCloseTo(2, 5);
    expect(cam.target[1]).toBeCloseTo(0, 5);
    expect(cam.target[2]).toBeCloseTo(0, 5);
  });

  it("frames all points inside the NDC cube", () => {
    // Three Gaussian-ish blobs.
    let n = 300;
    let x = new Float32Array(n);
    let y = new Float32Array(n);
    let z = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let c = i % 3;
      x[i] = (c - 1) * 5 + Math.sin(i) * 0.5;
      y[i] = Math.cos(i * 1.3) * 2;
      z[i] = Math.sin(i * 0.7) * 2 + (c - 1) * 3;
    }
    let state = fitCamera3D(x, y, z);
    let cam = new Camera3D(state, 800, 800);
    for (let i = 0; i < n; i++) {
      let p = cam.project([x[i], y[i], z[i]]);
      expect(Number.isNaN(p.x)).toBe(false);
      // Projected pixels should fall within the viewport (with a small margin).
      expect(p.x).toBeGreaterThanOrEqual(-40);
      expect(p.x).toBeLessThanOrEqual(840);
      expect(p.y).toBeGreaterThanOrEqual(-40);
      expect(p.y).toBeLessThanOrEqual(840);
      expect(p.depth).toBeGreaterThanOrEqual(0);
      expect(p.depth).toBeLessThanOrEqual(1);
    }
  });
});

describe("globalStrideIndices", () => {
  it("returns a strided subset of length min(count, cap) with in-range indices", () => {
    let idx = globalStrideIndices(1000, 50);
    expect(idx.length).toBe(50);
    expect(idx[0]).toBe(0);
    for (let k = 0; k < idx.length; k++) {
      expect(idx[k]).toBeGreaterThanOrEqual(0);
      expect(idx[k]).toBeLessThan(1000);
      if (k > 0) {
        expect(idx[k]).toBeGreaterThan(idx[k - 1]); // strictly increasing
      }
    }
  });

  it("never exceeds count", () => {
    let idx = globalStrideIndices(10, 50);
    expect(idx.length).toBe(10);
    expect(Math.max(...idx)).toBeLessThan(10);
  });
});

describe("frustumSampleIndices", () => {
  // Three well-separated blobs along x, like the 3D demo dataset. Points are
  // interleaved by index (i % 3) so a global stride spreads evenly across blobs.
  function blobs(perBlob: number) {
    let n = perBlob * 3;
    let x = new Float32Array(n);
    let y = new Float32Array(n);
    let z = new Float32Array(n);
    let centers = [-6, 0, 6];
    for (let i = 0; i < n; i++) {
      let c = i % 3;
      x[i] = centers[c] + Math.sin(i * 12.9898) * 0.4;
      y[i] = Math.cos(i * 4.1414) * 0.4;
      z[i] = Math.sin(i * 7.233) * 0.4;
    }
    return { x, y, z, n, centers };
  }

  const FOV = (50 / 180) * Math.PI;

  it("returns null when no downsampling is needed", () => {
    let { x, y, z } = blobs(10); // 30 points
    let vp = new Camera3D(fitCamera3D(x, y, z), 800, 800).viewProjection();
    expect(frustumSampleIndices(x, y, z, vp, 30)).toBeNull(); // count == cap
    expect(frustumSampleIndices(x, y, z, vp, 100)).toBeNull(); // count < cap
    expect(frustumSampleIndices(x, y, z, vp, 0)).toBeNull(); // disabled
  });

  it("caps the result length and returns valid in-range indices", () => {
    let { x, y, z, n } = blobs(100); // 300 points
    let vp = new Camera3D(fitCamera3D(x, y, z), 800, 800).viewProjection();
    let cap = 50;
    let idx = frustumSampleIndices(x, y, z, vp, cap)!;
    expect(idx).not.toBeNull();
    expect(idx.length).toBe(cap);
    for (let v of idx) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(n);
    }
  });

  it("refines the selection to the zoomed-in region (previously hidden points become reachable)", () => {
    let { x, y, z, centers } = blobs(100); // 300 points
    let cap = 30;
    let inRight = (idx: Uint32Array) => [...idx].filter((i) => Math.abs(x[i] - centers[2]) < 2).length / idx.length;

    // Full-frame view over all three blobs: a global stride spreads across them.
    let full = new Camera3D(fitCamera3D(x, y, z), 800, 800).viewProjection();
    let fullIdx = frustumSampleIndices(x, y, z, full, cap)!;
    let fullFrac = inRight(fullIdx);

    // Zoom onto the right-hand blob so the others fall outside the frustum.
    let zoomState: Camera3DState = { target: [centers[2], 0, 0], distance: 2.2, yaw: 0, pitch: 0, fov: FOV };
    let zoom = new Camera3D(zoomState, 800, 800).viewProjection();
    let zoomIdx = frustumSampleIndices(x, y, z, zoom, cap)!;
    let zoomFrac = inRight(zoomIdx);

    // The zoomed-in selection is dominated by the right blob, far more than the
    // global stride would be — this is the data-integrity fix.
    expect(zoomFrac).toBeGreaterThan(fullFrac);
    expect(zoomFrac).toBeGreaterThan(0.9);

    // It surfaces right-blob points the full-frame stride never included: points
    // omitted by the global subset are now drawn (and therefore pickable).
    let fullSet = new Set([...fullIdx]);
    let newlyVisible = [...zoomIdx].filter((i) => !fullSet.has(i) && Math.abs(x[i] - centers[2]) < 2);
    expect(newlyVisible.length).toBeGreaterThan(0);
  });

  it("excludes points outside / behind the frustum", () => {
    let { x, y, z, centers } = blobs(100);
    let zoomState: Camera3DState = { target: [centers[2], 0, 0], distance: 2.2, yaw: 0, pitch: 0, fov: FOV };
    let cam = new Camera3D(zoomState, 800, 800);
    let idx = frustumSampleIndices(x, y, z, cam.viewProjection(), 30)!;
    for (let i of idx) {
      // Every selected point projects to a finite (in-front-of-camera) pixel.
      let p = cam.project([x[i], y[i], z[i]]);
      expect(Number.isNaN(p.x)).toBe(false);
      // And it belongs to the targeted blob, not the culled ones.
      expect(Math.abs(x[i] - centers[2])).toBeLessThan(2);
    }
  });

  it("culls points outside the near/far clip depth", () => {
    // Right blob (visible) plus points placed far beyond the far plane along the
    // camera's view axis: they land inside the x/y NDC bounds but at depth > 1, so
    // the renderer would clip them. They must not consume the cap or hide the blob.
    let { x, y, z, centers } = blobs(100); // 300 points; blobs at -6, 0, 6
    let zoomState: Camera3DState = { target: [centers[2], 0, 0], distance: 2.2, yaw: 0, pitch: 0, fov: FOV };
    let cam = new Camera3D(zoomState, 800, 800);
    let eye = cam.eye();

    let extra = 50;
    let base = x.length;
    let n = base + extra;
    let X = new Float32Array(n);
    let Y = new Float32Array(n);
    let Z = new Float32Array(n);
    X.set(x);
    Y.set(y);
    Z.set(z);
    for (let k = 0; k < extra; k++) {
      // On the view axis, 5000 units in front of the eye (far plane ~ distance*1e3).
      X[base + k] = centers[2];
      Y[base + k] = 0;
      Z[base + k] = eye[2] - 5000;
    }

    // Sanity: a far point is in x/y bounds but beyond the far clip plane.
    let far = cam.project([centers[2], 0, eye[2] - 5000]);
    expect(far.depth).toBeGreaterThan(1);

    let idx = frustumSampleIndices(X, Y, Z, cam.viewProjection(), 30)!;
    // None of the appended far points (indices >= base) are selected.
    for (let v of idx) {
      expect(v).toBeLessThan(base);
    }
  });

  it("falls back to a non-empty global stride when the camera frames no data", () => {
    let { x, y, z, n } = blobs(100);
    let away: Camera3DState = { target: [1000, 1000, 1000], distance: 1, yaw: 0, pitch: 0, fov: FOV };
    let vp = new Camera3D(away, 800, 800).viewProjection();
    let cap = 30;
    let idx = frustumSampleIndices(x, y, z, vp, cap)!;
    expect(idx).not.toBeNull();
    expect(idx.length).toBe(cap);
    for (let v of idx) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(n);
    }
  });
});

describe("frustumSampleIndicesChunked", () => {
  function blobs(perBlob: number) {
    let n = perBlob * 3;
    let x = new Float32Array(n);
    let y = new Float32Array(n);
    let z = new Float32Array(n);
    let centers = [-6, 0, 6];
    for (let i = 0; i < n; i++) {
      let c = i % 3;
      x[i] = centers[c] + Math.sin(i * 12.9898) * 0.4;
      y[i] = Math.cos(i * 4.1414) * 0.4;
      z[i] = Math.sin(i * 7.233) * 0.4;
    }
    return { x, y, z, n, centers };
  }
  const FOV = (50 / 180) * Math.PI;
  let noYield = { yieldToHost: () => Promise.resolve(), shouldCancel: () => false, chunkSize: 7 };

  it("produces exactly the same result as the synchronous sampler across chunk boundaries", async () => {
    let { x, y, z, centers } = blobs(100); // 300 points
    for (let state of [
      new Camera3D(fitCamera3D(x, y, z), 800, 800), // full frame (strided)
      new Camera3D({ target: [centers[2], 0, 0], distance: 2.2, yaw: 0, pitch: 0, fov: FOV }, 800, 800), // zoomed
      new Camera3D({ target: [1000, 1000, 1000], distance: 1, yaw: 0, pitch: 0, fov: FOV }, 800, 800), // away
    ]) {
      let vp = state.viewProjection();
      for (let cap of [10, 30, 250]) {
        let sync = frustumSampleIndices(x, y, z, vp, cap);
        let chunked = await frustumSampleIndicesChunked(x, y, z, vp, cap, noYield);
        expect(Array.from(chunked!)).toEqual(Array.from(sync!));
      }
    }
  });

  it("returns null when no downsampling is needed", async () => {
    let { x, y, z } = blobs(10); // 30 points
    let vp = new Camera3D(fitCamera3D(x, y, z), 800, 800).viewProjection();
    expect(await frustumSampleIndicesChunked(x, y, z, vp, 30, noYield)).toBeNull();
  });

  it("returns undefined (and stops early) when canceled", async () => {
    let { x, y, z } = blobs(100);
    let vp = new Camera3D(fitCamera3D(x, y, z), 800, 800).viewProjection();
    let yields = 0;
    let result = await frustumSampleIndicesChunked(x, y, z, vp, 30, {
      chunkSize: 7,
      yieldToHost: () => Promise.resolve(),
      shouldCancel: () => yields++ >= 1, // cancel after the first chunk
    });
    expect(result).toBeUndefined();
  });
});

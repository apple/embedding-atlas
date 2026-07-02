// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import {
  matrix4_identity,
  matrix4_invert,
  matrix4_look_at,
  matrix4_mul_vec4,
  matrix4_multiply,
  matrix4_perspective,
  vector3_normalize,
  type Matrix4,
  type Vector3,
} from "../src/lib/matrix.js";

function expectMatrixClose(a: Matrix4, b: Matrix4, eps = 1e-5) {
  for (let i = 0; i < 16; i++) {
    expect(a[i]).toBeCloseTo(b[i], 5);
  }
  void eps;
}

describe("matrix4_identity", () => {
  it("is the multiplicative identity", () => {
    let id = matrix4_identity();
    let m: Matrix4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expectMatrixClose(matrix4_multiply(id, m), m);
    expectMatrixClose(matrix4_multiply(m, id), m);
  });

  it("maps a vector to itself", () => {
    let v = matrix4_mul_vec4(matrix4_identity(), [3, -2, 5, 1]);
    expect(v).toEqual([3, -2, 5, 1]);
  });
});

describe("matrix4_multiply", () => {
  it("is column-major: m * v applies translation in the last column", () => {
    // Column-major translation matrix (translate by (10, 20, 30)).
    let t: Matrix4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1];
    let v = matrix4_mul_vec4(t, [1, 2, 3, 1]);
    expect(v).toEqual([11, 22, 33, 1]);
  });

  it("composes transforms (translate then scale)", () => {
    let scale: Matrix4 = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];
    let translate: Matrix4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
    // scale * translate applied to a point: first translate, then scale.
    let m = matrix4_multiply(scale, translate);
    let v = matrix4_mul_vec4(m, [0, 0, 0, 1]);
    expect(v).toEqual([2, 2, 2, 1]);
  });
});

describe("matrix4_perspective", () => {
  it("maps the near plane to z=0 and the far plane to z=1 (WebGPU convention)", () => {
    let near = 0.5;
    let far = 50;
    let m = matrix4_perspective((60 / 180) * Math.PI, 1.5, near, far);
    // A point on the -z axis at the near plane.
    let atNear = matrix4_mul_vec4(m, [0, 0, -near, 1]);
    expect(atNear[2] / atNear[3]).toBeCloseTo(0, 4);
    let atFar = matrix4_mul_vec4(m, [0, 0, -far, 1]);
    expect(atFar[2] / atFar[3]).toBeCloseTo(1, 4);
  });

  it("perspective w equals the view distance (for size falloff)", () => {
    let m = matrix4_perspective((50 / 180) * Math.PI, 1, 0.1, 100);
    let p = matrix4_mul_vec4(m, [0, 0, -7, 1]);
    expect(p[3]).toBeCloseTo(7, 5);
  });
});

describe("matrix4_look_at", () => {
  it("places the eye at the origin of view space", () => {
    let eye: Vector3 = [3, 4, 5];
    let view = matrix4_look_at(eye, [0, 0, 0], [0, 1, 0]);
    let e = matrix4_mul_vec4(view, [eye[0], eye[1], eye[2], 1]);
    expect(e[0]).toBeCloseTo(0, 5);
    expect(e[1]).toBeCloseTo(0, 5);
    expect(e[2]).toBeCloseTo(0, 5);
  });

  it("places the target in front of the camera (negative z)", () => {
    let view = matrix4_look_at([0, 0, 10], [0, 0, 0], [0, 1, 0]);
    let t = matrix4_mul_vec4(view, [0, 0, 0, 1]);
    expect(t[2]).toBeLessThan(0);
    expect(t[2]).toBeCloseTo(-10, 5);
  });
});

describe("matrix4_invert", () => {
  it("inverts a view-projection so that inv * (m * p) == p", () => {
    let proj = matrix4_perspective((55 / 180) * Math.PI, 1.3, 0.1, 100);
    let view = matrix4_look_at([2, 3, 8], [1, 0, 0], [0, 1, 0]);
    let vp = matrix4_multiply(proj, view);
    let inv = matrix4_invert(vp);
    let p: Vector3 = [0.5, -1.2, 0.3];
    let clip = matrix4_mul_vec4(vp, [p[0], p[1], p[2], 1]);
    let back = matrix4_mul_vec4(inv, clip);
    expect(back[0] / back[3]).toBeCloseTo(p[0], 4);
    expect(back[1] / back[3]).toBeCloseTo(p[1], 4);
    expect(back[2] / back[3]).toBeCloseTo(p[2], 4);
  });

  it("returns identity for a singular matrix", () => {
    let zero: Matrix4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    expectMatrixClose(matrix4_invert(zero), matrix4_identity());
  });
});

describe("degenerate camera math", () => {
  it("vector3_normalize returns a finite vector for a zero-length input", () => {
    let v = vector3_normalize([0, 0, 0]);
    expect(v.every((c) => Number.isFinite(c))).toBe(true);
    expect(v).toEqual([0, 0, 0]);
  });

  it("matrix4_look_at yields a finite matrix when eye == center (degenerate forward)", () => {
    // A fully-zoomed camera or a zero-extent cloud can produce eye == center; the
    // normalize-zero guard must keep the view matrix free of NaN.
    let m = matrix4_look_at([1, 1, 1], [1, 1, 1], [0, 1, 0]);
    expect(m.every((c) => Number.isFinite(c))).toBe(true);
  });
});

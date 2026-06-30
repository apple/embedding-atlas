// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type Matrix3 = [number, number, number, number, number, number, number, number, number];
export type Vector2 = [number, number];
export type Vector3 = [number, number, number];
export type Vector4 = [number, number, number, number];

export function matrix3_zero(): Matrix3 {
  return [0, 0, 0, 0, 0, 0, 0, 0, 0];
}

export function matrix3_identity(): Matrix3 {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

export function matrix3_matrix_mul_matrix(m1: Matrix3, m2: Matrix3): Matrix3 {
  return [
    m1[0] * m2[0] + m1[3] * m2[1] + m1[6] * m2[2],
    m1[1] * m2[0] + m1[4] * m2[1] + m1[7] * m2[2],
    m1[2] * m2[0] + m1[5] * m2[1] + m1[8] * m2[2],
    m1[0] * m2[3] + m1[3] * m2[4] + m1[6] * m2[5],
    m1[1] * m2[3] + m1[4] * m2[4] + m1[7] * m2[5],
    m1[2] * m2[3] + m1[5] * m2[4] + m1[8] * m2[5],
    m1[0] * m2[6] + m1[3] * m2[7] + m1[6] * m2[8],
    m1[1] * m2[6] + m1[4] * m2[7] + m1[7] * m2[8],
    m1[2] * m2[6] + m1[5] * m2[7] + m1[8] * m2[8],
  ];
}

export function matrix3_matrix_mul_vector(m: Matrix3, v: Vector3): Vector3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

export function matrix3_vector_mul_matrix(v: Vector3, m: Matrix3): Vector3 {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

export function matrix3_determinant(m: Matrix3): number {
  return (
    m[0] * m[4] * m[8] -
    m[0] * m[5] * m[7] -
    m[1] * m[3] * m[8] +
    m[1] * m[5] * m[6] +
    m[2] * m[3] * m[7] -
    m[2] * m[4] * m[6]
  );
}

export function matrix3_inverse(m: Matrix3): Matrix3 {
  let det = matrix3_determinant(m);
  return [
    (m[4] * m[8] - m[5] * m[7]) / det,
    (m[2] * m[7] - m[1] * m[8]) / det,
    (m[1] * m[5] - m[2] * m[4]) / det,
    (m[5] * m[6] - m[3] * m[8]) / det,
    (m[0] * m[8] - m[2] * m[6]) / det,
    (m[2] * m[3] - m[0] * m[5]) / det,
    (m[3] * m[7] - m[4] * m[6]) / det,
    (m[1] * m[6] - m[0] * m[7]) / det,
    (m[0] * m[4] - m[1] * m[3]) / det,
  ];
}

/**
 * A 4x4 matrix stored in **column-major** order (16 numbers).
 *
 * This matches both WGSL `mat4x4<f32>` and WebGL `uniformMatrix4fv(..., false, m)`,
 * so the same buffer/array can be uploaded to either backend without transposing.
 * Element at column `c`, row `r` is `m[c * 4 + r]`.
 */
export type Matrix4 = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export function matrix4_identity(): Matrix4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/** Returns the matrix product `a * b` (column-major). */
export function matrix4_multiply(a: Matrix4, b: Matrix4): Matrix4 {
  let result = new Array(16) as unknown as Matrix4;
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      result[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return result;
}

/** Applies `m` to a column vector `v` (`m * v`). */
export function matrix4_mul_vec4(m: Matrix4, v: Vector4): Vector4 {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

/**
 * A right-handed perspective projection (camera looks down -z) that maps the
 * view volume to clip space with **z in [0, 1]** (WebGPU convention).
 *
 * The [0, 1] range is also valid for WebGL2 — visible depths land in the front
 * half of its [-1, 1] NDC range, which the depth test still orders correctly.
 */
export function matrix4_perspective(fovY: number, aspect: number, near: number, far: number): Matrix4 {
  let f = 1.0 / Math.tan(fovY / 2);
  let nf = 1.0 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0];
}

/** Returns the unit-length vector along `v` (or `v` divided by 1 if zero-length). */
export function vector3_normalize(v: Vector3): Vector3 {
  let l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Returns the cross product `a × b`. */
export function vector3_cross(a: Vector3, b: Vector3): Vector3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Right-handed look-at view matrix (column-major). */
export function matrix4_look_at(eye: Vector3, center: Vector3, up: Vector3): Matrix4 {
  let f = vector3_normalize([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
  let s = vector3_normalize(vector3_cross(f, up)); // right
  let u = vector3_cross(s, f); // up
  return [
    s[0],
    u[0],
    -f[0],
    0,
    s[1],
    u[1],
    -f[1],
    0,
    s[2],
    u[2],
    -f[2],
    0,
    -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]),
    -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]),
    f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2],
    1,
  ];
}

/** Full 4x4 inverse. Returns the identity if the matrix is singular. */
export function matrix4_invert(m: Matrix4): Matrix4 {
  let a00 = m[0],
    a01 = m[1],
    a02 = m[2],
    a03 = m[3];
  let a10 = m[4],
    a11 = m[5],
    a12 = m[6],
    a13 = m[7];
  let a20 = m[8],
    a21 = m[9],
    a22 = m[10],
    a23 = m[11];
  let a30 = m[12],
    a31 = m[13],
    a32 = m[14],
    a33 = m[15];

  let b00 = a00 * a11 - a01 * a10;
  let b01 = a00 * a12 - a02 * a10;
  let b02 = a00 * a13 - a03 * a10;
  let b03 = a01 * a12 - a02 * a11;
  let b04 = a01 * a13 - a03 * a11;
  let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30;
  let b07 = a20 * a32 - a22 * a30;
  let b08 = a20 * a33 - a23 * a30;
  let b09 = a21 * a32 - a22 * a31;
  let b10 = a21 * a33 - a23 * a31;
  let b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0) {
    return matrix4_identity();
  }
  let d = 1.0 / det;
  return [
    (a11 * b11 - a12 * b10 + a13 * b09) * d,
    (a02 * b10 - a01 * b11 - a03 * b09) * d,
    (a31 * b05 - a32 * b04 + a33 * b03) * d,
    (a22 * b04 - a21 * b05 - a23 * b03) * d,
    (a12 * b08 - a10 * b11 - a13 * b07) * d,
    (a00 * b11 - a02 * b08 + a03 * b07) * d,
    (a32 * b02 - a30 * b05 - a33 * b01) * d,
    (a20 * b05 - a22 * b02 + a23 * b01) * d,
    (a10 * b10 - a11 * b08 + a13 * b06) * d,
    (a01 * b08 - a00 * b10 - a03 * b06) * d,
    (a30 * b04 - a31 * b02 + a33 * b00) * d,
    (a21 * b02 - a20 * b04 - a23 * b00) * d,
    (a11 * b07 - a10 * b09 - a12 * b06) * d,
    (a00 * b09 - a01 * b07 + a02 * b06) * d,
    (a31 * b01 - a30 * b03 - a32 * b00) * d,
    (a20 * b03 - a21 * b01 + a22 * b00) * d,
  ];
}

/** Transforms a 3D point by `m` (treating it as `(x, y, z, 1)`) with perspective divide. */
export function matrix4_transform_point(m: Matrix4, p: Vector3): Vector3 {
  let [x, y, z, w] = matrix4_mul_vec4(m, [p[0], p[1], p[2], 1]);
  let iw = w !== 0 ? 1 / w : 1;
  return [x * iw, y * iw, z * iw];
}

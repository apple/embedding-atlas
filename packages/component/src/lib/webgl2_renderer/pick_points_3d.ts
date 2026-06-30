// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Dataflow, Node } from "../dataflow.js";
import type { Matrix4 } from "../matrix.js";
import { webglProgram } from "./utils.js";

const VERTEX = `#version 300 es
  precision highp float;
  uniform mat4 u_matrix;
  uniform float u_point_size;
  uniform float u_camera_distance;
  layout(location=0) in float x;
  layout(location=1) in float y;
  layout(location=2) in float z;
  flat out uint v_id;
  out float v_psize;
  void main() {
    vec4 clip = u_matrix * vec4(x, y, z, 1.0);
    // Reject points at/behind the camera (parity with the draw shader).
    if (clip.w <= 1e-4) {
      gl_Position = vec4(-2.0, -2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      v_psize = 0.0;
      v_id = 0u;
      return;
    }
    gl_Position = clip;
    float w = max(clip.w, 1e-6);
    // Match the draw shader's sprite size (2x radius) so the pickable disc
    // exactly covers the visible disc.
    float psize = clamp(2.0 * u_point_size * u_camera_distance / w, 1.0, 1024.0);
    gl_PointSize = psize;
    v_psize = psize;
    v_id = uint(gl_VertexID + 1);
  }
`;

// Encodes the (1-based) instance index into RGBA8 so it survives readback
// without integer-texture support. The round mask matches the visible disc.
// `precision highp int` is REQUIRED: fragment int/uint default to mediump (16-bit),
// which would drop the high bits of indices above ~65k before the RGBA encoding,
// resolving the wrong row for large datasets (notably on mobile GPUs).
const FRAGMENT = `#version 300 es
  precision highp float;
  precision highp int;
  flat in uint v_id;
  in float v_psize;
  out vec4 outColor;
  void main() {
    float r = length(gl_PointCoord.xy - vec2(0.5, 0.5)) * v_psize;
    if (v_psize / 2.0 - r <= 0.0) {
      discard;
    }
    uint id = v_id;
    outColor = vec4(
      float(id & 0xFFu) / 255.0,
      float((id >> 8u) & 0xFFu) / 255.0,
      float((id >> 16u) & 0xFFu) / 255.0,
      float((id >> 24u) & 0xFFu) / 255.0
    );
  }
`;

interface PickFramebuffer {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  depth: WebGLRenderbuffer;
  width: number;
  height: number;
}

// A dedicated RGBA8 framebuffer with a depth renderbuffer (the shared
// webglFramebuffer helper has no depth attachment).
function pickFramebuffer(
  state: {
    framebuffer?: WebGLFramebuffer;
    texture?: WebGLTexture;
    depth?: WebGLRenderbuffer;
    cacheKey?: string;
    destroy?: () => void;
  },
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
): PickFramebuffer {
  if (state.framebuffer == null || state.texture == null || state.depth == null) {
    let framebuffer = gl.createFramebuffer()!;
    let texture = gl.createTexture()!;
    let depth = gl.createRenderbuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    state.framebuffer = framebuffer;
    state.texture = texture;
    state.depth = depth;
    state.destroy = () => {
      gl.deleteFramebuffer(framebuffer);
      gl.deleteTexture(texture);
      gl.deleteRenderbuffer(depth);
    };
  }
  let cacheKey = `${width},${height}`;
  if (state.cacheKey != cacheKey) {
    state.cacheKey = cacheKey;
    gl.bindTexture(gl.TEXTURE_2D, state.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, state.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  }
  return { framebuffer: state.framebuffer, texture: state.texture, depth: state.depth, width, height };
}

export type Pick3DCommand = (
  matrix4: Matrix4,
  pointSize: number,
  cameraDistance: number,
  drawCount: number,
  redraw: boolean,
  px: number,
  py: number,
) => number | null;

/**
 * Picks the frontmost 3D point under a render-target pixel location (origin at
 * the top-left, matching pointer coordinates). Returns the instance index or
 * null. `px`/`py` are in the render target's device pixels.
 */
export function pickPoints3DCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  x: Node<WebGLBuffer>,
  y: Node<WebGLBuffer>,
  z: Node<WebGLBuffer>,
  count: Node<number>,
  sampleIndex: Node<WebGLBuffer | null>,
  width: Node<number>,
  height: Node<number>,
): Node<Pick3DCommand> {
  let program = df.statefulDerive([gl, VERTEX, FRAGMENT], webglProgram);
  let fb = df.statefulDerive([gl, width, height], pickFramebuffer);
  return df.derive(
    [gl, program, fb, x, y, z, count, sampleIndex, width, height],
    (gl, program, fb, x, y, z, count, sampleIndex, width, height) =>
      (matrix4, pointSize, cameraDistance, drawCount, redraw, px, py): number | null => {
        let n = Math.min(count, drawCount);
        if (n <= 0 || width <= 0 || height <= 0) {
          return null;
        }
        // Clamp into range so a redraw always renders (keeping the FBO valid) and
        // the readback never samples out of bounds.
        let ix = Math.min(width - 1, Math.max(0, Math.floor(px)));
        let iy = Math.min(height - 1, Math.max(0, Math.floor(py)));

        gl.bindFramebuffer(gl.FRAMEBUFFER, fb.framebuffer);
        gl.viewport(0, 0, width, height);

        // Only re-render the ID buffer when the view changed since the last pick;
        // repeated hovers over a static camera reuse it and just read back a pixel.
        if (redraw) {
          gl.disable(gl.BLEND);
          // IDs are encoded as exact RGBA8 byte values. DITHER (enabled by default)
          // is allowed to perturb stored color bytes, which would corrupt the
          // decoded index and pick the wrong point; turn it off for the ID pass.
          gl.disable(gl.DITHER);
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.LESS);
          gl.clearColor(0, 0, 0, 0);
          gl.clearDepth(1);
          gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

          gl.useProgram(program.program);
          gl.enableVertexAttribArray(0);
          gl.bindBuffer(gl.ARRAY_BUFFER, x);
          gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
          gl.enableVertexAttribArray(1);
          gl.bindBuffer(gl.ARRAY_BUFFER, y);
          gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
          gl.enableVertexAttribArray(2);
          gl.bindBuffer(gl.ARRAY_BUFFER, z);
          gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
          gl.bindBuffer(gl.ARRAY_BUFFER, null);

          gl.uniformMatrix4fv(program.uniforms.u_matrix, false, matrix4);
          gl.uniform1f(program.uniforms.u_point_size, pointSize);
          gl.uniform1f(program.uniforms.u_camera_distance, cameraDistance);

          if (sampleIndex != null) {
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sampleIndex);
            gl.drawElements(gl.POINTS, n, gl.UNSIGNED_INT, 0);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
          } else {
            gl.drawArrays(gl.POINTS, 0, n);
          }

          gl.disableVertexAttribArray(0);
          gl.disableVertexAttribArray(1);
          gl.disableVertexAttribArray(2);
          gl.disable(gl.DEPTH_TEST);
          // Restore DITHER to its default-enabled state for the shared GL context.
          gl.enable(gl.DITHER);
          gl.useProgram(null);
        }

        // readPixels origin is bottom-left; pointer y is top-down.
        let out = new Uint8Array(4);
        gl.readPixels(ix, height - 1 - iy, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        let id = out[0] | (out[1] << 8) | (out[2] << 16) | (out[3] << 24);
        return id === 0 ? null : id - 1;
      },
  );
}

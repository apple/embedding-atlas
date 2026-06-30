// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Dataflow, Node } from "../dataflow.js";
import type { Matrix4, Vector3 } from "../matrix.js";
import { webglProgram } from "./utils.js";

function shaderSource(hasCategory: boolean) {
  // Perspective point size: gl_PointSize is the on-screen diameter; scaling by
  // camera_distance / w makes a point at the focal distance render at point_size
  // and shrinks farther points. The fragment masks a round, anti-aliased disc.
  let vertexCommon = `#version 300 es
    precision highp float;
    uniform mat4 u_matrix;
    uniform float u_point_size;
    uniform float u_camera_distance;
    uniform float u_fog_density;
    uniform vec3 u_eye;

    layout(location=0) in float x;
    layout(location=1) in float y;
    layout(location=2) in float z;
  `;
  let vertexBody = (assignColor: string) => `
    out vec4 v_color;
    out float v_fog;
    out float v_psize;

    void main() {
      vec4 clip = u_matrix * vec4(x, y, z, 1.0);
      // Reject points at/behind the camera (parity with the WebGPU path).
      if (clip.w <= 1e-4) {
        gl_Position = vec4(-2.0, -2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        v_color = vec4(0.0);
        v_fog = 0.0;
        v_psize = 0.0;
        return;
      }
      gl_Position = clip;
      float w = max(clip.w, 1e-6);
      // The fragment masks a disc of radius v_psize/2 inside the sprite, so the
      // sprite must be 2x the desired pixel radius to match the WebGPU quad size.
      float psize = clamp(2.0 * u_point_size * u_camera_distance / w, 1.0, 1024.0);
      gl_PointSize = psize;
      v_psize = psize;
      float d = length(vec3(x, y, z) - u_eye);
      v_fog = clamp((d / max(u_camera_distance, 1e-6) - 0.5) * u_fog_density, 0.0, 0.9);
      ${assignColor}
    }
  `;
  let vertex: string;
  if (hasCategory) {
    vertex =
      vertexCommon +
      `
      uniform vec4 colorScheme[64];
      // Unsigned to match the UNSIGNED_BYTE attribute binding below (WebGL2 requires
      // integer-attribute signedness to match the shader input); also keeps category
      // values >= 128 positive.
      layout(location=3) in uint category;
    ` +
      vertexBody(`
      if (category < 64u) {
        v_color = vec4(colorScheme[category].rgb, 1.0);
      } else {
        v_color = vec4(0.5, 0.5, 0.5, 1.0);
      }
    `);
  } else {
    vertex =
      vertexCommon +
      `
      uniform vec4 colorScheme;
    ` +
      vertexBody(`
      v_color = vec4(colorScheme.rgb, 1.0);
    `);
  }
  let fragment = `#version 300 es
    precision highp float;
    uniform float u_gamma;
    uniform vec3 u_background;
    in vec4 v_color;
    in float v_fog;
    in float v_psize;
    out vec4 outColor;
    void main() {
      float r = length(gl_PointCoord.xy - vec2(0.5, 0.5)) * v_psize;
      float a = max(0.0, min(1.0, v_psize / 2.0 - r));
      if (a <= 0.0) {
        discard;
      }
      vec3 rgb = mix(v_color.rgb, u_background, v_fog);
      rgb = pow(rgb, vec3(1.0 / u_gamma));
      outColor = vec4(rgb, a);
    }
  `;
  return { vertex, fragment };
}

export type PaintPoints3DCommand = (
  matrix4: Matrix4,
  pointSize: number,
  cameraDistance: number,
  drawCount: number,
  fogDensity: number,
  eye: Vector3,
  colors: number[],
  gamma: number,
  background: [number, number, number],
) => void;

/**
 * Draws the data as a depth-tested 3D point cloud into the currently bound
 * framebuffer. The caller is responsible for binding the target, setting the
 * viewport, and clearing color + depth.
 */
export function paintPoints3DCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  x: Node<WebGLBuffer>,
  y: Node<WebGLBuffer>,
  z: Node<WebGLBuffer>,
  category: Node<WebGLBuffer> | null,
  count: Node<number>,
  sampleIndex: Node<WebGLBuffer | null>,
): Node<PaintPoints3DCommand> {
  let hasCategory = category != null;
  let source = shaderSource(hasCategory);
  let program = df.statefulDerive([gl, source.vertex, source.fragment], webglProgram);
  return df.derive(
    [gl, program, x, y, z, category, count, sampleIndex],
    (gl, program, x, y, z, category, count, sampleIndex) =>
      (matrix4, pointSize, cameraDistance, drawCount, fogDensity, eye, colors, gamma, background) => {
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LESS);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

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
        if (category != null) {
          gl.enableVertexAttribArray(3);
          gl.bindBuffer(gl.ARRAY_BUFFER, category);
          // The category buffer is a Uint8Array; read it unsigned so values >= 128
          // are not interpreted as negative (which would corrupt the color lookup).
          gl.vertexAttribIPointer(3, 1, gl.UNSIGNED_BYTE, 0, 0);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        gl.uniformMatrix4fv(program.uniforms.u_matrix, false, matrix4);
        gl.uniform1f(program.uniforms.u_point_size, pointSize);
        gl.uniform1f(program.uniforms.u_camera_distance, cameraDistance);
        gl.uniform1f(program.uniforms.u_fog_density, fogDensity);
        gl.uniform3f(program.uniforms.u_eye, eye[0], eye[1], eye[2]);
        gl.uniform1f(program.uniforms.u_gamma, gamma);
        gl.uniform3f(program.uniforms.u_background, background[0], background[1], background[2]);
        if (hasCategory) {
          // colorScheme is declared as vec4[64]; uploading more is an invalid
          // past-size update. Categories >= 64 fall back to gray in the shader.
          gl.uniform4fv(program.uniforms.colorScheme, colors.slice(0, 64 * 4));
        } else {
          gl.uniform4fv(program.uniforms.colorScheme, colors.slice(0, 4));
        }

        // Bounded by downsampleMaxPoints. When a strided index buffer exists, draw
        // that representative subset (gl_VertexID = the element value = real index);
        // otherwise draw all points (gl_VertexID = real index).
        if (sampleIndex != null) {
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sampleIndex);
          gl.drawElements(gl.POINTS, Math.min(count, drawCount), gl.UNSIGNED_INT, 0);
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        } else {
          gl.drawArrays(gl.POINTS, 0, Math.min(count, drawCount));
        }

        gl.disableVertexAttribArray(0);
        gl.disableVertexAttribArray(1);
        gl.disableVertexAttribArray(2);
        if (category != null) {
          gl.disableVertexAttribArray(3);
        }
        gl.useProgram(null);
        gl.disable(gl.DEPTH_TEST);
      },
  );
}

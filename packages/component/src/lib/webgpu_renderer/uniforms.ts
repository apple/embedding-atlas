// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Dataflow, DataflowNode } from "@embedding-atlas/utils";

import type { Matrix3, Vector2, Vector4 } from "../matrix.js";
import { gpuBuffer } from "./utils.js";

export interface Uniforms {
  count: number;
  category_count: number;
  framebuffer_width: number;
  framebuffer_height: number;
  density_width: number;
  density_height: number;
  gamma: number;
  point_size: number;
  point_alpha: number;
  points_alpha: number;
  density_scaler: number;
  quantization_step: number;
  density_alpha: number;
  contours_alpha: number;
  matrix: Matrix3;
  view_xy_scaler: Vector2;
  kde_causal: Vector4;
  kde_anticausal: Vector4;
  kde_a: Vector4;
  background_color: Vector4;
  category_colors: { r: number; g: number; b: number; a: number }[];
}

export class StructWriter {
  private i32View: Int32Array;
  private u32View: Uint32Array;
  private f32View: Float32Array;
  private offset: number;

  constructor(buffer: ArrayBuffer) {
    this.i32View = new Int32Array(buffer);
    this.u32View = new Uint32Array(buffer);
    this.f32View = new Float32Array(buffer);
    this.offset = 0;
  }

  private align2() {
    if (this.offset % 2 != 0) {
      this.offset += 2 - (this.offset % 2);
    }
  }

  private align4() {
    if (this.offset % 4 != 0) {
      this.offset += 4 - (this.offset % 4);
    }
  }

  f32(value: number) {
    this.f32View[this.offset++] = value;
  }

  u32(value: number) {
    this.u32View[this.offset++] = value;
  }

  i32(value: number) {
    this.i32View[this.offset++] = value;
  }

  vec2f(x: number, y: number) {
    this.align2();
    this.f32View[this.offset++] = x;
    this.f32View[this.offset++] = y;
  }

  vec3f(x: number, y: number, z: number) {
    this.align4();
    this.f32View[this.offset++] = x;
    this.f32View[this.offset++] = y;
    this.f32View[this.offset++] = z;
  }

  vec4f(x: number, y: number, z: number, w: number) {
    this.align4();
    this.f32View[this.offset++] = x;
    this.f32View[this.offset++] = y;
    this.f32View[this.offset++] = z;
    this.f32View[this.offset++] = w;
  }

  mat3x3f(matrix: Matrix3) {
    this.vec3f(matrix[0], matrix[1], matrix[2]);
    this.vec3f(matrix[3], matrix[4], matrix[5]);
    this.vec3f(matrix[6], matrix[7], matrix[8]);
  }

  byteOffset() {
    return this.offset * 4;
  }
}

export interface ModuleUniforms {
  buffer: DataflowNode<GPUBuffer>;
  update: DataflowNode<(uniforms: Uniforms) => void>;
}

export function makeModuleUniforms(df: Dataflow, device: DataflowNode<GPUDevice>): ModuleUniforms {
  const byteSize = 4288;
  let buffer = new ArrayBuffer(byteSize);

  let uniformBuffer = df.statefulDerive(
    [device, byteSize, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX],
    gpuBuffer,
  );

  return {
    buffer: uniformBuffer,
    update: df.derive([device, uniformBuffer], (device, uniformBuffer) => (uniforms: Uniforms) => {
      let writer = new StructWriter(buffer);
      writer.u32(uniforms.count);
      writer.u32(uniforms.category_count);
      writer.i32(uniforms.framebuffer_width);
      writer.i32(uniforms.framebuffer_height);
      writer.i32(uniforms.density_width);
      writer.i32(uniforms.density_height);
      writer.f32(uniforms.gamma);
      writer.f32(uniforms.point_size);
      writer.f32(uniforms.point_alpha);
      writer.f32(uniforms.points_alpha);
      writer.f32(uniforms.density_scaler);
      writer.f32(uniforms.quantization_step);
      writer.f32(uniforms.density_alpha);
      writer.f32(uniforms.contours_alpha);
      writer.mat3x3f(uniforms.matrix);
      writer.vec2f(...uniforms.view_xy_scaler);
      writer.vec4f(...uniforms.kde_causal);
      writer.vec4f(...uniforms.kde_anticausal);
      writer.vec4f(...uniforms.kde_a);
      writer.vec4f(...uniforms.background_color);
      let gamma = uniforms.gamma;
      for (let i = 0; i < Math.min(uniforms.category_colors.length, 256); i++) {
        let { r, g, b, a } = uniforms.category_colors[i];
        r = Math.pow(r, gamma);
        g = Math.pow(g, gamma);
        b = Math.pow(b, gamma);
        writer.vec4f(r, g, b, a);
      }
      device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, writer.byteOffset());
    }),
  };
}

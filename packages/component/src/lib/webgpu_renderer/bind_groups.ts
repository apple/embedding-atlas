// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { AuxiliaryResources, DataBuffers } from "./renderer.js";

export interface BindGroups {
  layouts: Node<{
    group0: GPUBindGroupLayout;
    group1: GPUBindGroupLayout;
    group2A: GPUBindGroupLayout;
    group2B: GPUBindGroupLayout;
    group3: GPUBindGroupLayout;
  }>;
  group0: Node<GPUBindGroup>;
  group1: Node<GPUBindGroup>;
  group2A: Node<GPUBindGroup>;
  group2B: Node<GPUBindGroup>;
  group3: Node<GPUBindGroup>;
  // Note: group4 for downsampling is managed separately in downsample.ts
}

export function makeBindGroupLayouts(device: GPUDevice): {
  group0: GPUBindGroupLayout;
  group1: GPUBindGroupLayout;
  group2A: GPUBindGroupLayout;
  group2B: GPUBindGroupLayout;
  group3: GPUBindGroupLayout;
} {
  const { COMPUTE, VERTEX, FRAGMENT } = GPUShaderStage;
  return {
    // Group 0
    group0: device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: COMPUTE | VERTEX | FRAGMENT, buffer: { type: "uniform" } }],
    }),
    // Group 1
    group1: device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: COMPUTE | VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: COMPUTE | VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: COMPUTE | VERTEX, buffer: { type: "read-only-storage" } },
      ],
    }),
    // Group 2
    group2A: device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: COMPUTE | FRAGMENT, buffer: { type: "storage" } }],
    }),
    group2B: device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: COMPUTE | FRAGMENT, buffer: { type: "storage" } },
        { binding: 2, visibility: COMPUTE | FRAGMENT, buffer: { type: "storage" } },
      ],
    }),
    // Group 3
    group3: device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "non-filtering" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    }),
  };
}

export function makeBindGroups(
  df: Dataflow,
  device: Node<GPUDevice>,
  uniformBuffer: Node<GPUBuffer>,
  dataBuffers: DataBuffers,
  auxiliaryResources: AuxiliaryResources,
): BindGroups {
  let layouts = df.derive([device], (device) => makeBindGroupLayouts(device));
  let group0 = df.derive([device, layouts, uniformBuffer], (device, layouts, uniformBuffer) =>
    device.createBindGroup({
      layout: layouts.group0,
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    }),
  );
  let group1 = df.derive(
    [device, layouts, dataBuffers.x, dataBuffers.y, dataBuffers.category],
    (device, layouts, x, y, category) =>
      device.createBindGroup({
        layout: layouts.group1,
        entries: [
          { binding: 0, resource: { buffer: x } },
          { binding: 1, resource: { buffer: y } },
          { binding: 2, resource: { buffer: category ?? x } },
        ],
      }),
  );
  let group2A = df.derive(
    [device, layouts, auxiliaryResources.countBuffer, auxiliaryResources.blurBuffer],
    (device, layouts, countBuffer, blurBuffer) =>
      device.createBindGroup({
        layout: layouts.group2A,
        entries: [{ binding: 0, resource: { buffer: countBuffer } }],
      }),
  );
  let group2B = df.derive(
    [device, layouts, auxiliaryResources.countBuffer, auxiliaryResources.blurBuffer],
    (device, layouts, countBuffer, blurBuffer) =>
      device.createBindGroup({
        layout: layouts.group2B,
        entries: [
          { binding: 1, resource: { buffer: countBuffer } },
          { binding: 2, resource: { buffer: blurBuffer } },
        ],
      }),
  );
  let group3 = df.derive(
    [device, layouts, auxiliaryResources.colorTexture, auxiliaryResources.alphaTexture],
    (device, layouts, colorTexture, alphaTexture) =>
      device.createBindGroup({
        layout: layouts.group3,
        entries: [
          { binding: 0, resource: device.createSampler({}) },
          { binding: 1, resource: colorTexture.createView() },
          { binding: 2, resource: alphaTexture.createView() },
        ],
      }),
  );

  return {
    layouts: layouts,
    group0: group0,
    group1: group1,
    group2A: group2A,
    group2B: group2B,
    group3: group3,
  };
}

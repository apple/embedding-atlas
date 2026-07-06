// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { BindGroups } from "./bind_groups.js";

const WORKGROUP_SIZE = 64;

export function makeGaussianBlurCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  bindGroups: BindGroups,
  width: Node<number>,
  height: Node<number>,
  categoryCount: Node<number>,
): Node<(encoder: GPUCommandEncoder) => void> {
  let pipeline1 = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [layouts.group0, layouts.group1, layouts.group2BlurForward, layouts.group3],
      }),
      compute: { module: module, entryPoint: "gaussian_blur_stage_1" },
    }),
  );
  let pipeline2 = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [layouts.group0, layouts.group1, layouts.group2B, layouts.group3],
      }),
      compute: { module: module, entryPoint: "gaussian_blur_stage_2" },
    }),
  );
  return df.derive(
    [
      pipeline1,
      pipeline2,
      bindGroups.group0,
      bindGroups.group1,
      bindGroups.group2BlurForward,
      bindGroups.group2B,
      bindGroups.group3,
      width,
      height,
      categoryCount,
    ],
    (pipeline1, pipeline2, group0, group1, group2BlurForward, group2B, group3, width, height, categoryCount) =>
      (encoder) => {
        let pass = encoder.beginComputePass();
        pass.setBindGroup(0, group0);
        pass.setBindGroup(1, group1);
        pass.setBindGroup(3, group3);
        // Forward pass reads count_buffer (group2BlurForward: bindings 0, 2).
        pass.setBindGroup(2, group2BlurForward);
        pass.setPipeline(pipeline1);
        pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP_SIZE), categoryCount);
        // Backward pass writes blur_buffer (group2B: bindings 1, 2). Using a separate
        // bind group keeps the aliasing count_buffer/blur_buffer bindings apart so no
        // single dispatch's pipeline layout declares both against the same buffer.
        pass.setBindGroup(2, group2B);
        pass.setPipeline(pipeline2);
        pass.dispatchWorkgroups(Math.ceil(height / WORKGROUP_SIZE), categoryCount);
        pass.end();
      },
  );
}

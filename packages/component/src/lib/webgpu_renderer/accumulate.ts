// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, DataflowNode } from "@embedding-atlas/utils";

import type { BindGroups } from "./bind_groups.js";
import type { AuxiliaryResources, DataBuffers } from "./renderer.js";

const WORKGROUP_SIZE_X = 64;
const WORKGROUP_MAX_COUNT_X = 64;

export function makeAccumulateCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  bindGroups: BindGroups,
  dataBuffers: DataBuffers,
  auxiliaryResources: AuxiliaryResources,
): DataflowNode<(encoder: GPUCommandEncoder) => void> {
  let pipeline = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.group0, layouts.group1, layouts.group2A] }),
      compute: { module: module, entryPoint: "accumulate" },
    }),
  );
  return df.derive(
    [
      pipeline,
      bindGroups.group0,
      bindGroups.group1,
      bindGroups.group2A,
      auxiliaryResources.countBuffer,
      dataBuffers.count,
    ],
    (pipeline, group0, group1, group2A, countBuffer, count) => (encoder) => {
      encoder.clearBuffer(countBuffer);
      if (count == 0) {
        return;
      }
      let pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      pass.setBindGroup(2, group2A);
      if (count <= WORKGROUP_SIZE_X * WORKGROUP_MAX_COUNT_X) {
        pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE_X));
      } else {
        pass.dispatchWorkgroups(WORKGROUP_MAX_COUNT_X, Math.ceil(count / (WORKGROUP_SIZE_X * WORKGROUP_MAX_COUNT_X)));
      }
      pass.end();
    },
  );
}

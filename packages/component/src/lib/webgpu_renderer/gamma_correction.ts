// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, DataflowNode } from "@embedding-atlas/utils";

import type { BindGroups } from "./bind_groups.js";

export function makeGammaCorrectionCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  format: GPUTextureFormat,
  bindGroups: BindGroups,
): DataflowNode<(encoder: GPUCommandEncoder, textureView: GPUTextureView) => void> {
  const pipeline = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [layouts.group0, layouts.group1, layouts.group2B, layouts.group3],
      }),
      vertex: { entryPoint: "gamma_correction_vs", module: module },
      fragment: { entryPoint: "gamma_correction_fs", module: module, targets: [{ format: format }] },
      primitive: { topology: "triangle-strip" },
    }),
  );

  return df.derive(
    [pipeline, bindGroups.group0, bindGroups.group1, bindGroups.group2B, bindGroups.group3],
    (pipeline, group0, group1, group2B, group3) => (encoder, textureView) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [{ clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store", view: textureView }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      pass.setBindGroup(2, group2B);
      pass.setBindGroup(3, group3);
      pass.draw(4);
      pass.end();
    },
  );
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, DataflowNode } from "@embedding-atlas/utils";

import type { BindGroups } from "./bind_groups.js";
import type { AuxiliaryResources } from "./renderer.js";

export function makeDrawDensityMapCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  bindGroups: BindGroups,
  auxiliaryResources: AuxiliaryResources,
): DataflowNode<(encoder: GPUCommandEncoder) => void> {
  const pipeline = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [layouts.group0, layouts.group1, layouts.group2B],
      }),
      vertex: { entryPoint: "draw_density_map_vs", module: module },
      fragment: {
        entryPoint: "draw_density_map_fs",
        module: module,
        targets: [
          {
            format: auxiliaryResources.colorTextureFormat,
            blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
          },
          {
            format: auxiliaryResources.alphaTextureFormat,
            blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
    }),
  );

  return df.derive(
    [
      pipeline,
      bindGroups.group0,
      bindGroups.group1,
      bindGroups.group2B,
      auxiliaryResources.colorTexture,
      auxiliaryResources.alphaTexture,
    ],
    (pipeline, group0, group1, group2B, colorTexture, alphaTexture) => (encoder) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [
          { loadOp: "load", storeOp: "store", view: colorTexture.createView() },
          { loadOp: "load", storeOp: "store", view: alphaTexture.createView() },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      pass.setBindGroup(2, group2B);
      pass.draw(4);
      pass.end();
    },
  );
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, DataflowNode } from "@embedding-atlas/utils";

export function makeGammaCorrectionCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  format: GPUTextureFormat,
  group0Layout: DataflowNode<GPUBindGroupLayout>,
  group3Layout: DataflowNode<GPUBindGroupLayout>,
  group0: DataflowNode<GPUBindGroup>,
  group3: DataflowNode<GPUBindGroup>,
): DataflowNode<(encoder: GPUCommandEncoder, textureView: GPUTextureView) => void> {
  // Gamma correction only reads the uniforms (group 0) and the framebuffer textures (group 3),
  // so groups 1 and 2 are empty. This keeps the pass independent of any layer's data buffers.
  const emptyLayout = df.derive([device], (device) => device.createBindGroupLayout({ entries: [] }));
  const emptyBindGroup = df.derive([device, emptyLayout], (device, layout) =>
    device.createBindGroup({ layout, entries: [] }),
  );

  const pipeline = df.derive(
    [device, module, group0Layout, emptyLayout, group3Layout],
    (device, module, group0Layout, emptyLayout, group3Layout) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [group0Layout, emptyLayout, emptyLayout, group3Layout],
        }),
        vertex: { entryPoint: "gamma_correction_vs", module: module },
        fragment: { entryPoint: "gamma_correction_fs", module: module, targets: [{ format: format }] },
        primitive: { topology: "triangle-strip" },
      }),
  );

  return df.derive(
    [pipeline, group0, emptyBindGroup, group3],
    (pipeline, group0, emptyBindGroup, group3) => (encoder, textureView) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [{ clearValue: [1, 1, 1, 1], loadOp: "clear", storeOp: "store", view: textureView }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, emptyBindGroup);
      pass.setBindGroup(2, emptyBindGroup);
      pass.setBindGroup(3, group3);
      pass.draw(4);
      pass.end();
    },
  );
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { BindGroups } from "./bind_groups.js";
import type { DownsampleResources } from "./downsample.js";
import type { AuxiliaryResources, DataBuffers } from "./renderer.js";

export function makeDrawPointsCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  bindGroups: BindGroups,
  dataBuffers: DataBuffers,
  auxiliaryResources: AuxiliaryResources,
): Node<(encoder: GPUCommandEncoder) => void> {
  const pipeline = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.group0, layouts.group1] }),
      vertex: { entryPoint: "points_vs", module: module },
      fragment: {
        entryPoint: "points_fs",
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
      dataBuffers.count,
      auxiliaryResources.colorTexture,
      auxiliaryResources.alphaTexture,
    ],
    (pipeline, group0, group1, count, colorTexture, alphaTexture) => (encoder) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [
          { clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store", view: colorTexture.createView() },
          { clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store", view: alphaTexture.createView() },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      if (count > 0) {
        pass.draw(4, count);
      }
      pass.end();
    },
  );
}

/**
 * Draw points using an index buffer for downsampled rendering.
 * Uses points_indexed_vs vertex shader that reads point indices from the index buffer.
 * Note: Uses group 2 for the vertex shader (read-only index buffer access).
 * Pipeline layout: group 0 (uniforms), group 1 (data buffers), group 2 (index buffer read).
 */
export function makeDrawPointsIndexedCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  bindGroups: BindGroups,
  downsampleResources: DownsampleResources,
  auxiliaryResources: AuxiliaryResources,
): Node<(encoder: GPUCommandEncoder, count: number) => void> {
  const pipeline = df.derive(
    [device, module, bindGroups.layouts, downsampleResources.vertexBindGroupLayout],
    (device, module, layouts, group5Layout) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [layouts.group0, layouts.group1, group5Layout],
        }),
        vertex: { entryPoint: "points_indexed_vs", module: module },
        fragment: {
          entryPoint: "points_fs",
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
      downsampleResources.vertexBindGroup,
      auxiliaryResources.colorTexture,
      auxiliaryResources.alphaTexture,
    ],
    (pipeline, group0, group1, group5, colorTexture, alphaTexture) => (encoder, count) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [
          { clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store", view: colorTexture.createView() },
          { clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store", view: alphaTexture.createView() },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      pass.setBindGroup(2, group5);
      if (count > 0) {
        pass.draw(4, count);
      }
      pass.end();
    },
  );
}

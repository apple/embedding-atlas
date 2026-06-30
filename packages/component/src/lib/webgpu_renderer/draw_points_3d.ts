// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { BindGroups } from "./bind_groups.js";
import type { DataBuffers } from "./renderer.js";

/**
 * Draws the data as a depth-tested 3D point cloud directly to the canvas.
 *
 * Unlike the 2D path (order-independent additive accumulation + a gamma pass),
 * 3D points are opaque, depth-tested billboards rendered in a single pass with
 * gamma applied in-shader. The returned command clears the color target to
 * `clearColor` and the depth target to 1.0 each frame.
 */
export function makeDrawPoints3DCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  format: GPUTextureFormat,
  bindGroups: BindGroups,
  dataBuffers: DataBuffers,
): Node<
  (
    encoder: GPUCommandEncoder,
    textureView: GPUTextureView,
    depthView: GPUTextureView,
    clearColor: GPUColor,
    drawCount: number,
  ) => void
> {
  const pipeline = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.group0, layouts.group1] }),
      vertex: { entryPoint: "points_3d_vs", module: module },
      fragment: {
        entryPoint: "points_3d_fs",
        module: module,
        targets: [
          {
            format: format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    }),
  );

  return df.derive(
    [pipeline, bindGroups.group0, bindGroups.group1],
    (pipeline, group0, group1) => (encoder, textureView, depthView, clearColor, drawCount) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [{ clearValue: clearColor, loadOp: "clear", storeOp: "store", view: textureView }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      if (drawCount > 0) {
        pass.draw(4, drawCount);
      }
      pass.end();
    },
  );
}

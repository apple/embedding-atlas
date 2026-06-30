// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { BindGroups } from "./bind_groups.js";
import { gpuBuffer, gpuTexture } from "./utils.js";

/**
 * Picks the frontmost 3D point under a render-target pixel location.
 *
 * Renders the same billboards as the 3D draw (same stride-bounded subset), but each
 * fragment writes the real point index + 1 to an r32uint target with the same depth
 * test, so the nearest point wins. A 1x1 region under the cursor is copied back and
 * decoded.
 *
 * `writeUniforms` (when redrawing) writes the 3D uniforms for THIS request's camera
 * snapshot, and is invoked INSIDE the serialized chain immediately before the redraw
 * is encoded. This keeps a queued pick from redrawing with a newer camera's matrix
 * that a later request wrote into the shared uniform buffer.
 */
export function makePick3DCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  bindGroups: BindGroups,
  width: Node<number>,
  height: Node<number>,
): Node<
  (
    px: number,
    py: number,
    drawCount: number,
    redraw: boolean,
    writeUniforms: (() => void) | null,
  ) => Promise<number | null>
> {
  const pickFormat: GPUTextureFormat = "r32uint";
  let usageColor = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
  let usageDepth = GPUTextureUsage.RENDER_ATTACHMENT;

  // Clamp to >= 1: WebGPU rejects zero-sized textures, and these nodes are evaluated
  // (allocated) before runPick's `width/height <= 0` no-op guard runs, so a collapsed
  // or resizing layout (0 width/height) would otherwise throw on texture creation.
  let pickWidth = df.derive([width], (w: number) => Math.max(1, w));
  let pickHeight = df.derive([height], (h: number) => Math.max(1, h));
  let pickTexture = df.statefulDerive([device, pickWidth, pickHeight, pickFormat, usageColor], gpuTexture);
  let depthTexture = df.statefulDerive(
    [device, pickWidth, pickHeight, "depth24plus" as GPUTextureFormat, usageDepth],
    gpuTexture,
  );
  // A single reused readback buffer (256 = min bytesPerRow), so hover-picking does
  // not allocate/destroy a buffer per request.
  let readbackBuffer = df.statefulDerive([device, 256, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ], gpuBuffer);

  const pipeline = df.derive([device, module, bindGroups.layouts], (device, module, layouts) =>
    device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.group0, layouts.group1] }),
      vertex: { entryPoint: "pick_3d_vs", module: module },
      fragment: { entryPoint: "pick_3d_fs", module: module, targets: [{ format: pickFormat }] },
      primitive: { topology: "triangle-strip" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    }),
  );

  // Serialize picks through a promise chain. This keeps the shared readbackBuffer
  // from being mapped concurrently WITHOUT dropping requests: a click that arrives
  // during an in-flight hover still runs (after it) and returns its real result,
  // rather than a false "no hit" that would clear selection.
  //
  // IMPORTANT: `chain` lives OUTSIDE the derived closure below so it persists across
  // closure rebuilds (which happen when sampleIndices refine, data swaps, or the view
  // resizes — those change pickTexture/group1/width/height). If it were reset on each
  // rebuild, a pick issued from the new closure could map readbackBuffer while an
  // older pick is still awaiting mapAsync on that same buffer, corrupting both.
  let chain: Promise<number | null> = Promise.resolve(null);

  return df.derive(
    [device, pipeline, bindGroups.group0, bindGroups.group1, pickTexture, depthTexture, readbackBuffer, width, height],
    (device, pipeline, group0, group1, pickTexture, depthTexture, readbackBuffer, width, height) => {
      async function runPick(
        px: number,
        py: number,
        drawCount: number,
        redraw: boolean,
        writeUniforms: (() => void) | null,
      ): Promise<number | null> {
        if (drawCount <= 0 || width <= 0 || height <= 0) {
          return null;
        }
        // Clamp into range so a redraw always renders (keeping the cached texture
        // valid) and the readback never samples out of bounds.
        let ix = Math.min(width - 1, Math.max(0, Math.floor(px)));
        let iy = Math.min(height - 1, Math.max(0, Math.floor(py)));
        let encoder = device.createCommandEncoder();
        // Only re-render the ID texture when the view changed since the last pick.
        // Repeated hovers over a static camera reuse it and just read back a pixel.
        if (redraw) {
          // Write THIS request's uniforms here, inside the serialized chain, so a
          // later request's camera cannot overwrite the shared buffer before this
          // redraw is encoded.
          writeUniforms?.();
          let pass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: pickTexture.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
            depthStencilAttachment: {
              view: depthTexture.createView(),
              depthClearValue: 1.0,
              depthLoadOp: "clear",
              depthStoreOp: "store",
            },
          });
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, group0);
          pass.setBindGroup(1, group1);
          pass.draw(4, drawCount);
          pass.end();
        }

        // bytesPerRow must be a multiple of 256.
        encoder.copyTextureToBuffer(
          { texture: pickTexture, origin: { x: ix, y: iy, z: 0 } },
          { buffer: readbackBuffer, bytesPerRow: 256 },
          { width: 1, height: 1, depthOrArrayLayers: 1 },
        );
        device.queue.submit([encoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        let value = new Uint32Array(readbackBuffer.getMappedRange(0, 4))[0];
        readbackBuffer.unmap();
        return value === 0 ? null : value - 1;
      }

      return (
        px: number,
        py: number,
        drawCount: number,
        redraw: boolean,
        writeUniforms: (() => void) | null,
      ): Promise<number | null> => {
        let result = chain.then(
          () => runPick(px, py, drawCount, redraw, writeUniforms),
          () => runPick(px, py, drawCount, redraw, writeUniforms),
        );
        // Keep the chain alive regardless of this pick's outcome.
        chain = result.then(
          () => null,
          () => null,
        );
        return result;
      };
    },
  );
}

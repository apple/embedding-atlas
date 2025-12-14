// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { DataBuffers } from "./renderer.js";
import { gpuBuffer } from "./utils.js";

const WORKGROUP_SIZE = 256;
// Fixed X workgroups count for 2D dispatch (256 * 256 = 65536 threads per row)
// This matches the stride used in the shader: id.y * 65536 + id.x
const WORKGROUPS_X = 256;
const THREADS_PER_ROW = WORKGROUPS_X * WORKGROUP_SIZE; // 65536

// Helper to compute 2D dispatch dimensions for large point counts
function computeDispatch(count: number): [number, number] {
  const totalWorkgroups = Math.ceil(count / WORKGROUP_SIZE);
  if (totalWorkgroups <= WORKGROUPS_X) {
    return [totalWorkgroups, 1];
  }
  // Use 2D dispatch with fixed X stride
  const y = Math.ceil(count / THREADS_PER_ROW);
  return [WORKGROUPS_X, y];
}

export interface DownsampleResources {
  uniformBuffer: Node<GPUBuffer>;
  countersBuffer: Node<GPUBuffer>;
  pointDataBuffer: Node<GPUBuffer>; // Merged: density + visibility + acceptance
  indexBuffer: Node<GPUBuffer>;
  // Group 3: for compute shaders (read_write access)
  bindGroupLayout: Node<GPUBindGroupLayout>;
  bindGroup: Node<GPUBindGroup>;
  // Group 2 in indexed draw pipeline: for vertex shader (read-only access to index buffer)
  vertexBindGroupLayout: Node<GPUBindGroupLayout>;
  vertexBindGroup: Node<GPUBindGroup>;
}

export interface DownsampleConfig {
  maxPoints: number;
  densityWeight: number;
  frameSeed: number;
}

export function makeDownsampleResources(
  df: Dataflow,
  device: Node<GPUDevice>,
  count: Node<number>,
  downsampleMaxPoints: Node<number | null>,
): DownsampleResources {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  // Uniform buffer for downsample uniforms (16 bytes: render_limit, frame_seed, density_weight, padding)
  const uniformBuffer = df.statefulDerive(
    [device, df.value(16), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST],
    gpuBuffer,
  );

  // Counters buffer: [visible_count, max_density_fixed, selected_count] = 12 bytes, pad to 16
  const countersBuffer = df.statefulDerive([device, df.value(16), usage], gpuBuffer);

  // Per-point buffers (4 bytes per point)
  const pointBufferSize = df.derive([count], (c) => Math.max(4, c * 4));
  // Merged buffer: stores density (>0 = visible+accepted, <=0 = not visible or rejected)
  const pointDataBuffer = df.statefulDerive([device, pointBufferSize, usage], gpuBuffer);

  // Index buffer: sized for max points (use default 4M if null)
  const indexBufferSize = df.derive([downsampleMaxPoints], (limit) => Math.max(4, (limit ?? 4000000) * 4));
  const indexBuffer = df.statefulDerive([device, indexBufferSize, usage], gpuBuffer);

  // Bind group layout for group 3 (compute shaders - read_write access)
  // 4 bindings: uniform + 3 storage buffers (removed prefix_sum, now use atomic counter)
  const bindGroupLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // counters
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // point_data
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // index_buffer_write
      ],
    }),
  );

  // Bind group for group 3 (compute)
  const bindGroup = df.derive(
    [device, bindGroupLayout, uniformBuffer, countersBuffer, pointDataBuffer, indexBuffer],
    (device, layout, uniform, counters, pointData, index) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: counters } },
          { binding: 2, resource: { buffer: pointData } },
          { binding: 3, resource: { buffer: index } },
        ],
      }),
  );

  // Bind group layout for indexed draw pipeline group 2 (vertex shader - read-only access to index buffer)
  const vertexBindGroupLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
    }),
  );

  // Bind group for indexed draw pipeline group 2 (vertex)
  const vertexBindGroup = df.derive([device, vertexBindGroupLayout, indexBuffer], (device, layout, index) =>
    device.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: index } }],
    }),
  );

  return {
    uniformBuffer,
    countersBuffer,
    pointDataBuffer,
    indexBuffer,
    bindGroupLayout,
    bindGroup,
    vertexBindGroupLayout,
    vertexBindGroup,
  };
}

export function makeDownsampleCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  group0Layout: Node<GPUBindGroupLayout>,
  group1Layout: Node<GPUBindGroupLayout>,
  blurBuffer: Node<GPUBuffer>, // Direct reference to blur_buffer for density lookup
  group0: Node<GPUBindGroup>,
  group1: Node<GPUBindGroup>,
  downsampleResources: DownsampleResources,
  dataBuffers: DataBuffers,
): Node<(encoder: GPUCommandEncoder, config: DownsampleConfig) => number> {
  // Create a minimal bind group layout for blur_buffer (just 1 storage buffer)
  // This keeps viewport_cull under the 8 storage buffer limit:
  // group1 (3) + blurOnly (1) + group4 (4) = 8
  // Note: Must match shader declaration which uses read_write (even though we only read)
  const blurOnlyLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
    }),
  );

  const blurOnlyBindGroup = df.derive([device, blurOnlyLayout, blurBuffer], (device, layout, buffer) =>
    device.createBindGroup({
      layout,
      entries: [{ binding: 1, resource: { buffer } }],
    }),
  );

  // Create empty layouts for unused group 3 in viewport_cull
  const emptyLayout = df.derive([device], (device) => device.createBindGroupLayout({ entries: [] }));
  const emptyBindGroup = df.derive([device, emptyLayout], (device, layout) =>
    device.createBindGroup({ layout, entries: [] }),
  );

  // viewport_cull needs blur_buffer for density lookup
  // Pipeline layout: [group0, group1, blurOnly, group3] to match @group(3) for downsample buffers
  const viewportCullPipeline = df.derive(
    [device, module, group0Layout, group1Layout, blurOnlyLayout, downsampleResources.bindGroupLayout],
    (device, module, group0, group1, group2, group3) =>
      device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1, group2, group3] }),
        compute: { module, entryPoint: "downsample_viewport_cull" },
      }),
  );

  // Other passes don't need blur_buffer - they work with point_data which was already computed
  // Pipeline layout: [group0, group1, empty, group3] to match @group() numbers
  const densitySamplePipeline = df.derive(
    [device, module, group0Layout, group1Layout, emptyLayout, downsampleResources.bindGroupLayout],
    (device, module, group0, group1, empty, group3) =>
      device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1, empty, group3] }),
        compute: { module, entryPoint: "downsample_density_sample" },
      }),
  );

  // Stream compaction using atomic counter (no prefix sum needed)
  const compactPipeline = df.derive(
    [device, module, group0Layout, group1Layout, emptyLayout, downsampleResources.bindGroupLayout],
    (device, module, group0, group1, empty, group3) =>
      device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1, empty, group3] }),
        compute: { module, entryPoint: "downsample_compact" },
      }),
  );

  return df.derive(
    [
      device,
      viewportCullPipeline,
      densitySamplePipeline,
      compactPipeline,
      group0,
      group1,
      blurOnlyBindGroup,
      emptyBindGroup,
      downsampleResources.bindGroup,
      downsampleResources.uniformBuffer,
      downsampleResources.countersBuffer,
      dataBuffers.count,
    ],
    (
      device,
      viewportCullPipeline,
      densitySamplePipeline,
      compactPipeline,
      group0,
      group1,
      group2Blur,
      emptyGroup,
      group3,
      uniformBuffer,
      countersBuffer,
      count,
    ) =>
      (encoder, config) => {
        if (count === 0 || config.maxPoints <= 0) {
          return 0;
        }

        // Update uniform buffer
        const uniformData = new ArrayBuffer(16);
        const uniformView = new DataView(uniformData);
        uniformView.setUint32(0, config.maxPoints, true);
        uniformView.setUint32(4, config.frameSeed, true);
        uniformView.setFloat32(8, config.densityWeight, true);
        uniformView.setFloat32(12, 0, true); // padding
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Clear counters
        encoder.clearBuffer(countersBuffer);

        const [workgroupsX, workgroupsY] = computeDispatch(count);

        // Pass 1: Viewport culling + density lookup (needs blur_buffer for density)
        // Pipeline layout: [group0, group1, blurOnly, group3]
        {
          const pass = encoder.beginComputePass();
          pass.setPipeline(viewportCullPipeline);
          pass.setBindGroup(0, group0);
          pass.setBindGroup(1, group1);
          pass.setBindGroup(2, group2Blur);
          pass.setBindGroup(3, group3);
          pass.dispatchWorkgroups(workgroupsX, workgroupsY);
          pass.end();
        }

        // Pass 2: Probabilistic acceptance based on density
        // Pipeline layout: [group0, group1, empty, group3]
        {
          const pass = encoder.beginComputePass();
          pass.setPipeline(densitySamplePipeline);
          pass.setBindGroup(0, group0);
          pass.setBindGroup(1, group1);
          pass.setBindGroup(2, emptyGroup);
          pass.setBindGroup(3, group3);
          pass.dispatchWorkgroups(workgroupsX, workgroupsY);
          pass.end();
        }

        // Pass 3: Stream compaction using atomic counter
        {
          const pass = encoder.beginComputePass();
          pass.setPipeline(compactPipeline);
          pass.setBindGroup(0, group0);
          pass.setBindGroup(1, group1);
          pass.setBindGroup(2, emptyGroup);
          pass.setBindGroup(3, group3);
          pass.dispatchWorkgroups(workgroupsX, workgroupsY);
          pass.end();
        }

        // Return the max points as the max possible count
        // The actual count will be determined by the GPU, but we use maxPoints as upper bound
        return Math.min(count, config.maxPoints);
      },
  );
}

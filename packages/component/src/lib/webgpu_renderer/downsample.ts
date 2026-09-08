// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { DataBuffers } from "./renderer.js";
import { gpuBuffer } from "./utils.js";
import { downsamplePipelineConstants, resolveWgConfig } from "./wg_config.js";

// Workgroup + dispatch stride come from the runtime WgConfig; the
// dispatch grid is (workgroups_x, y) where workgroups_x = stride /
// wg_size, matching the shader's `id.y * DOWNSAMPLE_STRIDE + id.x`.
function computeDispatch(count: number, wgSize: number, stride: number): [number, number] {
  const workgroupsX = Math.max(1, Math.floor(stride / wgSize));
  const totalWorkgroups = Math.ceil(count / wgSize);
  if (totalWorkgroups <= workgroupsX) {
    return [totalWorkgroups, 1];
  }
  const y = Math.ceil(count / stride);
  return [workgroupsX, y];
}

export interface DownsampleResources {
  uniformBuffer: Node<GPUBuffer>;
  countersBuffer: Node<GPUBuffer>;
  pointDataBuffer: Node<GPUBuffer>;
  // Compaction outputs — populated by the compact_accepted compute pass and
  // consumed by the indirect-draw vertex pipeline.
  compactIndicesBuffer: Node<GPUBuffer>;
  indirectArgsBuffer: Node<GPUBuffer>;
  // Group 3: for compute shaders (read_write access)
  bindGroupLayout: Node<GPUBindGroupLayout>;
  bindGroup: Node<GPUBindGroup>;
  // Group 2 in indexed draw pipeline: for vertex shader (read-only access to index buffer)
  vertexBindGroupLayout: Node<GPUBindGroupLayout>;
  vertexBindGroup: Node<GPUBindGroup>;
  // Group 2 in compacted draw pipeline: vertex reads compact_indices_read.
  compactedVertexBindGroupLayout: Node<GPUBindGroupLayout>;
  compactedVertexBindGroup: Node<GPUBindGroup>;
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
  // Uniform buffer for downsample uniforms (16 bytes: render_limit, frame_seed, density_weight, padding)
  const uniformBuffer = df.statefulDerive(
    [device, df.value(16), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST],
    gpuBuffer,
  );

  // Counters buffer: [visible_count, max_density_fixed] = 8 bytes, pad to 16
  const countersBuffer = df.statefulDerive(
    [device, df.value(16), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST],
    gpuBuffer,
  );

  // Per-point buffers (4 bytes per point), for point_data in the shader
  const pointBufferSize = df.derive([count], (c) => Math.max(4, c * 4));
  const pointDataBuffer = df.statefulDerive([device, pointBufferSize, GPUBufferUsage.STORAGE], gpuBuffer);

  // Compaction targets. compact_indices is sized for the maximum acceptable
  // points so the over-accept guard in the shader cannot overflow it.
  // indirect_args is the 16-byte drawIndirect descriptor; we re-zero
  // instanceCount each frame and the compact_accepted shader fills it.
  const compactBufferSize = df.derive([count, downsampleMaxPoints], (c, m) => Math.max(4, Math.min(c, m ?? c) * 4));
  const compactIndicesBuffer = df.statefulDerive([device, compactBufferSize, GPUBufferUsage.STORAGE], gpuBuffer);
  const indirectArgsBuffer = df.statefulDerive(
    [device, df.value(16), GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST],
    gpuBuffer,
  );

  // Bind group layout for group 3 (compute shaders - read_write access)
  // Adds bindings 3 (compact_indices) and 4 (indirect_args) to host the
  // compaction outputs for compact_accepted. viewport_cull / density_sample
  // also bind this layout but never touch the new bindings, so the extra
  // entries are inert for them.
  const bindGroupLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // counters
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // point_data
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // compact_indices
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // indirect_args
      ],
    }),
  );

  // Bind group for group 3 (compute)
  const bindGroup = df.derive(
    [device, bindGroupLayout, uniformBuffer, countersBuffer, pointDataBuffer, compactIndicesBuffer, indirectArgsBuffer],
    (device, layout, uniform, counters, pointData, compact, indirect) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: counters } },
          { binding: 2, resource: { buffer: pointData } },
          { binding: 3, resource: { buffer: compact } },
          { binding: 4, resource: { buffer: indirect } },
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
  const vertexBindGroup = df.derive([device, vertexBindGroupLayout, pointDataBuffer], (device, layout, buffer) =>
    device.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: buffer } }],
    }),
  );

  // Compacted draw pipeline group 2: a single read-only binding (binding 1) for
  // compact_indices_read. Distinct from vertexBindGroupLayout so the two draw
  // paths can coexist with their own pipeline layouts.
  const compactedVertexBindGroupLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
    }),
  );
  const compactedVertexBindGroup = df.derive(
    [device, compactedVertexBindGroupLayout, compactIndicesBuffer],
    (device, layout, buffer) =>
      device.createBindGroup({
        layout,
        entries: [{ binding: 1, resource: { buffer } }],
      }),
  );

  return {
    uniformBuffer,
    countersBuffer,
    pointDataBuffer,
    compactIndicesBuffer,
    indirectArgsBuffer,
    bindGroupLayout,
    bindGroup,
    vertexBindGroupLayout,
    vertexBindGroup,
    compactedVertexBindGroupLayout,
    compactedVertexBindGroup,
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
): Node<(config: DownsampleConfig) => void> {
  // Create a minimal bind group layout for blur_buffer (just 1 storage buffer)
  // This keeps viewport_cull under the 8 storage buffer limit:
  // group1 (3) + blurOnly (1) + group3 (4) = 8
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

  // Create empty layouts for unused group 2 in density_sample pipeline
  const emptyLayout = df.derive([device], (device) => device.createBindGroupLayout({ entries: [] }));
  const emptyBindGroup = df.derive([device, emptyLayout], (device, layout) =>
    device.createBindGroup({ layout, entries: [] }),
  );

  const wgConfig = resolveWgConfig();
  const dsConstants = downsamplePipelineConstants(wgConfig);

  // viewport_cull needs blur_buffer for density lookup
  // Pipeline layout: [group0, group1, blurOnly, group3] to match @group(3) for downsample buffers
  const viewportCullPipeline = df.derive(
    [device, module, group0Layout, group1Layout, blurOnlyLayout, downsampleResources.bindGroupLayout],
    (device, module, group0, group1, group2, group3) =>
      device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1, group2, group3] }),
        compute: { module, entryPoint: "downsample_viewport_cull", constants: dsConstants },
      }),
  );

  // Other passes don't need blur_buffer - they work with point_data which was already computed
  // Pipeline layout: [group0, group1, empty, group3] to match @group() numbers
  const densitySamplePipeline = df.derive(
    [device, module, group0Layout, group1Layout, emptyLayout, downsampleResources.bindGroupLayout],
    (device, module, group0, group1, empty, group3) =>
      device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1, empty, group3] }),
        compute: { module, entryPoint: "downsample_density_sample", constants: dsConstants },
      }),
  );

  // compact_accepted reads point_data and writes to compact_indices + indirect_args.
  // No blur_buffer or x/y/category data needed.
  const compactPipeline = df.derive(
    [device, module, group0Layout, group1Layout, emptyLayout, downsampleResources.bindGroupLayout],
    (device, module, group0, group1, empty, group3) =>
      device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1, empty, group3] }),
        compute: { module, entryPoint: "compact_accepted", constants: dsConstants },
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
      downsampleResources.indirectArgsBuffer,
      dataBuffers.count,
    ],
    (
      device,
      viewportCullPipeline,
      densitySamplePipeline,
      compactPipeline,
      group0,
      group1,
      blurOnlyGroup,
      emptyGroup,
      group3,
      uniformBuffer,
      countersBuffer,
      indirectArgsBuffer,
      count,
    ) =>
      (config) => {
        // Reset indirect args first so every early-exit path (count=0,
        // maxPoints=0) still clears the draw count.
        const initIndirect = new Uint32Array([4, 0, 0, 0]);
        device.queue.writeBuffer(indirectArgsBuffer, 0, initIndirect);

        if (count === 0 || config.maxPoints <= 0) {
          return 0;
        }

        const [workgroupsX, workgroupsY] = computeDispatch(count, wgConfig.downsampleCull, wgConfig.downsampleStride);

        // Chunked dispatch: each compute pass is split into K
        // command-buffer-sized chunks of (workgroupsX, ~workgroupsY/K)
        // each. K is sized so each chunk dispatches roughly
        // ``CHUNK_TARGET_THREADS`` (currently 64M) — well under Metal's
        // 5 s wall-clock MTLCommandBuffer watchdog regardless of N.
        //
        // Why not one cmd-buffer per pass: at 322 M total points (eubucco)
        // a single viewport_cull pass alone exceeded 5 s and tripped
        // ``kIOGPUCommandBufferCallbackErrorTimeout``. The poison
        // cascade of ``kIOGPUCommandBufferCallbackErrorSubmissionsIgnored``
        // then killed the renderer. Workgroup-local atomic reduction
        // (see program.wgsl ``wg_cull_visible``) collapsed the global
        // atomic hotspot but per-thread compute + memory bandwidth
        // were still O(N), so the watchdog still fired at world-zoom
        // (~322 M visible).
        //
        // The shader honours ``downsample_uniforms.chunk_offset_y`` —
        // each thread's effective index becomes
        // ``(id.y + chunk_offset_y) * DOWNSAMPLE_STRIDE + id.x``.
        // Re-writing the uniform between submits is properly sequenced
        // by the WebGPU queue (``writeBuffer`` is queued and runs in
        // submission order with respect to the cmd buffers it
        // sandwiches).
        const CHUNK_TARGET_THREADS = 16_000_000;
        const threadsPerWorkgroupY = workgroupsX * wgConfig.downsampleCull;
        const targetWorkgroupsPerChunk = Math.max(1, Math.floor(CHUNK_TARGET_THREADS / threadsPerWorkgroupY));
        const numChunks = Math.max(1, Math.ceil(workgroupsY / targetWorkgroupsPerChunk));
        const chunkSizeY = Math.ceil(workgroupsY / numChunks);

        const writeUniform = (chunkOffsetY: number) => {
          const uniformData = new ArrayBuffer(16);
          const uniformView = new DataView(uniformData);
          uniformView.setUint32(0, config.maxPoints, true);
          uniformView.setUint32(4, config.frameSeed, true);
          uniformView.setFloat32(8, config.densityWeight, true);
          uniformView.setUint32(12, chunkOffsetY, true);
          device.queue.writeBuffer(uniformBuffer, 0, uniformData);
        };

        const dispatchPass = (pipeline: GPUComputePipeline, group2: GPUBindGroup, isFirstSubmit: boolean) => {
          for (let chunk = 0; chunk < numChunks; chunk++) {
            const offsetY = chunk * chunkSizeY;
            const remaining = Math.min(chunkSizeY, workgroupsY - offsetY);
            if (remaining <= 0) break;

            writeUniform(offsetY);
            const enc = device.createCommandEncoder();
            // Counter reset must land in the FIRST cmd buffer of the
            // FIRST pass (viewport_cull) — every subsequent pass reads
            // the totals; chunked viewport_cull shares a single visible
            // count across all its chunks because they all atomicAdd
            // into the same counter.
            if (isFirstSubmit && chunk === 0) {
              enc.clearBuffer(countersBuffer);
            }
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group0);
            pass.setBindGroup(1, group1);
            pass.setBindGroup(2, group2);
            pass.setBindGroup(3, group3);
            pass.dispatchWorkgroups(workgroupsX, remaining);
            pass.end();
            device.queue.submit([enc.finish()]);
          }
        };

        if (!(globalThis as any).__atlasDownsampleDiagLogged) {
          (globalThis as any).__atlasDownsampleDiagLogged = true;
          console.log(
            `[atlas-downsample-diag] count=${count} workgroupsX=${workgroupsX} workgroupsY=${workgroupsY} chunkSizeY=${chunkSizeY} numChunks=${numChunks} (${numChunks * 3} cmd-bufs total)`,
          );
        }
        dispatchPass(viewportCullPipeline, blurOnlyGroup, true);
        dispatchPass(densitySamplePipeline, emptyGroup, false);
        dispatchPass(compactPipeline, emptyGroup, false);
      },
  );
}

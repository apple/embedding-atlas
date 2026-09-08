// Copyright (c) 2025 Apple Inc. Licensed under MIT License.
//
// One-shot u32 → f32 unpack for the scatter coordinate buffers.
//
// At 322 M rows the JS-side ``new Float32Array(N)`` for x and y was 2.576 GB
// of heap pressure and the single biggest reason a stock 4 GB Chrome tab
// went blank after pan. Sending packed integers over the wire and unpacking
// on the GPU bypasses that. The earlier cut used u16 (5 B/point) for the
// extra wire saving, but u16 quantises to ~110 m at the eubucco 40°-lon
// span — a visible street-level grid. u32 is the same wire size as raw
// f32 (8 B/point for x+y) but quantises to ~1.5 cm, sub-pixel at any zoom.
//
// Lifetime model: the u32 source buffer + the 16-byte uniform buffer are
// **ephemeral** — created via ``mappedAtCreation: true``, written, dispatched,
// then defer-destroyed once the queue drains. They are never persisted
// across data loads.
//
// Why ephemeral: a previous version held the packed source as a persistent
// dataflow node. At 322 M that's an extra ~1.288 GB per axis sitting in the
// GPU process on top of the persistent 1.288 GB f32 destinations — peak
// GPU-process residency would hit ~5 GB and crash the GPU process on cold
// load (every subsequent ``buffer.destroy()`` then throws "valid external
// Instance reference no longer exists"). Allocating fresh per dispatch
// keeps the steady-state GPU footprint at the f32 destinations only.
//
// Why ``mappedAtCreation: true`` instead of ``device.queue.writeBuffer``:
// ``writeBuffer`` allocates a same-size staging buffer inside the GPU
// process and copies through it. A 1.288 GB write means a 1.288 GB staging
// allocation in addition to the destination — same OOM trap. ``mappedAtCreation``
// hands us a renderer-process-mapped range, we ``set()`` into it, then
// ``unmap()`` transfers ownership to the GPU process — no double allocation.
//
// Why this returns Promise<void>: with u32 the ephemeral source is 2× the
// u16 size (1.288 GB vs 644 MB per axis), so two parallel calls would
// peak at 5.15 GB GPU residency — well past the ~4 GB Chrome budget. The
// caller awaits between X and Y so the X source is destroyed before the
// Y source allocates; that holds peak at 3.86 GB (same as the unsequenced
// u16 path used to). See ``EmbeddingRenderer.maybeRunUnpack``.

import type { Dataflow, Node } from "../dataflow.js";

import unpackShaderCode from "./unpack.wgsl?raw";

/** Bounds for u32 → f32 unpack: ``f32 = min + u32 * (max - min) / (2³² − 1)``. */
export interface CoordsBounds1D {
  min: number;
  max: number;
}

const UNPACK_WG_SIZE = 256;
// 2D-dispatch stride. WebGPU caps a single dispatch dimension at 65535,
// so the kernel walks ``id.y * STRIDE + id.x`` and we dispatch
// ``(STRIDE / wg_size, ceil(count / STRIDE))``. 65536 supports up to
// ~4 B points (65535 * 65536) before workgroups_y overflows. Must stay
// in sync with ``UNPACK_STRIDE`` in ``unpack.wgsl``.
const UNPACK_STRIDE = 65536;

// 2³² − 1 — the inverse-map denominator the shader's scale was computed
// against. Floats can represent 2³² exactly but not 2³² − 1; we keep the
// host arithmetic in f64 to avoid the float drift, then narrow to f32
// only when we encode the uniform.
const U32_MAX = 4_294_967_295;

/** Compile the unpack shader module. Cheap (<1 ms) and idempotent — the
 *  dataflow caches the result against ``device``. */
function makeUnpackModule(df: Dataflow, device: Node<GPUDevice>): Node<GPUShaderModule> {
  return df.derive([device], (device) => device.createShaderModule({ code: unpackShaderCode }));
}

/** Compile the unpack compute pipeline + bind group layout. The pipeline
 *  is reused across both axes and across data updates. */
export interface UnpackPipeline {
  pipeline: Node<GPUComputePipeline>;
  bindGroupLayout: Node<GPUBindGroupLayout>;
}

export function makeUnpackPipeline(df: Dataflow, device: Node<GPUDevice>): UnpackPipeline {
  const module = makeUnpackModule(df, device);
  const bindGroupLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    }),
  );
  const pipeline = df.derive([device, module, bindGroupLayout], (device, module, bindGroupLayout) =>
    device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: {
        module,
        entryPoint: "unpack",
        constants: { wg_unpack: UNPACK_WG_SIZE, UNPACK_STRIDE: UNPACK_STRIDE },
      },
    }),
  );
  return { pipeline, bindGroupLayout };
}

/** Encode + submit a one-shot compute pass that fills ``f32Dest`` from
 *  ``packed`` using the linear inverse-map advertised in ``bounds``.
 *
 *  Allocates the u32 source + uniform buffers fresh each call (via
 *  ``mappedAtCreation: true`` so no GPU-process staging copy) and
 *  defer-destroys them once the queue drains. The returned promise
 *  resolves *after* the destroy lands, so callers can chain
 *  ``await runUnpack(X); await runUnpack(Y)`` to keep peak GPU memory
 *  bounded — see file header. */
export async function runUnpack(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroupLayout: GPUBindGroupLayout,
  packed: Uint32Array,
  f32Dest: GPUBuffer,
  bounds: CoordsBounds1D,
): Promise<void> {
  const count = packed.length;
  // 4 bytes per u32. Min 4 to satisfy the WebGPU minimum binding size.
  const u32Bytes = Math.max(4, count * 4);
  const u32Buffer = device.createBuffer({
    size: u32Bytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint32Array(u32Buffer.getMappedRange()).set(packed);
  u32Buffer.unmap();

  // Uniforms: count u32, min f32, scale f32, chunk_offset_y u32 (16 bytes).
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const uniformView = new DataView(uniformBuffer.getMappedRange());
  uniformView.setUint32(0, count, /* littleEndian */ true);
  uniformView.setFloat32(4, bounds.min, true);
  // scale = (max - min) / (2^32 - 1). Encoded once on the host so the
  // shader is a single FMA per invocation. f64 host arithmetic keeps
  // the divisor exact (f32 cannot represent 2^32 − 1 precisely).
  const scale = (bounds.max - bounds.min) / U32_MAX;
  uniformView.setFloat32(8, scale, true);
  uniformView.setUint32(12, 0, true); // chunk_offset_y for the first chunk
  uniformBuffer.unmap();

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: u32Buffer } },
      { binding: 1, resource: { buffer: f32Dest } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  // Chunked dispatch. A single 322 M-thread dispatch trips the Metal
  // 5 s MTLCommandBuffer watchdog on cold loads — splitting it into
  // ``CHUNK_TARGET_THREADS``-sized cmd buffers keeps each well under
  // the wall-clock budget. The shader honours
  // ``params.chunk_offset_y`` so per-thread indices land in the right
  // slice of the destination buffer. 8 M (was 16 M) halves per-chunk
  // GPU time so a slow chunk under combined CPU/RAM contention still
  // clears the 5 s ceiling — paired with the same halving on
  // ``DRAW_CHUNK_INSTANCES`` in renderer.ts.
  const CHUNK_TARGET_THREADS = 8_000_000;
  const workgroupsX = UNPACK_STRIDE / UNPACK_WG_SIZE;
  const workgroupsY = Math.max(1, Math.ceil(count / UNPACK_STRIDE));
  const targetWorkgroupsPerChunk = Math.max(1, Math.floor(CHUNK_TARGET_THREADS / UNPACK_STRIDE));
  const numChunks = Math.max(1, Math.ceil(workgroupsY / targetWorkgroupsPerChunk));
  const chunkSizeY = Math.ceil(workgroupsY / numChunks);

  if (!(globalThis as any).__atlasUnpackDiagLogged) {
    (globalThis as any).__atlasUnpackDiagLogged = true;
    console.log(
      `[atlas-unpack-diag] count=${count} workgroupsY=${workgroupsY} chunkSizeY=${chunkSizeY} numChunks=${numChunks} bounds=[${bounds.min},${bounds.max}] u32Bytes=${u32Bytes}`,
    );
  }

  // Reusable 16-byte scratch for the per-chunk uniform write — only the
  // chunk_offset_y field changes between chunks.
  const uniformScratch = new ArrayBuffer(16);
  const scratchView = new DataView(uniformScratch);
  scratchView.setUint32(0, count, true);
  scratchView.setFloat32(4, bounds.min, true);
  scratchView.setFloat32(8, scale, true);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const offsetY = chunk * chunkSizeY;
    const remaining = Math.min(chunkSizeY, workgroupsY - offsetY);
    if (remaining <= 0) break;

    if (chunk > 0) {
      // First chunk's uniform was already populated via mappedAtCreation.
      scratchView.setUint32(12, offsetY, true);
      device.queue.writeBuffer(uniformBuffer, 0, uniformScratch);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, remaining);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // Defer-destroy: the in-flight dispatches still reference both buffers
  // until every submit completes. ``onSubmittedWorkDone`` resolves once
  // the queue drains — destroying earlier triggers the
  // ``MTLDevice``-poisoning fault that ``utils.ts:gpuBuffer`` already
  // works around for the persistent f32 destinations.
  await device.queue.onSubmittedWorkDone();
  u32Buffer.destroy();
  uniformBuffer.destroy();
}

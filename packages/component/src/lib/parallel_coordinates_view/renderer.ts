// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Dataflow, DataflowNode, DataflowValue } from "@embedding-atlas/utils";

import { StructWriter } from "../webgpu_renderer/uniforms.js";
import { gpuBuffer, gpuBufferData, gpuTexture } from "../webgpu_renderer/utils.js";

import binCode from "./bin.wgsl?raw";
import compositeCode from "./composite.wgsl?raw";
import ribbonCode from "./ribbon.wgsl?raw";

export interface ParallelCoordinatesRendererProps {
  /** Row-major, numRows * numFields floats in [0, 1] (0 = top, 1 = bottom). */
  values: Float32Array<ArrayBuffer>;
  /** Number of fields (axes). numRows is derived as values.length / numFields. */
  numFields: number;
  /** Per-row color value, normalized [0, 1], used to index the color LUT. */
  colorValue: Float32Array<ArrayBuffer>;
  /** Color lookup table: 256 * 4 RGBA bytes (sRGB). Only the first `colorLUTSize` entries are real. */
  colorLUT: Uint8Array<ArrayBuffer>;
  /** Number of real color LUT entries (1..256). */
  colorLUTSize: number;
  /** Color mapping mode: 0 = index (categorical; value is a LUT texel index), 1 = interpolate (continuous; value in [0,1]). */
  colorMode: number;
  /** Final opacity used directly by the render pipeline (the view applies any auto-opacity scaling). */
  opacity: number;
  /** Number of bins per axis. */
  binCount: number;
  /** Framebuffer size in device pixels. */
  width: number;
  height: number;
  /** Axis x positions in device pixels (length numFields). */
  axisXs: Float32Array<ArrayBuffer>;
  /** Plot extent in device pixels: value 0 -> plotY1 (top), value 1 -> plotY2 (bottom). */
  plotY1: number;
  plotY2: number;
  /** Background color in sRGB, each channel in [0, 1]. */
  backgroundColor: [number, number, number];
  gamma: number;
}

const SHARED_UNIFORM_SIZE = 64; // bytes
const PAIR_STRIDE = 256; // dynamic uniform offset alignment

interface TextureArrayState {
  texture?: GPUTexture;
  width?: number;
  height?: number;
  layers?: number;
  format?: GPUTextureFormat;
  usage?: GPUTextureUsageFlags;
  destroy?: () => void;
}

/** Stateful texture-array allocator (gpuTexture in utils.ts only supports single-layer textures). */
function gpuTextureArray(
  state: TextureArrayState,
  device: GPUDevice,
  width: number,
  height: number,
  layers: number,
  format: GPUTextureFormat,
  usage: GPUTextureUsageFlags,
): GPUTexture {
  if (
    state.texture == null ||
    state.width != width ||
    state.height != height ||
    state.layers != layers ||
    state.format != format ||
    state.usage != usage
  ) {
    state.texture?.destroy();
    state.texture = device.createTexture({ size: [width, height, layers], format: format, usage: usage });
    state.width = width;
    state.height = height;
    state.layers = layers;
    state.format = format;
    state.usage = usage;
    state.destroy = () => state.texture?.destroy();
  }
  return state.texture;
}

type BindGroupLayouts = {
  shared: GPUBindGroupLayout;
  binData: GPUBindGroupLayout;
  pair: GPUBindGroupLayout;
  ribbonData: GPUBindGroupLayout;
  composite: GPUBindGroupLayout;
};

export class ParallelCoordinatesRendererWebGPU {
  readonly props: ParallelCoordinatesRendererProps;

  private context: GPUCanvasContext;
  private df: Dataflow;
  private command: DataflowNode<(textureView: GPUTextureView) => void>;

  // Inputs that affect GPU resource allocation.
  private inValues: DataflowValue<Float32Array<ArrayBuffer>>;
  private inColorValue: DataflowValue<Float32Array<ArrayBuffer>>;
  private inColorLUT: DataflowValue<Uint8Array<ArrayBuffer>>;
  private inNumFields: DataflowValue<number>;
  private inBinCount: DataflowValue<number>;
  private inWidth: DataflowValue<number>;
  private inHeight: DataflowValue<number>;

  constructor(context: GPUCanvasContext, device: GPUDevice, format: GPUTextureFormat, width: number, height: number) {
    this.context = context;
    this.props = {
      values: new Float32Array(),
      numFields: 0,
      colorValue: new Float32Array(),
      colorLUT: new Uint8Array(256 * 4),
      colorLUTSize: 1,
      colorMode: 0,
      opacity: 0.8,
      binCount: 100,
      width: width,
      height: height,
      axisXs: new Float32Array(),
      plotY1: 0,
      plotY2: height,
      backgroundColor: [1, 1, 1],
      gamma: 2.2,
    };

    let df = new Dataflow();
    this.df = df;

    let vDevice = df.value(device);
    this.inValues = df.value(this.props.values);
    this.inColorValue = df.value(this.props.colorValue);
    this.inColorLUT = df.value(this.props.colorLUT);
    this.inNumFields = df.value(this.props.numFields);
    this.inBinCount = df.value(this.props.binCount);
    this.inWidth = df.value(width);
    this.inHeight = df.value(height);

    this.command = makeCommand(df, vDevice, format, () => this.props, {
      values: this.inValues,
      colorValue: this.inColorValue,
      colorLUT: this.inColorLUT,
      numFields: this.inNumFields,
      binCount: this.inBinCount,
      width: this.inWidth,
      height: this.inHeight,
    });
  }

  setProps(newProps: Partial<ParallelCoordinatesRendererProps>): boolean {
    let needsRender = false;
    let key: keyof ParallelCoordinatesRendererProps;
    for (key in newProps) {
      if (newProps[key] === this.props[key]) {
        continue;
      }
      (this.props as any)[key] = newProps[key];
      needsRender = true;
    }
    this.inValues.value = this.props.values;
    this.inColorValue.value = this.props.colorValue;
    this.inColorLUT.value = this.props.colorLUT;
    this.inNumFields.value = this.props.numFields;
    this.inBinCount.value = this.props.binCount;
    this.inWidth.value = this.props.width;
    this.inHeight.value = this.props.height;
    return needsRender;
  }

  render(): void {
    this.command.value(this.context.getCurrentTexture().createView());
  }

  destroy(): void {
    this.df.destroy();
  }
}

interface RenderInputs {
  values: DataflowValue<Float32Array<ArrayBuffer>>;
  colorValue: DataflowValue<Float32Array<ArrayBuffer>>;
  colorLUT: DataflowValue<Uint8Array<ArrayBuffer>>;
  numFields: DataflowValue<number>;
  binCount: DataflowValue<number>;
  width: DataflowValue<number>;
  height: DataflowValue<number>;
}

function makeCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  format: GPUTextureFormat,
  getProps: () => ParallelCoordinatesRendererProps,
  inputs: RenderInputs,
): DataflowNode<(textureView: GPUTextureView) => void> {
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

  // 32-bit float accumulation targets (exact counts, no f16 overflow). Additive blending into them needs
  // the float32-blendable feature, which the device is required to have. rgba32float is not filterable,
  // so its bind-group sample type must be "unfilterable-float".
  const accumFormat: GPUTextureFormat = "rgba32float";
  const accumSampleType: GPUTextureSampleType = "unfilterable-float";

  // ---- Shader modules (one per pass, so resource bindings don't collide) ----
  let binModule = df.derive([device], (device: GPUDevice) => device.createShaderModule({ code: binCode }));
  let ribbonModule = df.derive([device], (device: GPUDevice) => device.createShaderModule({ code: ribbonCode }));
  let compositeModule = df.derive([device], (device: GPUDevice) => device.createShaderModule({ code: compositeCode }));

  // ---- Derived sizes ----
  let count = df.derive([inputs.values, inputs.numFields], (v: Float32Array, nf: number) =>
    nf > 0 ? Math.floor(v.length / nf) : 0,
  );
  // One bin-texture layer per pair; pairs beyond the device's array-layer limit are not drawn.
  let pairCount = df.derive([device, inputs.numFields], (device: GPUDevice, nf: number) =>
    Math.min(Math.max(0, nf - 1), device.limits.maxTextureArrayLayers),
  );
  let layers = df.derive([pairCount], (p: number) => Math.max(1, p));

  // ---- Data buffers ----
  let valuesBuffer = df.statefulDerive(
    [
      device,
      df.statefulDerive([device, df.derive([inputs.values], (v: Float32Array) => v.byteLength), STORAGE], gpuBuffer),
      inputs.values,
    ],
    gpuBufferData,
  );
  let colorValueBuffer = df.statefulDerive(
    [
      device,
      df.statefulDerive(
        [device, df.derive([inputs.colorValue], (v: Float32Array) => v.byteLength), STORAGE],
        gpuBuffer,
      ),
      inputs.colorValue,
    ],
    gpuBufferData,
  );

  // ---- Color LUT texture (256 x 1, rgba8unorm) ----
  let lutTexture = df.statefulDerive(
    [device, 256, 1, "rgba8unorm" as GPUTextureFormat, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST],
    gpuTexture,
  );
  let lutWritten = df.derive(
    [device, lutTexture, inputs.colorLUT],
    (device: GPUDevice, texture: GPUTexture, lut: Uint8Array) => {
      device.queue.writeTexture({ texture: texture }, lut, { bytesPerRow: 256 * 4 }, { width: 256, height: 1 });
      return texture;
    },
  );

  // ---- Bin-space textures (one layer per pair) ----
  let binTexture = df.statefulDerive(
    [
      device,
      inputs.binCount,
      inputs.binCount,
      layers,
      accumFormat,
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    ],
    gpuTextureArray,
  );

  // ---- Accumulation framebuffer texture ----
  let oitUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  let colorTexture = df.statefulDerive([device, inputs.width, inputs.height, accumFormat, oitUsage], gpuTexture);

  // ---- Coverage framebuffer texture ----
  // Geometric edge coverage, accumulated additively but SEPARATELY from the density/count, so the
  // composite can apply anti-aliasing as a linear multiplier on top of the (saturated, capped-at-1) OIT
  // opacity. Without this, the AA coverage folded into the count gets crushed by the `1-exp` curve at high
  // opacity (opaque edges alias). Single channel, 32-bit like the color target.
  const coverageFormat: GPUTextureFormat = "r32float";
  let coverageTexture = df.statefulDerive([device, inputs.width, inputs.height, coverageFormat, oitUsage], gpuTexture);

  // ---- Uniform / small buffers ----
  let sharedUniform = df.statefulDerive([device, SHARED_UNIFORM_SIZE, UNIFORM], gpuBuffer);
  let pairUniform = df.statefulDerive(
    [device, df.derive([pairCount], (p: number) => Math.max(1, p) * PAIR_STRIDE), UNIFORM],
    gpuBuffer,
  );
  let axisXBuffer = df.statefulDerive(
    [device, df.derive([inputs.numFields], (nf: number) => Math.max(1, nf) * 4), STORAGE],
    gpuBuffer,
  );

  // ---- Bind group layouts ----
  let layouts = df.derive([device], (device: GPUDevice) => makeBindGroupLayouts(device, accumSampleType));

  // ---- Pipelines ----
  let binPipeline = df.derive(
    [device, binModule, layouts],
    (device: GPUDevice, module: GPUShaderModule, layouts: BindGroupLayouts) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [layouts.shared, layouts.binData, layouts.pair],
        }),
        vertex: { module: module, entryPoint: "bin_vs" },
        fragment: {
          module: module,
          entryPoint: "bin_fs",
          targets: [
            {
              format: accumFormat,
              blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
            },
          ],
        },
        primitive: { topology: "point-list" },
      }),
  );
  let ribbonPipeline = df.derive(
    [device, ribbonModule, layouts],
    (device: GPUDevice, module: GPUShaderModule, layouts: BindGroupLayouts) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.shared, layouts.ribbonData] }),
        vertex: { module: module, entryPoint: "ribbon_vs" },
        fragment: {
          module: module,
          entryPoint: "ribbon_fs",
          targets: [
            {
              format: accumFormat,
              blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
            },
            {
              format: coverageFormat,
              blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
            },
          ],
        },
        primitive: { topology: "triangle-list" },
      }),
  );
  let compositePipeline = df.derive(
    [device, compositeModule, layouts],
    (device: GPUDevice, module: GPUShaderModule, layouts: BindGroupLayouts) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layouts.shared, layouts.composite] }),
        vertex: { module: module, entryPoint: "composite_vs" },
        fragment: { module: module, entryPoint: "composite_fs", targets: [{ format: format }] },
        primitive: { topology: "triangle-list" },
      }),
  );

  // ---- Bind groups ----
  let bgShared = df.derive(
    [device, layouts, sharedUniform],
    (device: GPUDevice, layouts: BindGroupLayouts, buffer: GPUBuffer) =>
      device.createBindGroup({ layout: layouts.shared, entries: [{ binding: 0, resource: { buffer: buffer } }] }),
  );
  let bgBinData = df.derive(
    [device, layouts, valuesBuffer, colorValueBuffer, lutWritten],
    (device: GPUDevice, layouts: BindGroupLayouts, values: GPUBuffer, colorValue: GPUBuffer, lut: GPUTexture) =>
      device.createBindGroup({
        layout: layouts.binData,
        entries: [
          { binding: 0, resource: { buffer: values } },
          { binding: 1, resource: { buffer: colorValue } },
          { binding: 2, resource: lut.createView() },
        ],
      }),
  );
  let bgPair = df.derive(
    [device, layouts, pairUniform],
    (device: GPUDevice, layouts: BindGroupLayouts, buffer: GPUBuffer) =>
      device.createBindGroup({
        layout: layouts.pair,
        entries: [{ binding: 0, resource: { buffer: buffer, offset: 0, size: 16 } }],
      }),
  );
  let bgRibbonData = df.derive(
    [device, layouts, binTexture, axisXBuffer],
    (device: GPUDevice, layouts: BindGroupLayouts, binTexture: GPUTexture, axisXBuffer: GPUBuffer) =>
      device.createBindGroup({
        layout: layouts.ribbonData,
        entries: [
          { binding: 0, resource: binTexture.createView({ dimension: "2d-array" }) },
          { binding: 1, resource: { buffer: axisXBuffer } },
        ],
      }),
  );
  let bgComposite = df.derive(
    [device, layouts, colorTexture, coverageTexture],
    (device: GPUDevice, layouts: BindGroupLayouts, colorTexture: GPUTexture, coverageTexture: GPUTexture) =>
      device.createBindGroup({
        layout: layouts.composite,
        entries: [
          { binding: 0, resource: colorTexture.createView() },
          { binding: 1, resource: coverageTexture.createView() },
        ],
      }),
  );

  // CPU-side scratch buffers (grown on demand to the field count).
  let sharedScratch = new ArrayBuffer(SHARED_UNIFORM_SIZE);
  let pairScratchU32 = new Uint32Array(0);
  let axisScratch = new Float32Array(0);

  return df.derive(
    [
      device,
      count,
      pairCount,
      inputs.binCount,
      inputs.numFields,
      binPipeline,
      ribbonPipeline,
      compositePipeline,
      bgShared,
      bgBinData,
      bgPair,
      bgRibbonData,
      bgComposite,
      binTexture,
      colorTexture,
      coverageTexture,
      sharedUniform,
      pairUniform,
      axisXBuffer,
    ],
    (
      device: GPUDevice,
      count: number,
      pairCount: number,
      binCount: number,
      numFields: number,
      binPipeline: GPURenderPipeline,
      ribbonPipeline: GPURenderPipeline,
      compositePipeline: GPURenderPipeline,
      bgShared: GPUBindGroup,
      bgBinData: GPUBindGroup,
      bgPair: GPUBindGroup,
      bgRibbonData: GPUBindGroup,
      bgComposite: GPUBindGroup,
      binTexture: GPUTexture,
      colorTexture: GPUTexture,
      coverageTexture: GPUTexture,
      sharedUniform: GPUBuffer,
      pairUniform: GPUBuffer,
      axisXBuffer: GPUBuffer,
    ) =>
      (textureView: GPUTextureView) => {
        let props = getProps();

        // Shared uniform.
        let maxCount = Math.max(1, count / Math.max(1, binCount));
        let plotY1Ndc = 1 - (props.plotY1 / props.height) * 2;
        let plotY2Ndc = 1 - (props.plotY2 / props.height) * 2;
        // Linearize the sRGB background so the composite can mix it in linear space.
        let bg = props.backgroundColor;
        let g = props.gamma;
        let writer = new StructWriter(sharedScratch);
        writer.u32(binCount);
        writer.u32(numFields);
        writer.u32(count);
        writer.u32(Math.max(1, Math.min(256, props.colorLUTSize))); // lut_size
        writer.f32(props.opacity);
        writer.f32(maxCount);
        writer.f32(props.gamma);
        writer.u32(props.colorMode); // color_mode (0 = index, 1 = interpolate)
        writer.f32(plotY1Ndc);
        writer.f32(plotY2Ndc);
        writer.f32(props.width); // framebuffer width in device px (edge-feather AA)
        writer.f32(props.height); // framebuffer height in device px
        writer.vec4f(Math.pow(bg[0], g), Math.pow(bg[1], g), Math.pow(bg[2], g), 1);
        device.queue.writeBuffer(sharedUniform, 0, sharedScratch, 0, writer.byteOffset());

        // Per-pair uniform (dim_left at the start of each 256-byte block).
        if (pairScratchU32.length < (pairCount * PAIR_STRIDE) / 4) {
          pairScratchU32 = new Uint32Array((pairCount * PAIR_STRIDE) / 4);
        }
        for (let p = 0; p < pairCount; p++) {
          pairScratchU32[(p * PAIR_STRIDE) / 4] = p;
        }
        if (pairCount > 0) {
          device.queue.writeBuffer(pairUniform, 0, pairScratchU32.buffer, 0, pairCount * PAIR_STRIDE);
        }

        // Axis x positions in NDC.
        if (axisScratch.length < numFields) {
          axisScratch = new Float32Array(numFields);
        }
        for (let i = 0; i < numFields; i++) {
          axisScratch[i] = ((props.axisXs[i] ?? 0) / props.width) * 2 - 1;
        }
        if (numFields > 0) {
          device.queue.writeBuffer(axisXBuffer, 0, axisScratch.buffer, 0, numFields * 4);
        }

        let encoder = device.createCommandEncoder();

        // Pass 1: accumulate bin-space color blends, one render pass per pair (own array layer).
        for (let p = 0; p < pairCount; p++) {
          let view = binTexture.createView({ dimension: "2d", baseArrayLayer: p, arrayLayerCount: 1 });
          let pass = encoder.beginRenderPass({
            colorAttachments: [{ view: view, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
          });
          pass.setPipeline(binPipeline);
          pass.setBindGroup(0, bgShared);
          pass.setBindGroup(1, bgBinData);
          pass.setBindGroup(2, bgPair, [p * PAIR_STRIDE]);
          if (count > 0) {
            pass.draw(1, count);
          }
          pass.end();
        }

        // Pass 2: ribbons accumulate (Σ color, count) into the color target and geometric coverage into
        // a second target, both additively.
        {
          let pass = encoder.beginRenderPass({
            colorAttachments: [
              { view: colorTexture.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
              { view: coverageTexture.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
            ],
          });
          pass.setPipeline(ribbonPipeline);
          pass.setBindGroup(0, bgShared);
          pass.setBindGroup(1, bgRibbonData);
          let instances = pairCount * binCount * binCount;
          if (instances > 0 && count > 0) {
            pass.draw(6, instances);
          }
          pass.end();
        }

        // Pass 3: composite onto the canvas.
        {
          let pass = encoder.beginRenderPass({
            colorAttachments: [{ view: textureView, clearValue: [0, 0, 0, 1], loadOp: "clear", storeOp: "store" }],
          });
          pass.setPipeline(compositePipeline);
          pass.setBindGroup(0, bgShared);
          pass.setBindGroup(1, bgComposite);
          pass.draw(3);
          pass.end();
        }

        device.queue.submit([encoder.finish()]);
      },
  );
}

function makeBindGroupLayouts(device: GPUDevice, accumSampleType: GPUTextureSampleType): BindGroupLayouts {
  const { VERTEX, FRAGMENT } = GPUShaderStage;
  return {
    shared: device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: VERTEX | FRAGMENT, buffer: { type: "uniform" } }],
    }),
    binData: device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: VERTEX, texture: { sampleType: "float", viewDimension: "2d" } },
      ],
    }),
    pair: device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: VERTEX, buffer: { type: "uniform", hasDynamicOffset: true } }],
    }),
    ribbonData: device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VERTEX, texture: { sampleType: accumSampleType, viewDimension: "2d-array" } },
        { binding: 1, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      ],
    }),
    composite: device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: FRAGMENT, texture: { sampleType: accumSampleType, viewDimension: "2d" } },
        { binding: 1, visibility: FRAGMENT, texture: { sampleType: accumSampleType, viewDimension: "2d" } },
      ],
    }),
  };
}

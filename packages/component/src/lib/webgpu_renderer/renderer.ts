// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Dataflow, DataflowNode, DataflowValue } from "@embedding-atlas/utils";

import { defaultCategoryColors, parseColorNormalizedRgb } from "../colors.js";
import {
  matrix3_inverse,
  matrix3_matrix_mul_matrix,
  matrix3_vector_mul_matrix,
  type Matrix3,
  type Vector2,
  type Vector4,
} from "../matrix.js";
import type { Point, ViewportState } from "../utils.js";
import { Viewport } from "../viewport_utils.js";
import { makeModuleUniforms, type ModuleUniforms } from "./uniforms.js";
import { gpuBuffer, gpuBufferData, gpuTexture } from "./utils.js";

import { makeAccumulateCommand } from "./accumulate.js";
import { makeBindGroupLayouts, makeBindGroups } from "./bind_groups.js";
import { makeDownsampleCommand, makeDownsampleResources, type DownsampleConfig } from "./downsample.js";
import { makeDrawDensityMapCommand } from "./draw_density_map.js";
import { makeDrawPointsCommand, makeDrawPointsDownsampledCommand } from "./draw_points.js";
import { makeGammaCorrectionCommand } from "./gamma_correction.js";
import { makeGaussianBlurCommand } from "./gaussian_blur.js";
import { kdeConfig, type KDEConfig } from "./kde_config.js";

import programCode from "./program.wgsl?raw";

/** A data source (point coordinates and categories) uploaded to the GPU. Sources can be shared by multiple layers. */
export interface RendererSource {
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  category: Uint8Array<ArrayBuffer> | null;
}

/** A layer drawn by the renderer. */
export interface RendererLayer {
  /** A unique key for the layer. GPU resources are reused across updates with the same key. */
  key: string;
  type: "points" | "density";
  /** The key of the source in `EmbeddingRendererProps.sources`. */
  source: string;

  categoryCount: number;
  categoryColors: string[] | null;

  pointSize: number;
  pointAlpha: number;
  pointsAlpha: number;

  densityScaler: number;
  densityQuantizationStep: number;
  densityAlpha: number;
  contoursAlpha: number;

  /** Approximate maximum points to render. null/Infinity = no limit. */
  downsampleMaxPoints: number | null;
  /** Density weight for downsampling (0-10). */
  downsampleDensityWeight: number;
}

export interface EmbeddingRendererProps {
  colorScheme: "light" | "dark";

  viewportX: number;
  viewportY: number;
  viewportScale: number;

  densityBandwidth: number;

  gamma: number;
  width: number;
  height: number;

  sources: Record<string, RendererSource>;
  layers: RendererLayer[];
}

export interface DensityMap {
  data: Float32Array;
  width: number;
  height: number;
  coordinateAtPixel: (x: number, y: number) => Point;
}

interface SourceState {
  df: Dataflow;
  inputs: DataInputs;
  buffers: DataBuffers;
}

/** The shared density of a source with a given number of categories. */
interface DensityState {
  df: Dataflow;
  /** Whether any layer may need the density (see layerMayNeedDensity). If not, the buffers are not allocated. */
  enabled: DataflowValue<boolean>;
  buffers: DensityBuffers;
  encode: DataflowNode<(encoder: GPUCommandEncoder, frame: FrameParameters) => void>;
}

interface LayerState {
  type: RendererLayer["type"];
  source: SourceState;
  density: DensityState;
  df: Dataflow;
  inputs: LayerInputs;
  encode: DataflowNode<LayerEncoder>;
}

export class EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;

  private viewport: Viewport;
  private df: Dataflow;
  private device: DataflowNode<GPUDevice>;
  private module: DataflowNode<GPUShaderModule>;
  private context: GPUCanvasContext;
  private frame: FrameResources;
  private sources: Map<string, SourceState> = new Map();
  private layers: Map<string, LayerState> = new Map();
  /** Shared densities, keyed by source and category count. */
  private densities: Map<string, DensityState> = new Map();

  constructor(context: GPUCanvasContext, device: GPUDevice, format: GPUTextureFormat, width: number, height: number) {
    this.context = context;

    this.props = {
      colorScheme: "light",

      viewportX: 0,
      viewportY: 0,
      viewportScale: 1,

      densityBandwidth: 20,

      gamma: 2.2,
      width: width,
      height: height,

      sources: {},
      layers: [],
    };

    this.viewport = new Viewport({ x: 0, y: 0, scale: 1 }, width, height);

    this.df = new Dataflow();
    let df = this.df;
    this.device = df.value(device);
    this.module = df.derive([this.device], (device) => device.createShaderModule({ code: programCode }));
    this.frame = makeFrameResources(df, this.device, this.module, format, {
      width: df.value(width),
      height: df.value(height),
      densityBandwidth: df.value(this.props.densityBandwidth),
    });
  }

  /** Set renderer props. Returns true if a render is needed. */
  setProps(newProps: Partial<EmbeddingRendererProps>): boolean {
    let needsRender = false;
    let key: keyof EmbeddingRendererProps;
    for (key in newProps) {
      let value = newProps[key];
      if (key == "sources") {
        if (sourcesEqual(value as any, this.props.sources)) {
          continue;
        }
      } else if (key == "layers") {
        if (layersEqual(value as any, this.props.layers)) {
          continue;
        }
      } else if (value === this.props[key]) {
        continue;
      }
      (this.props as any)[key] = value;
      needsRender = true;
    }
    this.viewport.update(
      { x: this.props.viewportX, y: this.props.viewportY, scale: this.props.viewportScale },
      this.props.width,
      this.props.height,
    );
    this.frame.inputs.width.value = this.props.width;
    this.frame.inputs.height.value = this.props.height;
    this.frame.inputs.densityBandwidth.value = this.props.densityBandwidth;
    // Add and update sources first, then layers (and their densities), and destroy removed sources last,
    // so resources are always destroyed after the resources that depend on them.
    this.syncSources();
    this.syncLayers();
    this.removeStaleSources();
    return needsRender;
  }

  private syncSources() {
    let sources = this.props.sources;
    for (let key in sources) {
      let source = sources[key];
      let state = this.sources.get(key);
      if (state == null) {
        let df = this.df.subgraph();
        let inputs: DataInputs = {
          xData: df.value(source.x),
          yData: df.value(source.y),
          categoryData: df.value(source.category),
        };
        state = { df, inputs, buffers: makeDataBuffers(df, this.device, inputs) };
        this.sources.set(key, state);
      } else {
        state.inputs.xData.value = source.x;
        state.inputs.yData.value = source.y;
        state.inputs.categoryData.value = source.category;
      }
    }
  }

  /** Destroy sources that are no longer in the props (after the layers and densities using them). */
  private removeStaleSources() {
    for (let [key, state] of this.sources) {
      if (!(key in this.props.sources)) {
        state.df.destroy();
        this.sources.delete(key);
      }
    }
  }

  private syncLayers() {
    let visited = new Set<string>();
    let usedDensities = new Set<string>();
    let enabledDensities = new Set<string>();
    for (let layer of this.props.layers) {
      // Only sources in the current props (removed sources are destroyed after this, see setProps).
      let source = layer.source in this.props.sources ? this.sources.get(layer.source) : undefined;
      let categoryCount = source != null && this.props.sources[layer.source].category != null ? layer.categoryCount : 1;
      let densityKey = `${layer.source}#${categoryCount}`;
      let density = source != null ? this.densityFor(densityKey, source, categoryCount) : null;
      let state = this.layers.get(layer.key);
      if (
        state != null &&
        (density == null || state.source !== source || state.density !== density || state.type != layer.type)
      ) {
        state.df.destroy();
        this.layers.delete(layer.key);
        state = undefined;
      }
      if (source == null || density == null) {
        continue;
      }
      visited.add(layer.key);
      usedDensities.add(densityKey);
      if (layerMayNeedDensity(layer, source.buffers.count.value)) {
        enabledDensities.add(densityKey);
      }
      if (state == null) {
        let df = this.df.subgraph();
        let inputs: LayerInputs = {
          categoryCount: df.value(categoryCount),
          categoryColors: df.value(layer.categoryColors),
          downsampleMaxPoints: df.value(layer.downsampleMaxPoints),
        };
        let make = layer.type == "points" ? makePointsLayerCommand : makeDensityLayerCommand;
        let encode = make(df, this.device, this.module, this.frame, source.buffers, density.buffers, inputs);
        state = { type: layer.type, source, density, df, inputs, encode };
        this.layers.set(layer.key, state);
      } else {
        state.inputs.categoryColors.value = layer.categoryColors;
        state.inputs.downsampleMaxPoints.value = layer.downsampleMaxPoints;
      }
    }
    for (let [key, state] of this.layers) {
      if (!visited.has(key)) {
        state.df.destroy();
        this.layers.delete(key);
      }
    }
    // Destroy densities after the layers that use them.
    for (let [key, density] of this.densities) {
      if (!usedDensities.has(key)) {
        density.df.destroy();
        this.densities.delete(key);
      } else {
        density.enabled.value = enabledDensities.has(key);
      }
    }
  }

  private densityFor(key: string, source: SourceState, categoryCount: number): DensityState {
    // Densities are keyed by source key, and are destroyed (in syncLayers) before their source
    // (in removeStaleSources), so an existing density always belongs to the current source.
    let density = this.densities.get(key);
    if (density == null) {
      let df = this.df.subgraph();
      let enabled = df.value(false);
      let { buffers, encode } = makeSharedDensityCommand(
        df,
        this.device,
        this.module,
        this.frame,
        source.buffers,
        df.value(categoryCount),
        enabled,
      );
      density = { df, enabled, buffers, encode };
      this.densities.set(key, density);
    }
    return density;
  }

  /** Render */
  render(): void {
    let props = this.props;
    let device = this.device.value;
    let fbWidth = this.frame.fbWidth.value;
    let fbHeight = this.frame.fbHeight.value;
    let scalerX = props.width / fbWidth;
    let scalerY = props.height / fbHeight;
    let safeMarginAdjustmentMatrix: Matrix3 = [scalerX, 0, 0, 0, scalerY, 0, 0, 0, 1];
    let frame: FrameParameters = {
      matrix: matrix3_matrix_mul_matrix(safeMarginAdjustmentMatrix, this.viewport.matrix()),
      viewXYScaler: [1 / scalerX, 1 / scalerY],
      backgroundColor: props.colorScheme == "light" ? [1, 1, 1, 1] : [0, 0, 0, 1],
      gamma: props.gamma,
    };

    this.frame.uniforms.update.value({
      ...emptyUniforms(),
      framebuffer_width: fbWidth,
      framebuffer_height: fbHeight,
      gamma: frame.gamma,
      view_xy_scaler: frame.viewXYScaler,
      background_color: frame.backgroundColor,
    });

    let encoder = device.createCommandEncoder();
    this.frame.clear.value(encoder);
    // Estimate each shared density once, if any layer needs it in this frame.
    let densities = new Set<DensityState>();
    for (let layer of props.layers) {
      let state = this.layers.get(layer.key);
      if (state != null && layerNeedsDensity(layer, state.source.buffers.count.value)) {
        densities.add(state.density);
      }
    }
    for (let density of densities) {
      density.encode.value(encoder, frame);
    }
    // Layers accumulate into the shared framebuffer with order-independent blending
    // (sum of premultiplied colors and sum of log(1 - alpha)), so draw order doesn't matter.
    for (let layer of props.layers) {
      this.layers.get(layer.key)?.encode.value(encoder, layer, frame);
    }
    this.frame.gammaCorrection.value(encoder, this.context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
  }

  /** Destroy the renderer and free any resource */
  destroy(): void {
    this.sources.clear();
    this.layers.clear();
    this.densities.clear();
    this.df.destroy();
  }

  /** Produce a density map of points. */
  async densityMap(
    data: { x: Float32Array<ArrayBuffer>; y: Float32Array<ArrayBuffer> },
    width: number,
    height: number,
    radius: number,
    viewportState: ViewportState,
  ): Promise<DensityMap> {
    let subgraph = this.df.subgraph();
    // Use the GPU buffers of a source that holds exactly these arrays, otherwise upload them temporarily.
    // The buffers are read when the command is encoded, synchronously below, so they hold these arrays.
    let buffers: DataBuffers | null = null;
    for (let state of this.sources.values()) {
      if (state.inputs.xData.value === data.x && state.inputs.yData.value === data.y) {
        buffers = state.buffers;
        break;
      }
    }
    if (buffers == null) {
      buffers = makeDataBuffers(subgraph, this.device, {
        xData: subgraph.value(data.x),
        yData: subgraph.value(data.y),
        categoryData: subgraph.value(null),
      });
    }
    let { x, y, scale: s } = viewportState;
    let positionMatrix: Matrix3 = [s, 0, 0, 0, s, 0, -x * s, -y * s, 1];
    let inv_matrix = matrix3_inverse(positionMatrix);
    let cmd = makeDensityMapCommand(
      subgraph,
      this.device,
      this.module,
      makeModuleUniforms(subgraph, this.device),
      subgraph.value(width),
      subgraph.value(height),
      subgraph.value(radius),
      subgraph.value(positionMatrix),
      buffers,
    );
    let result = await cmd.value();
    subgraph.destroy();
    return {
      data: result,
      width: width,
      height: height,
      coordinateAtPixel: (x: number, y: number) => {
        let tx = (x / width) * 2 - 1;
        let ty = (y / height) * 2 - 1;
        let r = matrix3_vector_mul_matrix([tx, ty, 1], inv_matrix);
        return { x: r[0], y: r[1] };
      },
    };
  }
}

function sourcesEqual(a: Record<string, RendererSource>, b: Record<string, RendererSource>): boolean {
  let aKeys = Object.keys(a);
  if (aKeys.length != Object.keys(b).length) {
    return false;
  }
  for (let key of aKeys) {
    let sa = a[key];
    let sb = b[key];
    if (sb == null || sa.x !== sb.x || sa.y !== sb.y || sa.category !== sb.category) {
      return false;
    }
  }
  return true;
}

function layersEqual(a: RendererLayer[], b: RendererLayer[]): boolean {
  if (a.length != b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    let la = a[i] as any;
    let lb = b[i] as any;
    for (let key in la) {
      if (la[key] !== lb[key]) {
        return false;
      }
    }
  }
  return true;
}

interface FrameParameters {
  matrix: Matrix3;
  viewXYScaler: Vector2;
  backgroundColor: Vector4;
  gamma: number;
}

type LayerEncoder = (encoder: GPUCommandEncoder, layer: RendererLayer, frame: FrameParameters) => void;

export interface DataInputs {
  xData: DataflowValue<Float32Array<ArrayBuffer>>;
  yData: DataflowValue<Float32Array<ArrayBuffer>>;
  categoryData: DataflowValue<Uint8Array<ArrayBuffer> | null>;
}

interface LayerInputs {
  categoryCount: DataflowValue<number>;
  categoryColors: DataflowValue<string[] | null>;
  downsampleMaxPoints: DataflowValue<number | null>;
}

export interface DataBuffers {
  x: DataflowNode<GPUBuffer>;
  y: DataflowNode<GPUBuffer>;
  category: DataflowNode<GPUBuffer | null>;
  count: DataflowNode<number>;
}

export interface FramebufferTextures {
  colorTexture: DataflowNode<GPUTexture>;
  colorTextureFormat: GPUTextureFormat;
  alphaTexture: DataflowNode<GPUTexture>;
  alphaTextureFormat: GPUTextureFormat;
}

export interface DensityBuffers {
  countBuffer: DataflowNode<GPUBuffer>;
  blurBuffer: DataflowNode<GPUBuffer>;
}

export interface AuxiliaryResources extends FramebufferTextures, DensityBuffers {}

interface FrameResources {
  inputs: {
    width: DataflowValue<number>;
    height: DataflowValue<number>;
    densityBandwidth: DataflowValue<number>;
  };
  fbWidth: DataflowNode<number>;
  fbHeight: DataflowNode<number>;
  densityWidth: DataflowNode<number>;
  densityHeight: DataflowNode<number>;
  densityPixelRatio: number;
  kdeCoeffs: DataflowNode<KDEConfig>;
  textures: FramebufferTextures;
  uniforms: ModuleUniforms;
  clear: DataflowNode<(encoder: GPUCommandEncoder) => void>;
  gammaCorrection: DataflowNode<(encoder: GPUCommandEncoder, textureView: GPUTextureView) => void>;
}

function makeDataBuffers(df: Dataflow, device: DataflowNode<GPUDevice>, inputs: DataInputs): DataBuffers {
  let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const count = df.derive([inputs.xData], (d: ArrayLike<number>) => d.length);
  const xyDataSize = df.derive([count], (c) => c * 4);
  const categoryDataSize = count;
  const xBuffer = df.statefulDerive(
    [device, df.statefulDerive([device, xyDataSize, usage], gpuBuffer), inputs.xData],
    gpuBufferData,
  );
  const yBuffer = df.statefulDerive(
    [device, df.statefulDerive([device, xyDataSize, usage], gpuBuffer), inputs.yData],
    gpuBufferData,
  );
  const categoryBuffer = df.statefulDerive(
    [device, df.statefulDerive([device, categoryDataSize, usage], gpuBuffer), inputs.categoryData],
    gpuBufferData,
  );
  return { x: xBuffer, y: yBuffer, category: categoryBuffer, count: count };
}

function makeFramebufferTextures(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  framebufferWidth: DataflowNode<number>,
  framebufferHeight: DataflowNode<number>,
): FramebufferTextures {
  let colorTextureFormat: GPUTextureFormat = "rgba16float";
  let alphaTextureFormat: GPUTextureFormat = "r16float";
  let usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  let colorTexture = df.statefulDerive(
    [device, framebufferWidth, framebufferHeight, colorTextureFormat, usage],
    gpuTexture,
  );
  let alphaTexture = df.statefulDerive(
    [device, framebufferWidth, framebufferHeight, alphaTextureFormat, usage],
    gpuTexture,
  );
  return {
    colorTexture: colorTexture,
    alphaTexture: alphaTexture,
    colorTextureFormat: colorTextureFormat,
    alphaTextureFormat: alphaTextureFormat,
  };
}

function makeDensityBuffers(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  densityWidth: DataflowNode<number>,
  densityHeight: DataflowNode<number>,
  categoryCount: DataflowNode<number>,
  enabled: DataflowNode<boolean> | null = null,
): DensityBuffers {
  // When disabled, use minimal placeholder buffers (so bind groups stay valid).
  let enabledNode = enabled ?? df.value(true);
  let countBufferSize = df.derive(
    [densityWidth, densityHeight, categoryCount, enabledNode],
    (w: number, h: number, c: number, e: boolean) => (e ? w * h * c * 4 : 0), // w * h * categoryCount * sizeof(uint32)
  );
  let blurBufferSize = df.derive(
    [densityWidth, densityHeight, categoryCount, enabledNode],
    (w: number, h: number, c: number, e: boolean) => (e ? w * h * c * 2 : 0), // w * h * categoryCount * sizeof(f16)
  );
  let countBuffer = df.statefulDerive(
    [device, countBufferSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC],
    gpuBuffer,
  );
  let blurBuffer = df.statefulDerive([device, blurBufferSize, GPUBufferUsage.STORAGE], gpuBuffer);
  return { countBuffer: countBuffer, blurBuffer: blurBuffer };
}

function makeAuxiliaryResources(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  framebufferWidth: DataflowNode<number>,
  framebufferHeight: DataflowNode<number>,
  densityWidth: DataflowNode<number>,
  densityHeight: DataflowNode<number>,
  categoryCount: DataflowNode<number>,
): AuxiliaryResources {
  return {
    ...makeFramebufferTextures(df, device, framebufferWidth, framebufferHeight),
    ...makeDensityBuffers(df, device, densityWidth, densityHeight, categoryCount),
  };
}

function makeFrameResources(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  format: GPUTextureFormat,
  inputs: FrameResources["inputs"],
): FrameResources {
  const densityPixelRatio = 4;
  let safeMargin = df.derive([inputs.densityBandwidth], (r: number) => Math.ceil(r * 3) + 1);
  let fbWidth = df.derive([inputs.width, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  let fbHeight = df.derive([inputs.height, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  let densityWidth = df.derive([fbWidth], (x) => Math.ceil(x / densityPixelRatio));
  let densityHeight = df.derive([fbHeight], (x) => Math.ceil(x / densityPixelRatio));
  let kdeCoeffs = df.derive(
    [inputs.densityBandwidth, fbWidth, densityWidth],
    (bandwidth: number, fbWidth: number, densityWidth: number) => kdeConfig((bandwidth / fbWidth) * densityWidth),
  );

  let textures = makeFramebufferTextures(df, device, fbWidth, fbHeight);
  let uniforms = makeModuleUniforms(df, device);

  let layouts = df.derive([device], (device) => makeBindGroupLayouts(device));
  let group0Layout = df.derive([layouts], (l) => l.group0);
  let group3Layout = df.derive([layouts], (l) => l.group3);
  let group0 = df.derive([device, group0Layout, uniforms.buffer], (device, layout, uniformBuffer) =>
    device.createBindGroup({ layout: layout, entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] }),
  );
  let group3 = df.derive(
    [device, group3Layout, textures.colorTexture, textures.alphaTexture],
    (device, layout, colorTexture, alphaTexture) =>
      device.createBindGroup({
        layout: layout,
        entries: [
          { binding: 0, resource: device.createSampler({}) },
          { binding: 1, resource: colorTexture.createView() },
          { binding: 2, resource: alphaTexture.createView() },
        ],
      }),
  );

  let clear = df.derive(
    [textures.colorTexture, textures.alphaTexture],
    (colorTexture, alphaTexture) => (encoder: GPUCommandEncoder) => {
      let pass = encoder.beginRenderPass({
        colorAttachments: [
          { clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store", view: colorTexture.createView() },
          { clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store", view: alphaTexture.createView() },
        ],
      });
      pass.end();
    },
  );

  let gammaCorrection = makeGammaCorrectionCommand(
    df,
    device,
    module,
    format,
    group0Layout,
    group3Layout,
    group0,
    group3,
  );

  return {
    inputs,
    fbWidth,
    fbHeight,
    densityWidth,
    densityHeight,
    densityPixelRatio,
    kdeCoeffs,
    textures,
    uniforms,
    clear,
    gammaCorrection,
  };
}

function emptyUniforms() {
  return {
    count: 0,
    category_count: 1,
    framebuffer_width: 0,
    framebuffer_height: 0,
    density_width: 0,
    density_height: 0,
    gamma: 1,
    point_size: 0,
    point_alpha: 0,
    points_alpha: 0,
    density_scaler: 0,
    quantization_step: 0,
    density_alpha: 0,
    contours_alpha: 0,
    matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1] as Matrix3,
    view_xy_scaler: [1, 1] as Vector2,
    kde_causal: [0, 0, 0, 0] as Vector4,
    kde_anticausal: [0, 0, 0, 0] as Vector4,
    kde_a: [0, 0, 0, 0] as Vector4,
    background_color: [0, 0, 0, 0] as Vector4,
    category_colors: [],
  };
}

/** The density of a source, estimated once per frame and shared by the layers that need it
 *  (density layers, and points layers that downsample). */
function makeSharedDensityCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  frame: FrameResources,
  dataBuffers: DataBuffers,
  categoryCount: DataflowNode<number>,
  enabled: DataflowNode<boolean>,
): { buffers: DensityBuffers; encode: DataflowNode<(encoder: GPUCommandEncoder, frame: FrameParameters) => void> } {
  let uniforms = makeModuleUniforms(df, device);
  let buffers = makeDensityBuffers(df, device, frame.densityWidth, frame.densityHeight, categoryCount, enabled);
  let auxiliaryResources: AuxiliaryResources = { ...frame.textures, ...buffers };
  let bindGroups = makeBindGroups(df, device, uniforms.buffer, dataBuffers, auxiliaryResources);
  let accumulate = makeAccumulateCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let gaussianBlur = makeGaussianBlurCommand(
    df,
    device,
    module,
    bindGroups,
    frame.fbWidth,
    frame.fbHeight,
    categoryCount,
  );
  let encode = df.derive(
    [
      uniforms.update,
      frame.fbWidth,
      frame.fbHeight,
      frame.densityWidth,
      frame.densityHeight,
      frame.kdeCoeffs,
      dataBuffers.count,
      categoryCount,
      accumulate,
      gaussianBlur,
    ],
    (update, fbWidth, fbHeight, densityWidth, densityHeight, kde, count, categoryCount, accumulate, gaussianBlur) =>
      (encoder: GPUCommandEncoder, frameParams: FrameParameters) => {
        update({
          ...emptyUniforms(),
          count: count,
          category_count: categoryCount,
          framebuffer_width: fbWidth,
          framebuffer_height: fbHeight,
          density_width: densityWidth,
          density_height: densityHeight,
          matrix: frameParams.matrix,
          kde_causal: kde.kde_causal,
          kde_anticausal: kde.kde_anticausal,
          kde_a: kde.kde_a,
        });
        accumulate(encoder);
        gaussianBlur(encoder);
      },
  );
  return { buffers, encode };
}

/** Resources shared by points and density layers: uniforms and bind groups (reading the shared density). */
function makeLayerResources(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  frame: FrameResources,
  dataBuffers: DataBuffers,
  densityBuffers: DensityBuffers,
  inputs: LayerInputs,
) {
  let uniforms = makeModuleUniforms(df, device);
  let auxiliaryResources: AuxiliaryResources = { ...frame.textures, ...densityBuffers };
  let bindGroups = makeBindGroups(df, device, uniforms.buffer, dataBuffers, auxiliaryResources);
  let categoryColors = df.derive(
    [inputs.categoryColors, inputs.categoryCount],
    (colors: string[] | null, count: number) => {
      if (colors == null) {
        colors = defaultCategoryColors(count);
      }
      return colors.map((x) => parseColorNormalizedRgb(x));
    },
  );
  let updateUniforms = df.derive(
    [
      uniforms.update,
      frame.fbWidth,
      frame.fbHeight,
      frame.densityWidth,
      frame.densityHeight,
      frame.kdeCoeffs,
      dataBuffers.count,
      inputs.categoryCount,
      categoryColors,
    ],
    (update, fbWidth, fbHeight, densityWidth, densityHeight, kde, count, categoryCount: number, categoryColors) =>
      (layer: RendererLayer, frameParams: FrameParameters) => {
        update({
          count: count,
          category_count: categoryCount,
          framebuffer_width: fbWidth,
          framebuffer_height: fbHeight,
          density_width: densityWidth,
          density_height: densityHeight,
          gamma: frameParams.gamma,
          point_size: layer.pointSize,
          point_alpha: layer.pointAlpha,
          points_alpha: layer.pointsAlpha,
          density_scaler: layer.densityScaler / (frame.densityPixelRatio * frame.densityPixelRatio),
          quantization_step: layer.densityQuantizationStep,
          density_alpha: layer.densityAlpha,
          contours_alpha: layer.contoursAlpha,
          matrix: frameParams.matrix,
          view_xy_scaler: frameParams.viewXYScaler,
          kde_causal: kde.kde_causal,
          kde_anticausal: kde.kde_anticausal,
          kde_a: kde.kde_a,
          background_color: frameParams.backgroundColor,
          category_colors: categoryColors,
        });
      },
  );
  return { auxiliaryResources, bindGroups, updateUniforms };
}

/** The effective max points of a points layer, or null if downsampling is disabled.
 *  null, Infinity, or invalid values (<=0, NaN) disable downsampling. */
function effectiveMaxPoints(layer: RendererLayer): number | null {
  const maxPoints = layer.downsampleMaxPoints;
  return maxPoints === null || !Number.isFinite(maxPoints) || maxPoints <= 0 ? null : maxPoints;
}

/** Whether a layer may need its source's density, regardless of its per-frame opacity
 *  (so the density buffers aren't reallocated when, e.g., zooming changes a density layer's alpha). */
function layerMayNeedDensity(layer: RendererLayer, count: number): boolean {
  return layer.type == "density" || layerNeedsDensity(layer, count);
}

/** Whether a layer needs its source's density in this frame. */
function layerNeedsDensity(layer: RendererLayer, count: number): boolean {
  if (layer.type == "density") {
    return layer.densityAlpha > 0 || layer.contoursAlpha > 0;
  }
  // Points layers use the density for density-based downsampling.
  let maxPoints = effectiveMaxPoints(layer);
  return maxPoints !== null && count > maxPoints;
}

function makePointsLayerCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  frame: FrameResources,
  dataBuffers: DataBuffers,
  densityBuffers: DensityBuffers,
  inputs: LayerInputs,
): DataflowNode<LayerEncoder> {
  let { auxiliaryResources, bindGroups, updateUniforms } = makeLayerResources(
    df,
    device,
    frame,
    dataBuffers,
    densityBuffers,
    inputs,
  );

  let downsampleResources = makeDownsampleResources(df, device, dataBuffers.count, inputs.downsampleMaxPoints);
  let drawPoints = makeDrawPointsCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let drawPointsDownsampled = makeDrawPointsDownsampledCommand(
    df,
    device,
    module,
    bindGroups,
    downsampleResources,
    auxiliaryResources,
  );
  let downsample = makeDownsampleCommand(
    df,
    device,
    module,
    df.derive([bindGroups.layouts], (l) => l.group0),
    df.derive([bindGroups.layouts], (l) => l.group1),
    auxiliaryResources.blurBuffer, // The shared density, for density-based sampling
    bindGroups.group0,
    bindGroups.group1,
    downsampleResources,
    dataBuffers,
  );

  return df.derive(
    [updateUniforms, dataBuffers.count, drawPoints, drawPointsDownsampled, downsample],
    (updateUniforms, count, drawPoints, drawPointsDownsampled, downsample) => (encoder, layer, frameParams) => {
      updateUniforms(layer, frameParams);

      if (layerNeedsDensity(layer, count)) {
        // The shared density was computed before the layers are drawn (see `render`).
        // Run downsampling pipeline with fixed seed for deterministic sampling
        // Using 42 ensures the same points are always accepted/rejected
        // Viewport culling handles which points are visible
        const downsampleConfig: DownsampleConfig = {
          maxPoints: effectiveMaxPoints(layer)!,
          densityWeight: layer.downsampleDensityWeight,
          frameSeed: 42,
        };

        // Perform downsample, this fills the point_data buffer.
        downsample(encoder, downsampleConfig);

        // Draw downsampled points
        drawPointsDownsampled(encoder, count);
      } else {
        drawPoints(encoder);
      }
    },
  );
}

function makeDensityLayerCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  frame: FrameResources,
  dataBuffers: DataBuffers,
  densityBuffers: DensityBuffers,
  inputs: LayerInputs,
): DataflowNode<LayerEncoder> {
  let { auxiliaryResources, bindGroups, updateUniforms } = makeLayerResources(
    df,
    device,
    frame,
    dataBuffers,
    densityBuffers,
    inputs,
  );
  let drawDensityMap = makeDrawDensityMapCommand(df, device, module, bindGroups, auxiliaryResources);

  return df.derive(
    [updateUniforms, drawDensityMap],
    (updateUniforms, drawDensityMap) => (encoder, layer, frameParams) => {
      if (!layerNeedsDensity(layer, 0)) {
        return;
      }
      // The shared density was computed before the layers are drawn (see `render`).
      updateUniforms(layer, frameParams);
      drawDensityMap(encoder);
    },
  );
}

function makeDensityMapCommand(
  df: Dataflow,
  device: DataflowNode<GPUDevice>,
  module: DataflowNode<GPUShaderModule>,
  uniforms: ModuleUniforms,
  width: DataflowNode<number>,
  height: DataflowNode<number>,
  radius: DataflowNode<number>,
  matrix: DataflowNode<Matrix3>,
  dataBuffers: DataBuffers,
): DataflowNode<() => Promise<Float32Array>> {
  let auxiliaryResources = makeAuxiliaryResources(df, device, width, height, width, height, df.value(1));
  let bindGroups = makeBindGroups(df, device, uniforms.buffer, dataBuffers, auxiliaryResources);
  let accumulate = makeAccumulateCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let gaussianBlur = makeGaussianBlurCommand(df, device, module, bindGroups, width, height, df.value(1));

  return df.derive(
    [
      device,
      width,
      height,
      dataBuffers.count,
      uniforms.update,
      radius,
      matrix,
      accumulate,
      gaussianBlur,
      auxiliaryResources.countBuffer,
    ],
    (device, width, height, count, updateUniforms, radius, matrix, accumulate, gaussianBlur, countBuffer) => () => {
      let encoder = device.createCommandEncoder();
      let kde_coeffs = kdeConfig(radius);
      updateUniforms({
        ...emptyUniforms(),
        count: count,
        category_count: 1,
        framebuffer_width: width,
        framebuffer_height: height,
        density_width: width,
        density_height: height,
        matrix: matrix,
        kde_causal: kde_coeffs.kde_causal,
        kde_anticausal: kde_coeffs.kde_anticausal,
        kde_a: kde_coeffs.kde_a,
      });
      accumulate(encoder);
      gaussianBlur(encoder);

      let outputBuffer = device.createBuffer({
        size: width * height * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(countBuffer, 0, outputBuffer, 0, width * height * 2);
      device.queue.submit([encoder.finish()]);
      return outputBuffer.mapAsync(GPUMapMode.READ, 0, width * height * 2).then(() => {
        return convertFloat16ToFloat32Array(outputBuffer.getMappedRange());
      });
    },
  );
}

function convertFloat16ToFloat32Array(inputs: ArrayBuffer): Float32Array {
  let view = new Uint16Array(inputs);
  let result = new Uint32Array(view.length);
  for (let i = 0; i < view.length; i++) {
    let t1 = view[i] & 0x7fff;
    let t2 = view[i] & 0x8000;
    let t3 = view[i] & 0x7c00;
    t1 <<= 13;
    t2 <<= 16;
    t1 += 0x38000000;
    t1 = t3 == 0 ? 0 : t1;
    t1 |= t2;
    result[i] = t1;
  }
  return new Float32Array(result.buffer);
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { defaultCategoryColors, parseColorNormalizedRgb } from "../colors.js";
import { Dataflow, Node, ValueNode } from "../dataflow.js";
import {
  matrix3_identity,
  matrix3_inverse,
  matrix3_matrix_mul_matrix,
  matrix3_vector_mul_matrix,
  type Matrix3,
  type Vector4,
} from "../matrix.js";
import type { Point, ViewportState } from "../utils.js";
import { Viewport } from "../viewport_utils.js";
import { makeModuleUniforms, type ModuleUniforms } from "./uniforms.js";
import { gpuBuffer, gpuBufferData, gpuTexture } from "./utils.js";

import { makeAccumulateCommand } from "./accumulate.js";
import { makeBindGroups } from "./bind_groups.js";
import { makeDownsampleCommand, makeDownsampleResources, type DownsampleConfig } from "./downsample.js";
import { makeDrawDensityMapCommand } from "./draw_density_map.js";
import {
  makeDrawPointsCommand,
  makeDrawPointsCompactedCommand,
  makeDrawPointsDownsampledCommand,
} from "./draw_points.js";
import { makeGammaCorrectionCommand } from "./gamma_correction.js";
import { makeGaussianBlurCommand } from "./gaussian_blur.js";
import { kdeConfig } from "./kde_config.js";
import { makeUnpackPipeline, runUnpack, type UnpackPipeline } from "./unpack.js";

import programCode from "./program.wgsl?raw";

export type RenderMode = "points" | "density";

export interface EmbeddingRendererProps {
  mode: RenderMode;
  colorScheme: "light" | "dark";

  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  /** Optional u32-packed coordinate input. When set, `x`/`y` are ignored. */
  xPacked?: Uint32Array<ArrayBuffer> | null;
  yPacked?: Uint32Array<ArrayBuffer> | null;
  coordsBoundsX?: [number, number] | null;
  coordsBoundsY?: [number, number] | null;
  category: Uint8Array<ArrayBuffer> | null;

  categoryCount: number;
  categoryColors: string[] | null;
  viewportX: number;
  viewportY: number;
  viewportScale: number;
  pointSize: number;
  pointAlpha: number;
  pointsAlpha: number;
  densityScaler: number;
  densityBandwidth: number;
  densityQuantizationStep: number;
  densityAlpha: number;
  contoursAlpha: number;
  gamma: number;
  width: number;
  height: number;
  downsampleMaxPoints: number | null;
  downsampleDensityWeight: number;
  isGis: boolean;
  skipDownsampleCompute?: boolean;
}

export interface DensityMap {
  data: Float32Array;
  width: number;
  height: number;
  coordinateAtPixel: (x: number, y: number) => Point;
}

/** Rewrite the f16-enabled WGSL program to its f32 equivalent.
 *
 *  The blur/KDE pipeline stores intermediate densities in an f16 storage
 *  buffer aliased with a u32 count buffer (they share the 4-byte cell
 *  layout: one u32 == two adjacent f16). On adapters that don't expose
 *  the `shader-f16` feature (e.g. NVIDIA Pascal / Intel UHD via Dawn D3D12)
 *  we instead use one f32 per cell, which keeps the byte layout of the
 *  aliased u32 atomics intact.
 *
 *  The substitutions here are exhaustive for the current `program.wgsl`:
 *    - strip `enable f16;`
 *    - `array<f16>` → `array<f32>`
 *    - `f16(y0)` → `y0` (already f32)
 *    - `f16(f32((*dst)[offset]) + y0)` → `f32((*dst)[offset]) + y0`
 *    - `bitcast<u32>(vec2((*src)[offset * 2], (*src)[offset * 2 + 1]))`
 *       → `bitcast<u32>((*src)[offset])` (read the aliased u32 from one
 *       f32 slot instead of two f16 slots).
 *  Host-side the blur buffer size doubles from 2 to 4 bytes per cell
 *  (see blurBufferSize below).
 */
/** Sentinel ``Float32Array`` used to "park" the f32 fill path while the
 *  packed (u16 → GPU compute unpack) path drives the storage buffers.
 *  Reusing one fixed reference makes ``gpuBufferData``'s identity check
 *  short-circuit to a no-op (it sees ``state.data === EMPTY_F32`` on
 *  every subsequent setProps and skips the writeBuffer). */
const EMPTY_F32: Float32Array<ArrayBuffer> = new Float32Array(new ArrayBuffer(0));

function boundsEqual(a: [number, number] | null, b: [number, number] | null): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  return a[0] === b[0] && a[1] === b[1];
}

function f32ProgramOf(src: string): string {
  return src
    .replace(/^enable f16;\s*$/m, "// enable f16; // stripped for f32 fallback path")
    .replace(/array<f16>/g, "array<f32>")
    .replace(/f16\(y0\)/g, "y0")
    .replace(/f16\(f32\(\(\*dst\)\[offset\]\) \+ y0\)/g, "f32((*dst)[offset]) + y0")
    .replace(
      /bitcast<u32>\(vec2\(\(\*src\)\[offset \* 2\], \(\*src\)\[offset \* 2 \+ 1\]\)\)/g,
      "bitcast<u32>((*src)[offset])",
    );
}

export class EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;
  readonly gpuDevice: GPUDevice;
  readonly useF16: boolean;

  private viewport: Viewport;
  private df: Dataflow;
  private device: Node<GPUDevice>;
  private module: Node<GPUShaderModule>;
  private uniforms: ModuleUniforms;
  private context: GPUCanvasContext;
  private renderInputs: RenderInputs;
  private dataBuffers: DataBuffers;
  private renderer: Node<(props: EmbeddingRendererProps, textureView: GPUTextureView) => void>;
  /** GPU-side u32 unpack pipeline (compute pipeline + bind-group layout
   *  only — the per-dispatch u32 source + uniform buffers are allocated
   *  ephemerally inside ``runUnpack`` and defer-destroyed). The downstream
   *  ``dataBuffers.x``/``y`` f32 storage buffers are filled by this
   *  pipeline rather than by a JS-side ``Float32Array(N)`` allocation. */
  private unpack: UnpackPipeline;
  /** Last (xPacked, yPacked) values fed through the unpack pipeline. We
   *  re-run the unpack only when the buffer reference *or* the bounds
   *  change — Mosaic occasionally re-emits the same arrow batch on a
   *  filter rebind, and a redundant 322 M-element compute dispatch is
   *  expensive enough to make worth skipping. */
  private lastXPacked: Uint32Array<ArrayBuffer> | null = null;
  private lastYPacked: Uint32Array<ArrayBuffer> | null = null;
  private lastCoordsBoundsX: [number, number] | null = null;
  private lastCoordsBoundsY: [number, number] | null = null;
  /** Promise chain for the (X, Y) unpack pair. We chain so each call
   *  awaits the previous unpack's defer-destroy before allocating its
   *  own u32 source — keeping peak GPU residency at one ephemeral u32
   *  source at a time (1.288 GB on 322 M), instead of two simultaneously
   *  (which would push the GPU process past the ~4 GB cap). The render
   *  loop ``await``s this before submitting render commands so the f32
   *  destinations are guaranteed populated by the time the draw reads
   *  them. */
  private _unpackInFlight: Promise<void> = Promise.resolve();

  constructor(
    context: GPUCanvasContext,
    device: GPUDevice,
    format: GPUTextureFormat,
    width: number,
    height: number,
    useF16: boolean = true,
  ) {
    this.useF16 = useF16;
    this.context = context;
    this.gpuDevice = device;

    this.props = {
      mode: "points",
      colorScheme: "light",
      x: new Float32Array(),
      y: new Float32Array(),
      xPacked: null,
      yPacked: null,
      coordsBoundsX: null,
      coordsBoundsY: null,
      category: null,

      categoryCount: 1,
      categoryColors: null,

      viewportX: 0,
      viewportY: 0,
      viewportScale: 1,

      pointSize: 1,
      pointAlpha: 1,
      pointsAlpha: 1,

      densityScaler: 1,
      densityBandwidth: 1,
      densityQuantizationStep: 0.1,
      contoursAlpha: 1,
      densityAlpha: 1,

      gamma: 2.2,
      width: width,
      height: height,

      downsampleMaxPoints: 4000000,
      downsampleDensityWeight: 5,
      isGis: false,
      skipDownsampleCompute: false,
    };

    this.viewport = new Viewport({ x: 0, y: 0, scale: 1 }, width, height);

    this.df = new Dataflow();
    let df = this.df;
    this.renderInputs = {
      mode: df.value(this.props.mode),
      colorScheme: df.value(this.props.colorScheme),
      // ``count`` is the canonical source of truth for buffer sizing —
      // it is set by ``setProps`` from whichever input is active
      // (``x.length`` or ``xPacked.length``). Keeping it separate from
      // ``xData`` lets the f32 storage buffer be sized correctly even
      // when the f32 path is dormant (packed mode).
      count: df.value(0),
      xData: df.value(this.props.x),
      yData: df.value(this.props.y),
      categoryData: df.value(this.props.category),
      categoryCount: df.value(this.props.categoryCount),
      categoryColors: df.value(this.props.categoryColors),
      matrix: df.value(matrix3_identity()),
      width: df.value(width),
      height: df.value(height),
      pointSize: df.value(this.props.pointSize),
      densityBandwidth: df.value(this.props.densityBandwidth),
      downsampleMaxPoints: df.value(this.props.downsampleMaxPoints),
      downsampleDensityWeight: df.value(this.props.downsampleDensityWeight),
    };
    this.device = df.value(device);
    this.dataBuffers = makeDataBuffers(df, this.device, this.renderInputs);
    // The pipeline itself is cheap to compile (one ~30-line WGSL kernel)
    // and lives for the device's lifetime. The u32 source + uniform
    // buffers a dispatch needs are allocated fresh per call inside
    // ``runUnpack`` — keeping ~1.288 GB of persistent u32 GPU memory
    // off the books at 322 M points (the persistent dataflow node would
    // crash the GPU process on cold load against the ~4 GB cap).
    this.unpack = makeUnpackPipeline(df, this.device);
    const moduleCode = this.useF16 ? programCode : f32ProgramOf(programCode);
    this.module = df.derive([this.device], (device) => device.createShaderModule({ code: moduleCode }));
    this.uniforms = makeModuleUniforms(df, this.device);
    this.renderer = makeRenderCommand(
      df,
      this.device,
      this.module,
      this.uniforms,
      format,
      this.renderInputs,
      this.dataBuffers,
      this.useF16,
    );
  }

  /** Set renderer props. Returns true if a render is needed. */
  setProps(newProps: Partial<EmbeddingRendererProps>): boolean {
    let needsRender = false;
    let key: keyof EmbeddingRendererProps;
    for (key in newProps) {
      if (newProps[key] === this.props[key]) {
        continue;
      }
      (this.props as any)[key] = newProps[key];
      needsRender = true;
    }
    this.viewport.update(
      { x: this.props.viewportX, y: this.props.viewportY, scale: this.props.viewportScale },
      this.props.width,
      this.props.height,
      this.props.isGis,
    );
    // Decide which fill path is active. Packed mode requires both axes
    // to be u32 + their bounds — partial coverage falls back to the
    // f32 path (the renderer never mixes fill paths within a frame).
    const usePacked =
      this.props.xPacked != null &&
      this.props.yPacked != null &&
      this.props.coordsBoundsX != null &&
      this.props.coordsBoundsY != null;
    // Memory-leak fix (heap snapshot proved 4 × 627 MiB Uint32Arrays were
    // pinned via these refs after the consumer nulled xPacked, sealing the
    // V8 ArrayBuffer pool below the next batch's 1.3 GB demand). When
    // packed mode is OFF (consumer signalled null), drop the previous
    // packed refs so V8 can reclaim the backing stores. Identity check is
    // restored on the next non-null setProps via the unpack pass.
    if (!usePacked) {
      this.lastXPacked = null;
      this.lastYPacked = null;
      this.lastCoordsBoundsX = null;
      this.lastCoordsBoundsY = null;
    }
    const count = usePacked ? this.props.xPacked!.length : this.props.x.length;
    this.renderInputs.count.value = count;
    this.renderInputs.mode.value = this.props.mode;
    this.renderInputs.colorScheme.value = this.props.colorScheme;
    if (usePacked) {
      // Park the f32 path on a fixed empty array so ``gpuBufferData``
      // sees an unchanged reference and skips its writeBuffer call —
      // the f32 storage buffer is filled by the unpack compute pass
      // below instead.
      if (this.renderInputs.xData.value !== EMPTY_F32) this.renderInputs.xData.value = EMPTY_F32;
      if (this.renderInputs.yData.value !== EMPTY_F32) this.renderInputs.yData.value = EMPTY_F32;
    } else {
      this.renderInputs.xData.value = this.props.x;
      this.renderInputs.yData.value = this.props.y;
    }
    this.renderInputs.categoryData.value = this.props.category;
    this.renderInputs.categoryColors.value = this.props.categoryColors;
    if (this.props.category != null) {
      this.renderInputs.categoryCount.value = this.props.categoryCount;
    } else {
      this.renderInputs.categoryCount.value = 1;
    }
    this.renderInputs.matrix.value = this.viewport.matrix();
    this.renderInputs.width.value = this.props.width;
    this.renderInputs.height.value = this.props.height;
    this.renderInputs.pointSize.value = this.props.pointSize;
    this.renderInputs.densityBandwidth.value = this.props.densityBandwidth;
    this.renderInputs.downsampleMaxPoints.value = this.props.downsampleMaxPoints;
    this.renderInputs.downsampleDensityWeight.value = this.props.downsampleDensityWeight;

    if (usePacked) {
      this.maybeRunUnpack(
        this.props.xPacked!,
        this.props.yPacked!,
        this.props.coordsBoundsX!,
        this.props.coordsBoundsY!,
        count,
      );
    }
    return needsRender;
  }

  /** Run the GPU u32 → f32 unpack pass when (a) the packed buffer
   *  reference changed *or* (b) the bounds changed. We materialise
   *  ``dataBuffers.x``/``y`` first (their factories observe the new
   *  ``count``) so they're sized to fit before the unpack writes into
   *  them. Skipped on identical re-emit — Mosaic occasionally re-fires
   *  the same arrow batch, and a redundant 322 M-element compute
   *  dispatch costs both watt-hours and post-pan latency.
   *
   *  X and Y are sequenced (X submit + drain + destroy, *then* Y) so
   *  only one ephemeral 1.288 GB u32 source is alive at a time. The
   *  chain is exposed via ``unpackInFlight`` so the render loop can
   *  ``await`` it before drawing — without that, the post-data-load
   *  frame would render against a half-populated f32 destination. */
  private maybeRunUnpack(
    xPacked: Uint32Array<ArrayBuffer>,
    yPacked: Uint32Array<ArrayBuffer>,
    boundsX: [number, number],
    boundsY: [number, number],
    count: number,
  ): void {
    const sameX = this.lastXPacked === xPacked && boundsEqual(this.lastCoordsBoundsX, boundsX);
    const sameY = this.lastYPacked === yPacked && boundsEqual(this.lastCoordsBoundsY, boundsY);
    if (sameX && sameY) {
      return;
    }
    // A single storage-buffer binding > device limit silently drops
    // its dispatch on Dawn (the validation error fires on
    // ``uncapturederror`` and the bind group is left as a hole). A
    // clear console.error beats a blank canvas. With u32 the source
    // binding is now the same size as the f32 destination (count * 4),
    // so checking either is sufficient.
    const bindBytes = count * 4;
    const limit = this.gpuDevice.limits.maxStorageBufferBindingSize;
    if (bindBytes > limit) {
      console.error(
        `[atlas-gpu] storage binding ${bindBytes} bytes exceeds maxStorageBufferBindingSize ${limit} — unpack skipped (count=${count}). Consider splitting the dataset.`,
      );
      return;
    }
    const pipeline = this.unpack.pipeline.value;
    const layout = this.unpack.bindGroupLayout.value;
    const device = this.gpuDevice;
    // Capture refs to the destination buffers and bounds *now* — they
    // could change before the chain executes. The chain itself fetches
    // ``dataBuffers.x.value`` lazily at run time so a count-driven
    // re-derive lands in the right place.
    const runX = !sameX;
    const runY = !sameY;
    // Memory-leak fix: drop the previous identity refs *before* the unpack
    // chain awaits — without this the OLD lastXPacked (627 MiB at 164 M
    // points, 1.29 GB at 322 M) lives alongside the NEW xPacked param for
    // the duration of runUnpack, doubling the V8 ArrayBuffer pressure
    // exactly when toArray() in the next scatter is competing for the
    // same pool. Re-assignment to the new value happens after each
    // ``runUnpack`` resolves below.
    if (runX) {
      this.lastXPacked = null;
      this.lastCoordsBoundsX = null;
    }
    if (runY) {
      this.lastYPacked = null;
      this.lastCoordsBoundsY = null;
    }
    const boundsXCopy: [number, number] = [boundsX[0], boundsX[1]];
    const boundsYCopy: [number, number] = [boundsY[0], boundsY[1]];
    // Self-healing chain: ``.catch`` swallows a poisoned previous chain
    // (e.g. device.lost mid-unpack) so subsequent maybeRunUnpack calls
    // can still queue work. The render loop's own watchdog already
    // surfaces the underlying device failure.
    this._unpackInFlight = this._unpackInFlight
      .catch(() => {})
      .then(async () => {
        if (runX) {
          const xDest = this.dataBuffers.x.value;
          await runUnpack(device, pipeline, layout, xPacked, xDest, {
            min: boundsXCopy[0],
            max: boundsXCopy[1],
          });
          this.lastXPacked = xPacked;
          this.lastCoordsBoundsX = boundsXCopy;
        }
        if (runY) {
          const yDest = this.dataBuffers.y.value;
          await runUnpack(device, pipeline, layout, yPacked, yDest, {
            min: boundsYCopy[0],
            max: boundsYCopy[1],
          });
          this.lastYPacked = yPacked;
          this.lastCoordsBoundsY = boundsYCopy;
        }
      })
      .catch((err) => {
        console.error("[atlas-gpu] u32 unpack chain failed:", err);
      });
  }

  /** Promise that resolves when the most recently scheduled unpack
   *  chain has fully landed (X submit + drain + destroy, then Y same).
   *  Render-loop callers should ``await`` this before submitting a
   *  draw so the f32 storage buffers are populated when the vertex
   *  shader reads them. */
  get unpackInFlight(): Promise<void> {
    return this._unpackInFlight;
  }

  /** Render */
  render(): void {
    this.renderer.value(this.props, this.context.getCurrentTexture().createView());
  }

  /** Destroy the renderer and free any resource */
  destroy(): void {
    this.df.destroy();
  }

  /** Produce a density map */
  async densityMap(width: number, height: number, radius: number, viewportState: ViewportState): Promise<DensityMap> {
    let subgraph = this.df.subgraph();
    let { x, y, scale: s } = viewportState;
    let positionMatrix: Matrix3 = [s, 0, 0, 0, s, 0, -x * s, -y * s, 1];
    let inv_matrix = matrix3_inverse(positionMatrix);
    let cmd = makeDensityMapCommand(
      subgraph,
      this.device,
      this.module,
      this.uniforms,
      subgraph.value(width),
      subgraph.value(height),
      subgraph.value(radius),
      subgraph.value(positionMatrix),
      this.dataBuffers,
    );
    let data = await cmd.value();
    let isGis = this.props.isGis;
    subgraph.destroy();
    return {
      data: data,
      width: width,
      height: height,
      coordinateAtPixel: (x: number, y: number) => {
        let tx = (x / width) * 2 - 1;
        let ty = (y / height) * 2 - 1;
        let r = matrix3_vector_mul_matrix([tx, ty, 1], inv_matrix);
        return { x: r[0], y: isGis ? Viewport.unprojectLat(r[1]) : r[1] };
      },
    };
  }
}

export interface RenderInputs {
  mode: ValueNode<RenderMode>;
  colorScheme: ValueNode<"light" | "dark">;
  /** Canonical point count — drives storage-buffer sizing regardless of
   *  whether the active fill path is f32 (``xData``) or packed u32
   *  (``setProps`` runs the unpack compute pass). */
  count: ValueNode<number>;
  xData: ValueNode<Float32Array<ArrayBuffer>>;
  yData: ValueNode<Float32Array<ArrayBuffer>>;
  categoryData: ValueNode<Uint8Array<ArrayBuffer> | null>;
  categoryCount: ValueNode<number>;
  categoryColors: ValueNode<string[] | null>;
  pointSize: ValueNode<number>;
  densityBandwidth: ValueNode<number>;
  matrix: ValueNode<Matrix3>;
  width: ValueNode<number>;
  height: ValueNode<number>;
  downsampleMaxPoints: ValueNode<number | null>;
  downsampleDensityWeight: ValueNode<number>;
}

export interface DataBuffers {
  x: Node<GPUBuffer>;
  y: Node<GPUBuffer>;
  category: Node<GPUBuffer | null>;
  count: Node<number>;
}

export interface AuxiliaryResources {
  colorTexture: Node<GPUTexture>;
  colorTextureFormat: GPUTextureFormat;
  alphaTexture: Node<GPUTexture>;
  alphaTextureFormat: GPUTextureFormat;
  countBuffer: Node<GPUBuffer>;
  blurBuffer: Node<GPUBuffer>;
}

function makeDataBuffers(df: Dataflow, device: Node<GPUDevice>, inputs: RenderInputs): DataBuffers {
  let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  // ``inputs.count`` is the renderer-set canonical count. In the f32
  // fill path it equals ``xData.length`` (set in ``setProps``); in the
  // packed path the renderer sets it from ``xPacked.length`` and writes
  // the f32 buffer through the unpack compute pass instead of through
  // ``gpuBufferData``.
  const count = inputs.count;
  const xyDataSize = df.derive([count], (c: number) => c * 4);
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

export function makeAuxiliaryResources(
  df: Dataflow,
  device: Node<GPUDevice>,
  framebufferWidth: Node<number>,
  framebufferHeight: Node<number>,
  densityWidth: Node<number>,
  densityHeight: Node<number>,
  categoryCount: Node<number>,
  useF16: boolean = true,
): AuxiliaryResources {
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
  let countBufferSize = df.derive(
    [densityWidth, densityHeight, categoryCount],
    (w: number, h: number, c: number) => w * h * c * 4, // w * h * categoryCount * sizeof(uint32)
  );
  // 2 bytes/cell for f16, 4 bytes/cell for the f32 fallback (one f32 per
  // cell keeps the aliased u32 count-buffer layout intact).
  const blurBytesPerCell = useF16 ? 2 : 4;
  let blurBufferSize = df.derive(
    [densityWidth, densityHeight, categoryCount],
    (w: number, h: number, c: number) => w * h * c * blurBytesPerCell,
  );
  let countBuffer = df.statefulDerive(
    [device, countBufferSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC],
    gpuBuffer,
  );
  let blurBuffer = df.statefulDerive([device, blurBufferSize, GPUBufferUsage.STORAGE], gpuBuffer);

  return {
    colorTexture: colorTexture,
    alphaTexture: alphaTexture,
    colorTextureFormat: colorTextureFormat,
    alphaTextureFormat: alphaTextureFormat,
    countBuffer: countBuffer,
    blurBuffer: blurBuffer,
  };
}

function makeRenderCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  uniforms: ModuleUniforms,
  format: GPUTextureFormat,
  inputs: RenderInputs,
  dataBuffers: DataBuffers,
  useF16: boolean = true,
): Node<(props: EmbeddingRendererProps, textureView: GPUTextureView) => void> {
  const densityPixelRatio = 4;
  let safeMargin = df.derive([inputs.densityBandwidth], (r: number) => Math.ceil(r * 3) + 1);
  let fbWidth = df.derive([inputs.width, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  let fbHeight = df.derive([inputs.height, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  let densityWidth = df.derive([fbWidth], (x) => Math.ceil(x / densityPixelRatio));
  let densityHeight = df.derive([fbHeight], (x) => Math.ceil(x / densityPixelRatio));

  let auxiliaryResources = makeAuxiliaryResources(
    df,
    device,
    fbWidth,
    fbHeight,
    densityWidth,
    densityHeight,
    inputs.categoryCount,
    useF16,
  );
  let bindGroups = makeBindGroups(df, device, uniforms.buffer, dataBuffers, auxiliaryResources);

  // Create downsampling resources
  let downsampleResources = makeDownsampleResources(df, device, dataBuffers.count, inputs.downsampleMaxPoints);

  let accumulate = makeAccumulateCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let drawPoints = makeDrawPointsCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let drawPointsDownsampled = makeDrawPointsDownsampledCommand(
    df,
    device,
    module,
    bindGroups,
    downsampleResources,
    auxiliaryResources,
  );
  let drawPointsCompacted = makeDrawPointsCompactedCommand(
    df,
    device,
    module,
    bindGroups,
    downsampleResources,
    auxiliaryResources,
  );
  let drawDensityMap = makeDrawDensityMapCommand(df, device, module, bindGroups, auxiliaryResources);
  let gammaCorrection = makeGammaCorrectionCommand(df, device, module, format, bindGroups);
  let gaussianBlur = makeGaussianBlurCommand(df, device, module, bindGroups, fbWidth, fbHeight, inputs.categoryCount);

  // Create downsampling command
  let layoutsNode = df.derive([bindGroups.layouts], (layouts) => layouts);
  let downsample = makeDownsampleCommand(
    df,
    device,
    module,
    df.derive([layoutsNode], (l) => l.group0),
    df.derive([layoutsNode], (l) => l.group1),
    auxiliaryResources.blurBuffer, // Pass blur buffer directly for density lookup
    bindGroups.group0,
    bindGroups.group1,
    downsampleResources,
    dataBuffers,
  );

  let kde_coeffs = df.derive(
    [inputs.densityBandwidth, fbWidth, densityWidth],
    (bandwidth: number, fbWidth: number, densityWidth: number) => kdeConfig((bandwidth / fbWidth) * densityWidth),
  );
  let categoryColors = df.derive(
    [inputs.categoryColors, inputs.categoryCount],
    (colors: string[] | null, count: number) => {
      if (colors == null) {
        colors = defaultCategoryColors(count);
      }
      return colors.map((x) => parseColorNormalizedRgb(x));
    },
  );

  return df.derive(
    [
      device,
      fbWidth,
      fbHeight,
      densityWidth,
      densityHeight,
      uniforms.update,
      dataBuffers.count,
      inputs.matrix,
      categoryColors,
      drawPoints,
      drawPointsDownsampled,
      drawPointsCompacted,
      gammaCorrection,
      accumulate,
      gaussianBlur,
      drawDensityMap,
      downsample,
      kde_coeffs,
    ],
    (
      device,
      fbWidth,
      fbHeight,
      densityWidth,
      densityHeight,
      updateUniforms,
      count: number,
      positionMatrix: Matrix3,
      categoryColors,
      drawPoints,
      drawPointsDownsampled,
      drawPointsCompacted,
      gammaCorrection,
      accumulate,
      gaussianBlur,
      drawDensityMap,
      downsample,
      kde_coeffs,
    ) =>
      (props, textureView) => {
        let backgroundColor: Vector4 = props.colorScheme == "light" ? [1, 1, 1, 0] : [0, 0, 0, 0];
        let scalerX = props.width / fbWidth;
        let scalerY = props.height / fbHeight;
        let safeMarginAdjustmentMatrix: Matrix3 = [scalerX, 0, 0, 0, scalerY, 0, 0, 0, 1];
        let matrix = matrix3_matrix_mul_matrix(safeMarginAdjustmentMatrix, positionMatrix);
        updateUniforms({
          count: count,
          category_count: props.categoryCount,
          framebuffer_width: fbWidth,
          framebuffer_height: fbHeight,
          density_width: densityWidth,
          density_height: densityHeight,
          gamma: props.gamma,
          point_size: Math.max(props.mode == "points" ? 0.3 : 0.1, props.pointSize),
          point_alpha: props.pointAlpha,
          points_alpha: props.pointsAlpha,
          density_scaler: props.densityScaler / (densityPixelRatio * densityPixelRatio),
          quantization_step: props.densityQuantizationStep,
          density_alpha: props.densityAlpha,
          contours_alpha: props.contoursAlpha,
          matrix: matrix,
          view_xy_scaler: [1 / scalerX, 1 / scalerY],
          kde_causal: kde_coeffs.kde_causal,
          kde_anticausal: kde_coeffs.kde_anticausal,
          kde_a: kde_coeffs.kde_a,
          background_color: backgroundColor,
          category_colors: categoryColors,
        });

        // Check if downsampling is enabled
        // Normalize the maxPoints value: null, Infinity, NaN, or negative disable downsampling.
        // 0 is a valid request to render zero points (flows through the compacted path,
        // which resets indirect args and draws no instances).
        const maxPoints = props.downsampleMaxPoints;
        const userMaxPoints =
          maxPoints === null || maxPoints === Infinity || !Number.isFinite(maxPoints) || maxPoints < 0
            ? null
            : maxPoints;
        // ``"All"`` (userMaxPoints == null) MUST draw every point, no
        // matter how large N is — sampling silently drops sparse regions
        // (a 322 M Europe-wide dataset rendered with a 50 M density cap
        // showed only Germany/Austria/CZ/Finland; UK/France/Spain/Italy
        // got dropped because density_sample favoured the densest areas).
        // The chunked drawPoints path below splits the instance draw
        // into 16 M-sized cmd buffers so each fits Metal's 5 s wall-clock
        // watchdog, and ``onSubmittedWorkDone`` is awaited per submit so
        // the 30 s backpressure timeout never trips. Cold init + each
        // pan re-render both go through this path; users explicitly
        // accept the longer per-frame budget in exchange for a complete
        // map.
        const effectiveMaxPoints = userMaxPoints;
        const useDownsampling = effectiveMaxPoints !== null && count > effectiveMaxPoints;

        if (count > 1_000_000 && !(globalThis as any).__atlasRenderDiagLogged) {
          (globalThis as any).__atlasRenderDiagLogged = true;
          console.log(
            `[atlas-renderdiag] count=${count} useDownsampling=${useDownsampling} effectiveMaxPoints=${effectiveMaxPoints} mode=${props.mode} densityWeight=${props.downsampleDensityWeight} skipDownsample=${props.skipDownsampleCompute} usePacked=${props.xPacked != null}`,
          );
        }

        if (useDownsampling) {
          const wantsDensityOverlay = props.mode == "density" && (props.densityAlpha > 0 || props.contoursAlpha > 0);
          const wantsDensityWeighting = props.downsampleDensityWeight > 0;

          // accumulate + blur populate ``blur_buffer`` which the
          // ``downsample_viewport_cull`` pass then reads to weight per-point
          // acceptance. Because ``downsample`` now submits its own command
          // buffers directly (per-pass + per-chunk to stay under Metal's
          // 5 s wall-clock watchdog at 322 M points), we must finish + submit
          // the accumulate/blur encoder BEFORE the first downsample submit —
          // otherwise the cull pass races on a stale (zero on first frame)
          // ``blur_buffer``, every point sees density ≈ 0, and the
          // probabilistic acceptance produces a visually empty scatter.
          if (wantsDensityWeighting || wantsDensityOverlay) {
            const preEncoder = device.createCommandEncoder();
            accumulate(preEncoder);
            gaussianBlur(preEncoder);
            device.queue.submit([preEncoder.finish()]);
          }

          const downsampleConfig: DownsampleConfig = {
            maxPoints: effectiveMaxPoints!,
            densityWeight: wantsDensityWeighting ? props.downsampleDensityWeight : 0,
            frameSeed: 42,
          };

          // ``downsample`` internally submits 3 passes × ~K chunks (typically
          // 3 × 5 = 15 cmd buffers at 322 M) so each one stays under the Metal
          // watchdog. WebGPU queue ordering keeps the cull → sample → compact
          // → drawIndirect dependency chain intact across the separate submits.
          if (!props.skipDownsampleCompute) {
            downsample(downsampleConfig);
          }

          const drawEncoder = device.createCommandEncoder();
          drawPointsCompacted(drawEncoder);
          if (wantsDensityOverlay) {
            drawDensityMap(drawEncoder);
          }
          gammaCorrection(drawEncoder, textureView);
          device.queue.submit([drawEncoder.finish()]);
        } else {
          // Chunked instance draw. ``pass.draw(4, count)`` for the
          // ``downsampleMaxPoints = "All"`` case at 322 M instances tries
          // to push 1.3 B vertex shader runs through one MTLCommandBuffer
          // — easily exceeds Metal's 5 s wall-clock watchdog and
          // ``onSubmittedWorkDone()`` never resolves. Splitting the draw
          // into ``DRAW_CHUNK_INSTANCES``-sized cmd buffers keeps every
          // cmd buffer under the watchdog while still drawing every
          // point. Only the first chunk clears the color/alpha
          // attachments — subsequent chunks ``load`` and additively
          // blend onto the partial buffer.
          //
          // 1 M instances chosen empirically: 16 M / 4 M / 2 M each
          // tripped Metal's 5 s watchdog at 322 M when the renderer
          // shared the box with a heavyweight sidecar (DuckDB at
          // 50 + GB) and a parallel column-chart-discovery batch
          // pushed one cmd buffer past 5 s of GPU wall. 2 M nominally
          // had 16× headroom but it was not enough under combined
          // CPU/RAM/GPU contention: a single dense world-view chunk
          // could see fragment overdraw + paging slowdowns multiply
          // execution time well beyond the empirical mean. 1 M doubles
          // headroom (32×) at the cost of 322 chunks at 322 M; encoder
          // setup is microseconds-per-chunk so this adds ~ tens of ms
          // of CPU overhead — invisible against the multi-second GPU
          // wall it protects. Pan re-renders at country/city zoom
          // still early-exit most VS runs via the bounds check and
          // complete in a few s regardless of chunk count.
          const DRAW_CHUNK_INSTANCES = 1_000_000;
          const totalChunks = count > 0 ? Math.max(1, Math.ceil(count / DRAW_CHUNK_INSTANCES)) : 1;
          if (count > 0) {
            for (let chunk = 0; chunk < totalChunks; chunk++) {
              const instanceFirst = chunk * DRAW_CHUNK_INSTANCES;
              const instanceCount = Math.min(DRAW_CHUNK_INSTANCES, count - instanceFirst);
              if (instanceCount <= 0) break;
              const enc = device.createCommandEncoder();
              drawPoints(enc, instanceFirst, instanceCount, chunk === 0);
              device.queue.submit([enc.finish()]);
            }
          } else {
            // Even at count==0 we still need a single render pass to
            // clear the colour attachment so the gamma stage doesn't
            // sample stale contents.
            const enc = device.createCommandEncoder();
            drawPoints(enc, 0, 0, true);
            device.queue.submit([enc.finish()]);
          }

          const tailEncoder = device.createCommandEncoder();
          if (props.mode == "density") {
            if (props.densityAlpha > 0 || props.contoursAlpha > 0) {
              accumulate(tailEncoder);
              gaussianBlur(tailEncoder);
              drawDensityMap(tailEncoder);
            }
          }
          gammaCorrection(tailEncoder, textureView);
          device.queue.submit([tailEncoder.finish()]);
        }
      },
  );
}

function makeDensityMapCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  uniforms: ModuleUniforms,
  width: Node<number>,
  height: Node<number>,
  radius: Node<number>,
  matrix: Node<Matrix3>,
  dataBuffers: DataBuffers,
): Node<() => Promise<Float32Array>> {
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
        count: count,
        category_count: 1,
        framebuffer_width: width,
        framebuffer_height: height,
        density_width: width,
        density_height: height,
        gamma: 1,
        point_size: 0,
        point_alpha: 0,
        points_alpha: 0,
        density_scaler: 0,
        quantization_step: 0,
        density_alpha: 0,
        contours_alpha: 0,
        matrix: matrix,
        view_xy_scaler: [1, 1],
        kde_causal: kde_coeffs.kde_causal,
        kde_anticausal: kde_coeffs.kde_anticausal,
        kde_a: kde_coeffs.kde_a,
        background_color: [0, 0, 0, 0],
        category_colors: [],
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

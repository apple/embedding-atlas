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
import type { DensityMap, EmbeddingRenderer, EmbeddingRendererProps, RenderMode } from "../renderer_interface.js";
import type { ViewportState } from "../utils.js";
import { Viewport } from "../viewport_utils.js";
import { makeModuleUniforms, type ModuleUniforms } from "./uniforms.js";
import { gpuBuffer, gpuBufferData, gpuTexture } from "./utils.js";

import { makeAccumulateCommand } from "./accumulate.js";
import { makeBindGroups } from "./bind_groups.js";
import { makeDrawDensityMapCommand } from "./draw_density_map.js";
import { makeDrawPointsCommand } from "./draw_points.js";
import { makeGammaCorrectionCommand } from "./gamma_correction.js";
import { makeGaussianBlurCommand } from "./gaussian_blur.js";
import { kdeConfig } from "./kde_config.js";

import programCode from "./program.wgsl?raw";

export class EmbeddingRendererWebGPU implements EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;

  private viewport: Viewport;
  private df: Dataflow;
  private device: Node<GPUDevice>;
  private module: Node<GPUShaderModule>;
  private uniforms: ModuleUniforms;
  private context: GPUCanvasContext;
  private renderInputs: RenderInputs;
  private dataBuffers: DataBuffers;
  private renderer: Node<(props: EmbeddingRendererProps, textureView: GPUTextureView) => void>;

  constructor(context: GPUCanvasContext, device: GPUDevice, format: GPUTextureFormat, width: number, height: number) {
    this.context = context;

    this.props = {
      mode: "points",
      colorScheme: "light",
      x: new Float32Array(),
      y: new Float32Array(),
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
    };

    this.viewport = new Viewport({ x: 0, y: 0, scale: 1 }, width, height);

    this.df = new Dataflow();
    let df = this.df;
    this.renderInputs = {
      mode: df.value(this.props.mode),
      colorScheme: df.value(this.props.colorScheme),
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
    };
    this.device = df.value(device);
    this.dataBuffers = makeDataBuffers(df, this.device, this.renderInputs);
    this.module = df.derive([this.device], (device) => device.createShaderModule({ code: programCode }));
    this.uniforms = makeModuleUniforms(df, this.device);
    this.renderer = makeRenderCommand(
      df,
      this.device,
      this.module,
      this.uniforms,
      format,
      this.renderInputs,
      this.dataBuffers,
    );
  }

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
    );
    this.renderInputs.mode.value = this.props.mode;
    this.renderInputs.colorScheme.value = this.props.colorScheme;
    this.renderInputs.xData.value = this.props.x;
    this.renderInputs.yData.value = this.props.y;
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
    return needsRender;
  }

  render(): void {
    this.renderer.value(this.props, this.context.getCurrentTexture().createView());
  }

  destroy(): void {
    this.df.destroy();
  }

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
    subgraph.destroy();
    return {
      data: data,
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

export interface RenderInputs {
  mode: ValueNode<RenderMode>;
  colorScheme: ValueNode<"light" | "dark">;
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

export function makeAuxiliaryResources(
  df: Dataflow,
  device: Node<GPUDevice>,
  framebufferWidth: Node<number>,
  framebufferHeight: Node<number>,
  densityWidth: Node<number>,
  densityHeight: Node<number>,
  categoryCount: Node<number>,
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
  let blurBufferSize = df.derive(
    [densityWidth, densityHeight, categoryCount],
    (w: number, h: number, c: number) => w * h * c * 2, // w * h * categoryCount * sizeof(f16)
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
  );
  let bindGroups = makeBindGroups(df, device, uniforms.buffer, dataBuffers, auxiliaryResources);

  let accumulate = makeAccumulateCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let drawPoints = makeDrawPointsCommand(df, device, module, bindGroups, dataBuffers, auxiliaryResources);
  let drawDensityMap = makeDrawDensityMapCommand(df, device, module, bindGroups, auxiliaryResources);
  let gammaCorrection = makeGammaCorrectionCommand(df, device, module, format, bindGroups);
  let gaussianBlur = makeGaussianBlurCommand(df, device, module, bindGroups, fbWidth, fbHeight, inputs.categoryCount);

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
      gammaCorrection,
      accumulate,
      gaussianBlur,
      drawDensityMap,
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
      gammaCorrection,
      accumulate,
      gaussianBlur,
      drawDensityMap,
      kde_coeffs,
    ) =>
      (props, textureView) => {
        let backgroundColor: Vector4 = props.colorScheme == "light" ? [1, 1, 1, 1] : [0, 0, 0, 1];
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
          point_size: Math.max(props.mode == "points" ? 3 : 1, props.pointSize),
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
        let encoder = device.createCommandEncoder();
        drawPoints(encoder);
        if (props.mode == "density") {
          if (props.densityAlpha > 0 || props.contoursAlpha > 0) {
            accumulate(encoder);
            gaussianBlur(encoder);
            drawDensityMap(encoder);
          }
        }
        gammaCorrection(encoder, textureView);
        device.queue.submit([encoder.finish()]);
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

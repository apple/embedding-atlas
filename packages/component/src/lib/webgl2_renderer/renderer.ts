// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { defaultCategoryColors, parseColorNormalizedRgb } from "../colors.js";
import { Dataflow, Node, ValueNode } from "../dataflow.js";
import {
  matrix3_identity,
  matrix3_inverse,
  matrix3_matrix_mul_matrix,
  matrix3_vector_mul_matrix,
  type Matrix3,
} from "../matrix.js";
import type { DensityMap, EmbeddingRenderer, EmbeddingRendererProps, RenderMode } from "../renderer_interface.js";
import type { ViewportState } from "../utils.js";
import { Viewport } from "../viewport_utils.js";
import { discBlurCommand } from "./disc_blur.js";
import { fillCountBufferCommand } from "./fill_count_buffer.js";
import { gammaCorrectionCommand } from "./gamma_correction.js";
import { gaussianBlurCommand, gaussianBlurPixelRadius } from "./gaussian_blur.js";
import { gaussianBlurR20Command, gaussianBlurR20PixelRadius } from "./gaussian_blur_2.js";
import { paintContoursCommand } from "./paint_contours.js";
import { paintDensityMapCommand } from "./paint_density_map.js";
import { paintDiscretePointsCommand } from "./paint_discrete_points.js";
import { paintPointsCommand } from "./paint_points.js";
import { webglBuffer, webglFramebuffer } from "./utils.js";

export class EmbeddingRendererWebGL2 implements EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;

  private viewport: Viewport;
  private df: Dataflow;
  private gl: Node<WebGL2RenderingContext>;
  private renderInputs: RenderInputs;
  private dataBuffers: DataBuffers;
  private renderer: Node<(props: EmbeddingRendererProps) => void>;

  constructor(context: WebGL2RenderingContext, width: number, height: number) {
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

      downsampleMaxPoints: 4000000,
      downsampleDensityWeight: 5,
    };

    this.viewport = new Viewport({ x: 0, y: 0, scale: 1 }, width, height);

    let df = new Dataflow();
    let gl = df.value(context);
    this.df = df;
    this.gl = gl;
    this.renderInputs = {
      mode: df.value(this.props.mode),
      colorScheme: df.value(this.props.colorScheme),
      xData: df.value(this.props.x),
      yData: df.value(this.props.y),
      categoryData: df.value(this.props.category),
      categoryCount: df.value(this.props.categoryCount),
      matrix: df.value(matrix3_identity()),
      width: df.value(width),
      height: df.value(height),
      pointSize: df.value(this.props.pointSize),
      densityBandwidth: df.value(this.props.densityBandwidth),
      downsampleMaxPoints: df.value(this.props.downsampleMaxPoints),
      downsampleDensityWeight: df.value(this.props.downsampleDensityWeight),
    };
    this.dataBuffers = dataBuffers(df, gl, this.renderInputs);
    this.renderer = renderCommand(df, gl, this.renderInputs, this.dataBuffers);
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
    return needsRender;
  }

  render(): void {
    this.renderer.value(this.props);
  }

  destroy(): void {
    this.df.destroy();
  }

  async densityMap(width: number, height: number, radius: number, viewportState: ViewportState): Promise<DensityMap> {
    let df = this.df.subgraph();
    let cmd = densityMapCommand(df, this.gl, this.dataBuffers, df.value(width), df.value(height), df.value(radius));
    let { x, y, scale: s } = viewportState;
    let positionMatrix: Matrix3 = [s, 0, 0, 0, s, 0, -x * s, -y * s, 1];
    let data = cmd.value(positionMatrix);
    let inv_matrix = matrix3_inverse(positionMatrix);
    df.destroy();
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

interface RenderInputs {
  mode: ValueNode<RenderMode>;
  colorScheme: ValueNode<"light" | "dark">;
  xData: ValueNode<number[] | Float32Array>;
  yData: ValueNode<number[] | Float32Array>;
  categoryData: ValueNode<number[] | Uint8Array | null>;
  categoryCount: ValueNode<number>;
  pointSize: ValueNode<number>;
  densityBandwidth: ValueNode<number>;
  matrix: ValueNode<Matrix3>;
  width: ValueNode<number>;
  height: ValueNode<number>;
  downsampleMaxPoints: ValueNode<number | null>;
  downsampleDensityWeight: ValueNode<number>;
}

interface DataBuffers {
  x: Node<WebGLBuffer>;
  y: Node<WebGLBuffer>;
  category: Node<WebGLBuffer | null>;
  count: Node<number>;
}

function dataBuffers(df: Dataflow, gl: Node<WebGL2RenderingContext>, inputs: RenderInputs): DataBuffers {
  const xBuffer = df.statefulDerive([gl, inputs.xData, "f32"], webglBuffer);
  const yBuffer = df.statefulDerive([gl, inputs.yData, "f32"], webglBuffer);
  const categoryBuffer = df.if(
    df.derive([inputs.categoryData], (x) => x != null),
    (df) => df.statefulDerive([gl, df.assertNotNull(inputs.categoryData), "u8"], webglBuffer),
    (df) => df.value(null),
  );
  const count = df.derive([inputs.xData], (d: ArrayLike<number>) => d.length);
  return { x: xBuffer, y: yBuffer, category: categoryBuffer, count: count };
}

function renderCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  inputs: RenderInputs,
  dataBuffers: DataBuffers,
): Node<(props: EmbeddingRendererProps) => void> {
  return df.switch(inputs.mode, {
    points: (df) => pointsRenderCommand(df, gl, inputs, dataBuffers),
    density: (df) => densityRenderCommand(df, gl, inputs, dataBuffers),
  });
}

function pointsRenderCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  inputs: RenderInputs,
  buffers: DataBuffers,
): Node<(props: EmbeddingRendererProps) => void> {
  const hasCategory = df.derive([inputs.categoryCount], (x: number) => x > 1);
  const linearFB = df.statefulDerive([gl, inputs.width, inputs.height, 4, "f32"], webglFramebuffer);
  let paintDiscretePoints = df.if(
    hasCategory,
    (df) => paintDiscretePointsCommand(df, gl, buffers.x, buffers.y, df.assertNotNull(buffers.category), buffers.count),
    (df) => paintDiscretePointsCommand(df, gl, buffers.x, buffers.y, null, buffers.count),
  );
  let gammaCorrection = gammaCorrectionCommand(df, gl);
  return df.derive(
    [gl, linearFB, paintDiscretePoints, gammaCorrection, inputs.colorScheme, inputs.matrix, inputs.categoryCount],
    (
      gl,
      linearFB,
      paintDiscretePoints,
      gammaCorrection,
      colorScheme: "light" | "dark",
      matrix: Matrix3,
      categoryCount: number,
    ) =>
      (props) => {
        let colorMatrix: number[] = [];
        let categoryColors = props.categoryColors ?? defaultCategoryColors(props.categoryCount);
        for (let i = 0; i < categoryCount; i++) {
          if (i < categoryColors.length) {
            let { r, g, b } = parseColorNormalizedRgb(categoryColors[i]);
            r = Math.pow(r, props.gamma);
            g = Math.pow(g, props.gamma);
            b = Math.pow(b, props.gamma);
            colorMatrix = colorMatrix.concat([r, g, b, 1]);
          } else {
            colorMatrix = colorMatrix.concat([0.5, 0.5, 0.5, 1]);
          }
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, linearFB.framebuffer);
        gl.viewport(0, 0, linearFB.width, linearFB.height);
        if (colorScheme == "light") {
          gl.clearColor(1, 1, 1, 1);
        } else {
          gl.clearColor(0, 0, 0, 1);
        }
        gl.clear(gl.COLOR_BUFFER_BIT);
        paintDiscretePoints(matrix, Math.max(3, props.pointSize), props.pointAlpha * props.pointsAlpha, colorMatrix);

        // Convert linear RGB to sRGB for display
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, props.width, props.height);
        gammaCorrection(linearFB.texture, props.gamma);
      },
  );
}

function densityRenderCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  inputs: RenderInputs,
  buffers: DataBuffers,
): Node<(props: EmbeddingRendererProps) => void> {
  let safeMargin = df.derive([inputs.densityBandwidth], (r: number) => gaussianBlurR20PixelRadius(r) + 1);
  let fbWidth = df.derive([inputs.width, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  let fbHeight = df.derive([inputs.height, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  const hasCategory = df.derive([inputs.categoryCount], (x: number) => x > 1);
  const countFB = df.statefulDerive([gl, fbWidth, fbHeight, 4, "f32"], webglFramebuffer);
  const linearFB = df.statefulDerive([gl, fbWidth, fbHeight, 4, "f32"], webglFramebuffer);
  const tempFB1 = df.statefulDerive([gl, fbWidth, fbHeight, 4, "f32"], webglFramebuffer);
  const tempFB2 = df.statefulDerive([gl, fbWidth, fbHeight, 4, "f32"], webglFramebuffer);
  let fillCountBuffer = df.if(
    hasCategory,
    (df) => fillCountBufferCommand(df, gl, buffers.x, buffers.y, df.assertNotNull(buffers.category), buffers.count),
    (df) => fillCountBufferCommand(df, gl, buffers.x, buffers.y, null, buffers.count),
  );
  let discBlur = discBlurCommand(df, gl, inputs.pointSize);
  let gaussianBlur = gaussianBlurR20Command(df, gl, inputs.densityBandwidth);
  let paintPoints = paintPointsCommand(df, gl);
  let paintDensityMap = paintDensityMapCommand(df, gl);
  let paintContours = paintContoursCommand(df, gl);
  let gammaCorrection = gammaCorrectionCommand(df, gl);
  return df.derive(
    [
      gl,
      countFB,
      linearFB,
      tempFB1,
      tempFB2,
      inputs.colorScheme,
      inputs.matrix,
      fillCountBuffer,
      discBlur,
      gaussianBlur,
      paintPoints,
      paintDensityMap,
      paintContours,
      gammaCorrection,
    ],
    (
      gl: WebGL2RenderingContext,
      countFB,
      linearFB,
      tempFB1,
      tempFB2,
      colorScheme: "light" | "dark",
      positionMatrix: Matrix3,
      fillCountBuffer,
      discBlur,
      gaussianBlur,
      paintPoints,
      paintDensityMap,
      paintContours,
      gammaCorrection,
    ) =>
      (props) => {
        let categoryColors = props.categoryColors ?? defaultCategoryColors(props.categoryCount);
        let colorMatrix: number[] = [];
        for (let i = 0; i < 4; i++) {
          if (i < categoryColors.length) {
            let { r, g, b } = parseColorNormalizedRgb(categoryColors[i]);
            r = Math.pow(r, props.gamma);
            g = Math.pow(g, props.gamma);
            b = Math.pow(b, props.gamma);
            colorMatrix = colorMatrix.concat([r, g, b, 1]);
          } else {
            colorMatrix = colorMatrix.concat([0.5, 0.5, 0.5, 1]);
          }
        }

        let scalerX = props.width / linearFB.width;
        let scalerY = props.height / linearFB.height;

        let safeMarginAdjustmentMatrix: Matrix3 = [scalerX, 0, 0, 0, scalerY, 0, 0, 0, 1];
        let matrix = matrix3_matrix_mul_matrix(safeMarginAdjustmentMatrix, positionMatrix);

        // Fill the count buffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, countFB.framebuffer);
        gl.viewport(0, 0, countFB.width, countFB.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        fillCountBuffer(matrix);

        // Clear
        gl.bindFramebuffer(gl.FRAMEBUFFER, linearFB.framebuffer);
        gl.viewport(0, 0, linearFB.width, linearFB.height);
        if (colorScheme == "light") {
          gl.clearColor(1, 1, 1, 1);
        } else {
          gl.clearColor(0, 0, 0, 1);
        }

        gl.clear(gl.COLOR_BUFFER_BIT);

        // Draw points
        if (props.pointAlpha > 0 && props.pointsAlpha > 0) {
          discBlur(countFB.texture, tempFB1, tempFB2);
          gl.bindFramebuffer(gl.FRAMEBUFFER, linearFB.framebuffer);
          paintPoints(tempFB1, props.pointAlpha, props.pointsAlpha, colorMatrix, colorScheme);
        }

        // Draw density map and contours
        if (props.densityScaler > 0 && (props.densityAlpha > 0 || props.contoursAlpha > 0)) {
          gaussianBlur(countFB.texture, tempFB1, tempFB2);
          gl.bindFramebuffer(gl.FRAMEBUFFER, linearFB.framebuffer);
          // Density map
          if (props.densityAlpha > 0) {
            paintDensityMap(
              tempFB1,
              props.densityScaler,
              props.densityQuantizationStep,
              props.densityAlpha,
              colorMatrix,
              colorScheme,
            );
          }
          // Contours
          if (props.contoursAlpha > 0) {
            for (let i = 0; i < categoryColors.length; i++) {
              let channelMask: number[] = [0, 0, 0, 0];
              channelMask[i] = 1;
              paintContours(
                tempFB1,
                props.densityScaler,
                props.densityQuantizationStep,
                props.contoursAlpha,
                channelMask,
                colorMatrix.slice(i * 4, i * 4 + 4),
              );
            }
          }
        }
        // Convert linear RGB to sRGB for display
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, props.width, props.height);
        gammaCorrection(linearFB.texture, props.gamma, 1 / scalerX, 1 / scalerY);
      },
  );
}

function densityMapCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  dataBuffers: DataBuffers,
  width: Node<number>,
  height: Node<number>,
  bandwidth: Node<number>,
): Node<(matrix: Matrix3) => Float32Array> {
  let safeMargin = df.derive([bandwidth], (r: number) => gaussianBlurPixelRadius(r) + 1);
  let fbWidth = df.derive([width, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  let fbHeight = df.derive([height, safeMargin], (x: number, safeMargin: number) => x + safeMargin * 2);
  const countFB = df.statefulDerive([gl, fbWidth, fbHeight, 1, "f32"], webglFramebuffer);
  const tempFB1 = df.statefulDerive([gl, fbWidth, fbHeight, 1, "f32"], webglFramebuffer);
  const tempFB2 = df.statefulDerive([gl, fbWidth, fbHeight, 1, "f32"], webglFramebuffer);
  let fillCountBuffer = fillCountBufferCommand(df, gl, dataBuffers.x, dataBuffers.y, null, dataBuffers.count);
  let gaussianBlur = gaussianBlurCommand(df, gl, bandwidth);
  return df.derive(
    [gl, safeMargin, width, height, countFB, tempFB1, tempFB2, fillCountBuffer, gaussianBlur],
    (gl, safeMargin, width, height, countFB, tempFB1, tempFB2, fillCountBuffer, gaussianBlur) => (positionMatrix) => {
      let scalerX = width / countFB.width;
      let scalerY = height / countFB.height;

      let safeMarginAdjustmentMatrix: Matrix3 = [scalerX, 0, 0, 0, scalerY, 0, 0, 0, 1];
      let matrix = matrix3_matrix_mul_matrix(safeMarginAdjustmentMatrix, positionMatrix);

      gl.bindFramebuffer(gl.FRAMEBUFFER, countFB.framebuffer);
      gl.viewport(0, 0, countFB.width, countFB.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      fillCountBuffer(matrix);

      gaussianBlur(countFB.texture, tempFB1, tempFB2);

      gl.bindFramebuffer(gl.FRAMEBUFFER, tempFB1.framebuffer);
      let result = new Float32Array(width * height);
      gl.readPixels(safeMargin, safeMargin, width, height, gl.RED, gl.FLOAT, result);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      return result;
    },
  );
}

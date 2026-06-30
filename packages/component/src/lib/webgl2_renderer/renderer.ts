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
import { draw3DCount } from "../renderer_interface.js";
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
import { paintPoints3DCommand } from "./paint_points_3d.js";
import { paintPointsCommand } from "./paint_points.js";
import { pickPoints3DCommand, type Pick3DCommand } from "./pick_points_3d.js";
import { webglBuffer, webglElementBuffer, webglFramebuffer } from "./utils.js";

// Stable fallback so the z vertex buffer is always constructible without churn.
const EMPTY_Z = new Float32Array(1);

// Builds the flat RGBA gamma-encoded category color matrix the WebGL paint shaders
// consume: `slots` entries, each a category color (gamma-applied) or a gray
// fallback past the provided colors. `defaultCount` sizes the auto-generated palette
// when none is supplied. Shared by the 2D points, 3D points, and density paths.
function buildGammaColorMatrix(
  categoryColors: string[] | null,
  slots: number,
  defaultCount: number,
  gamma: number,
): number[] {
  let colors = categoryColors ?? defaultCategoryColors(defaultCount);
  let out: number[] = [];
  for (let i = 0; i < slots; i++) {
    if (i < colors.length) {
      let { r, g, b } = parseColorNormalizedRgb(colors[i]);
      out.push(Math.pow(r, gamma), Math.pow(g, gamma), Math.pow(b, gamma), 1);
    } else {
      out.push(0.5, 0.5, 0.5, 1);
    }
  }
  return out;
}

export class EmbeddingRendererWebGL2 implements EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;

  private viewport: Viewport;
  private df: Dataflow;
  private gl: Node<WebGL2RenderingContext>;
  private renderInputs: RenderInputs;
  private dataBuffers: DataBuffers;
  private renderer: Node<(props: EmbeddingRendererProps) => void>;
  private renderer3D: Node<(props: EmbeddingRendererProps) => void>;
  private picker: Node<Pick3DCommand>;
  private pickDirty: boolean = true;
  // Bumped on every view-changing setProps; lets an in-flight pick detect that the
  // view changed before its redraw finished and avoid marking the cache clean.
  private pickGeneration: number = 0;

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

      z: null,
      viewProjection: null,
      cameraEye: [0, 0, 0],
      cameraDistance: 1,
      pointSize3D: 6,
      fogDensity: 0.6,
      sampleIndices: null,
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
      zData: df.value(this.props.z ?? EMPTY_Z),
      categoryData: df.value(this.props.category),
      categoryCount: df.value(this.props.categoryCount),
      sampleIndices: df.value(this.props.sampleIndices),
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
    this.renderer3D = points3DRenderCommand(df, gl, this.renderInputs, this.dataBuffers);
    this.picker = pickPoints3DCommand(
      df,
      gl,
      this.dataBuffers.x,
      this.dataBuffers.y,
      this.dataBuffers.z,
      this.dataBuffers.count,
      this.dataBuffers.sampleIndex,
      this.renderInputs.width,
      this.renderInputs.height,
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
    this.renderInputs.zData.value = this.props.z ?? EMPTY_Z;
    this.renderInputs.categoryData.value = this.props.category;
    this.renderInputs.sampleIndices.value = this.props.sampleIndices;
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
    if (needsRender) {
      // The view changed; the cached 3D pick buffer must be redrawn on next pick.
      // Bump the generation so an in-flight pick does not clear pickDirty afterward.
      this.pickDirty = true;
      this.pickGeneration++;
    }
    return needsRender;
  }

  render(): void {
    if (this.props.viewProjection != null && this.props.z != null) {
      this.renderer3D.value(this.props);
    } else {
      this.renderer.value(this.props);
    }
  }

  async pick(x: number, y: number): Promise<number | null | undefined> {
    if (this.props.viewProjection == null || this.props.z == null) {
      return null;
    }
    let cameraDistance = this.props.cameraDistance;
    // Same sample count as the draw, so picking matches the visible subset.
    let sampleCount = draw3DCount(this.props.sampleIndices, this.props.x.length, this.props.downsampleMaxPoints);
    // Re-render the pick buffer only when the view changed since the last pick.
    let redraw = this.pickDirty;
    let willRedraw = redraw && sampleCount > 0;
    // Capture the view generation so the cache is only marked clean if no newer view
    // change (setProps) happened while this pick ran.
    let gen = this.pickGeneration;
    try {
      let value = await this.picker.value(
        this.props.viewProjection,
        this.props.pointSize3D,
        cameraDistance,
        sampleCount,
        redraw,
        x,
        y,
      );
      // If the view changed (setProps bumped the generation) while this pick ran, its
      // result is stale — report `undefined` (indeterminate, not a no-hit) so callers
      // leave selection unchanged, and leave the cache dirty for the next pick.
      if (this.pickGeneration !== gen) {
        return undefined;
      }
      // Otherwise the redraw, if any, matches the current view: mark the cache clean.
      if (willRedraw) {
        this.pickDirty = false;
      }
      return value;
    } catch (e) {
      // Keep the cache dirty so the next pick redraws; report `undefined` so a
      // transient failure is not misread as an intentional empty-space click.
      return undefined;
    }
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
  zData: ValueNode<number[] | Float32Array>;
  categoryData: ValueNode<number[] | Uint8Array | null>;
  categoryCount: ValueNode<number>;
  sampleIndices: ValueNode<Uint32Array | null>;
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
  z: Node<WebGLBuffer>;
  category: Node<WebGLBuffer | null>;
  count: Node<number>;
  /** Strided element-index buffer for 3D downsampling (null = draw all). */
  sampleIndex: Node<WebGLBuffer | null>;
}

function dataBuffers(df: Dataflow, gl: Node<WebGL2RenderingContext>, inputs: RenderInputs): DataBuffers {
  const xBuffer = df.statefulDerive([gl, inputs.xData, "f32"], webglBuffer);
  const yBuffer = df.statefulDerive([gl, inputs.yData, "f32"], webglBuffer);
  const zBuffer = df.statefulDerive([gl, inputs.zData, "f32"], webglBuffer);
  const categoryBuffer = df.if(
    df.derive([inputs.categoryData], (x) => x != null),
    (df) => df.statefulDerive([gl, df.assertNotNull(inputs.categoryData), "u8"], webglBuffer),
    (df) => df.value(null),
  );
  const count = df.derive([inputs.xData], (d: ArrayLike<number>) => d.length);
  // The 3D draw set (frustum-culled, strided subset for the current camera) is
  // computed by the view and passed in; null means draw all points.
  const sampleIndex = df.if(
    df.derive([inputs.sampleIndices], (d) => d != null),
    (df) => df.statefulDerive([gl, df.assertNotNull(inputs.sampleIndices)], webglElementBuffer),
    (df) => df.value(null),
  );
  return { x: xBuffer, y: yBuffer, z: zBuffer, category: categoryBuffer, count: count, sampleIndex: sampleIndex };
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
        let colorMatrix = buildGammaColorMatrix(props.categoryColors, categoryCount, props.categoryCount, props.gamma);

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

function points3DRenderCommand(
  df: Dataflow,
  gl: Node<WebGL2RenderingContext>,
  inputs: RenderInputs,
  buffers: DataBuffers,
): Node<(props: EmbeddingRendererProps) => void> {
  const hasCategory = df.derive([inputs.categoryCount], (x: number) => x > 1);
  let paintPoints3D = df.if(
    hasCategory,
    (df) =>
      paintPoints3DCommand(
        df,
        gl,
        buffers.x,
        buffers.y,
        buffers.z,
        df.assertNotNull(buffers.category),
        buffers.count,
        buffers.sampleIndex,
      ),
    (df) => paintPoints3DCommand(df, gl, buffers.x, buffers.y, buffers.z, null, buffers.count, buffers.sampleIndex),
  );
  return df.derive(
    [gl, paintPoints3D, inputs.colorScheme, inputs.categoryCount],
    (gl, paintPoints3D, colorScheme: "light" | "dark", categoryCount: number) => (props) => {
      if (props.viewProjection == null) {
        return;
      }
      let colorMatrix = buildGammaColorMatrix(props.categoryColors, categoryCount, props.categoryCount, props.gamma);
      // Gamma-encoded background (matches the in-shader gamma applied by the points).
      let background: [number, number, number] = colorScheme == "light" ? [1, 1, 1] : [0, 0, 0];
      let cameraDistance = props.cameraDistance;
      let sampleCount = draw3DCount(props.sampleIndices, props.x.length, props.downsampleMaxPoints);
      let invGamma = 1 / props.gamma;

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, props.width, props.height);
      gl.depthMask(true);
      gl.clearColor(
        Math.pow(background[0], invGamma),
        Math.pow(background[1], invGamma),
        Math.pow(background[2], invGamma),
        1,
      );
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      paintPoints3D(
        props.viewProjection,
        props.pointSize3D,
        cameraDistance,
        sampleCount,
        props.fogDensity,
        props.cameraEye,
        colorMatrix,
        props.gamma,
        background,
      );
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
        // Density uses up to 4 channels regardless of categoryCount; `categoryColors`
        // is also reused by the contours pass below.
        let colorMatrix = buildGammaColorMatrix(props.categoryColors, 4, props.categoryCount, props.gamma);

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

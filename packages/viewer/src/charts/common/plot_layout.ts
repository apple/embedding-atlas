import { solveExtents } from "./solve_extents.js";
import type { IntermediatePositionScale, PlotLayout } from "./types.js";

function sanitize(value: number | undefined) {
  if (typeof value == "number") {
    if (isFinite(value) && value > 0) {
      return value;
    }
  }
}

const defaultWidth = 300;
const defaultHeight = 200;

export function resolvePlotLayout(
  layout: PlotLayout,
  scales?: { x?: IntermediatePositionScale; y?: IntermediatePositionScale },
): {
  width: number;
  height: number;
  plotRect: { x: number; y: number; width: number; height: number };
} {
  let xs: [number, number][] = [];
  let ys: [number, number][] = [];

  if (scales?.x && scales?.x.labels.length > 0) {
    for (let label of scales.x.labels) {
      let p = scales.x.base.apply(label.value);
      xs.push([p[0], p[1] - label.size.width / 2]);
      xs.push([p[0], p[1] + label.size.width / 2]);
      ys.push([1, label.size.height + label.padding]);
    }
  }
  if (scales?.y && scales?.y.labels.length > 0) {
    for (let label of scales.y.labels) {
      let p = scales.y.base.apply(label.value);
      xs.push([0, -label.size.width - label.padding]);
      ys.push([p[0], p[1] - label.size.height / 2]);
      ys.push([p[0], p[1] + label.size.height / 2]);
    }
  }

  let { width, height, plotWidth, plotHeight, plotAspectRatio } = layout;

  width = sanitize(width);
  height = sanitize(height);
  plotWidth = sanitize(plotWidth);
  plotHeight = sanitize(plotHeight);
  plotAspectRatio = sanitize(plotAspectRatio);

  let left = 0;
  let right = 0;
  let top = 0;
  let bottom = 0;

  if (plotAspectRatio == undefined) {
    // Without plotAspectRatio, X and Y dimensions are independent. Layout them separately.
    if (width == undefined && plotWidth != undefined) {
      [left, right] = solveExtents(xs, plotWidth, "inner");
      width = plotWidth + left + right;
    } else {
      width = width ?? defaultWidth;
      [left, right] = solveExtents(xs, width, "outer");
      plotWidth = width - left - right;
    }
    if (height == undefined && plotHeight != undefined) {
      [top, bottom] = solveExtents(ys, plotHeight, "inner");
      height = plotHeight + top + bottom;
    } else {
      height = height ?? defaultHeight;
      [top, bottom] = solveExtents(ys, height, "outer");
      plotHeight = height - top - bottom;
    }
  } else {
    // Constraint: plotWidth / plotHeight == plotAspectRatio.
    if (plotWidth != undefined || plotHeight != undefined) {
      // If one of plotWidth or plotHeight is specified, we can compute the other with aspect ratio.
      if (plotWidth != undefined) {
        plotHeight = plotWidth / plotAspectRatio;
      } else if (plotHeight != undefined) {
        plotWidth = plotHeight! * plotAspectRatio;
      } else {
        throw new Error("unreachable");
      }
      let [left, right] = solveExtents(xs, plotWidth, "inner");
      let [top, bottom] = solveExtents(ys, plotHeight, "inner");
      // Still respect the original width / height if specified. Place the plot area at (left, top)
      width = width ?? plotWidth + left + right;
      height = height ?? plotHeight + top + bottom;
      return { width, height, plotRect: { x: left, y: top, width: plotWidth, height: plotHeight } };
    } else if (height == undefined) {
      // If height is not specified (aka., flexible), then we compute height from width.
      width = width ?? 400;
      [left, right] = solveExtents(xs, width, "outer");
      let plotWidth = width - left - right;
      let plotHeight = plotWidth / plotAspectRatio;
      [top, bottom] = solveExtents(ys, plotHeight, "inner");
      height = plotHeight + top + bottom;
      return { width, height, plotRect: { x: left, y: top, width: plotWidth, height: plotHeight } };
    } else if (width == undefined) {
      // If width is not specified (aka., flexible), then we compute width from height.
      height = height ?? 400;
      [top, bottom] = solveExtents(ys, height, "outer");
      plotHeight = height - top - bottom;
      plotWidth = plotHeight * plotAspectRatio;
      [left, right] = solveExtents(ys, plotWidth, "inner");
      width = plotWidth + left + right;
      return { width, height, plotRect: { x: left, y: top, width: plotWidth, height: plotHeight } };
    } else {
      // If both width and height are specified, we need to fit with both dimensions.
      // First try fit on the X dimension
      [left, right] = solveExtents(xs, width, "outer");
      plotWidth = width - left - right;
      plotHeight = plotWidth / plotAspectRatio;
      [top, bottom] = solveExtents(ys, plotHeight, "inner");
      if (top + bottom + plotHeight > height) {
        // If it doesn't fit for y, use the Y dimension
        [top, bottom] = solveExtents(ys, height, "outer");
        plotHeight = height - top - bottom;
        plotWidth = plotHeight * plotAspectRatio;
        [left, right] = solveExtents(xs, plotWidth, "inner");
      }
    }
  }
  return { width, height, plotRect: { x: left, y: top, width: plotWidth, height: plotHeight } };
}

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  import Label from "./Label.svelte";

  import { makeScale, resolveLabelOverlap } from "./infer.js";
  import { resolvePlotLayout } from "./plot_layout.js";
  import { type ChartTheme } from "./theme.js";
  import type { AxisConfig, ConcreteScale, ScaleConfig, XYFrameProxy } from "./types.js";

  const SAFE_MARGIN = 4;

  interface Props {
    xScale?: ScaleConfig;
    xAxis?: AxisConfig;
    yScale?: ScaleConfig;
    yAxis?: AxisConfig;

    otherScales?: {
      color?: ConcreteScale<string>;
      size?: ConcreteScale<number>;
    };

    width?: number;
    height?: number;

    plotWidth?: number;
    plotHeight?: number;

    plotAspectRatio?: number;

    theme: ChartTheme;

    childrenBelow?: Snippet<[XYFrameProxy]>;
    children?: Snippet<[XYFrameProxy]>;
  }

  let {
    xScale,
    xAxis = {},
    yScale,
    yAxis = {},
    otherScales,
    width,
    height,
    plotWidth,
    plotHeight,
    plotAspectRatio,
    theme,
    children,
    childrenBelow,
  }: Props = $props();

  // Infer intermediate scales
  let intermediateXScale = $derived(xScale ? makeScale(xScale, xAxis, "x", theme) : undefined);
  let intermediateYScale = $derived(yScale ? makeScale(yScale, yAxis, "y", theme) : undefined);

  // Layout
  let layout = $derived(
    resolvePlotLayout(
      { width, height, plotWidth, plotHeight, plotAspectRatio },
      {
        x: intermediateXScale,
        y: intermediateYScale,
      },
    ),
  );

  let viewWidth = $derived(layout.width);
  let viewHeight = $derived(layout.height);
  let plotRectX = $derived(layout.plotRect.x);
  let plotRectY = $derived(layout.plotRect.y);
  let plotRectWidth = $derived(layout.plotRect.width);
  let plotRectHeight = $derived(layout.plotRect.height);

  let concreteXScale = $derived(intermediateXScale?.concrete([0, plotRectWidth]));
  let concreteYScale = $derived(intermediateYScale?.concrete([0, plotRectHeight]));

  let xLabels = $derived(
    intermediateXScale && concreteXScale
      ? resolveLabelOverlap(
          intermediateXScale.labels,
          (label) => ({
            center: concreteXScale.apply(label.value),
            length: label.size.width,
            priority: label.level,
          }),
          { gap: 4 },
        )
      : [],
  );

  let yLabels = $derived(
    intermediateYScale && concreteYScale
      ? resolveLabelOverlap(
          intermediateYScale.labels,
          (label) => ({
            center: concreteYScale.apply(label.value),
            length: label.size.height,
            priority: label.level,
          }),
          { gap: 2 },
        )
      : [],
  );

  let proxy: XYFrameProxy = $derived({
    plotWidth: plotRectWidth,
    plotHeight: plotRectHeight,
    scale: {
      x: concreteXScale,
      y: concreteYScale,
      ...(otherScales ?? {}),
    },
  });
</script>

<div
  style:width="{viewWidth}py"
  style:height="{viewHeight}px"
  style:position="relative"
  style:user-select="none"
  style:-webkit-user-select="none"
  style:cursor="default"
>
  <svg
    width={viewWidth + SAFE_MARGIN * 2}
    height={viewHeight + SAFE_MARGIN * 2}
    style:position="absolute"
    style:left="-{SAFE_MARGIN}px"
    style:top="-{SAFE_MARGIN}px"
  >
    <g transform="translate({SAFE_MARGIN + plotRectX},{SAFE_MARGIN + plotRectY})">
      {@render childrenBelow?.(proxy)}
      {#if intermediateXScale && concreteXScale && xAxis}
        <g>
          {#each intermediateXScale.ticks as tick}
            {@const x = concreteXScale.apply(tick.value)}
            <line
              x1={x}
              y1={plotRectHeight}
              x2={x}
              y2={plotRectHeight + (tick.level == 0 ? 3 : 0)}
              stroke={theme.gridColor}
              stroke-opacity={tick.level == 0 ? 1 : 0.4}
              stroke-linecap="butt"
            />
          {/each}
          {#each intermediateXScale.gridLines as gridLine}
            {@const x = concreteXScale.apply(gridLine.value)}
            {#each concreteYScale?.rangeBands ?? [] as [y1, y2]}
              <line
                x1={x}
                y1={Math.min(y1, y2)}
                x2={x}
                y2={Math.max(y1, y2)}
                stroke={theme.gridColor}
                stroke-opacity={gridLine.level == 0 ? 1 : 0.4}
                stroke-dasharray="1,3"
                stroke-linecap="square"
              />
            {/each}
          {/each}
          {#each xLabels as label}
            <Label label={label} dimension="x" proxy={proxy} color={theme.labelColor} />
          {/each}
        </g>
      {/if}
      {#if intermediateYScale && concreteYScale && yAxis}
        <g>
          {#each intermediateYScale.ticks as tick}
            {@const y = concreteYScale.apply(tick.value)}
            <line
              x1={-(tick.level == 0 ? 3 : 0)}
              y1={y}
              x2={0}
              y2={y}
              stroke={theme.gridColor}
              stroke-opacity={tick.level == 0 ? 1 : 0.4}
              stroke-linecap="butt"
            />
          {/each}
          {#each intermediateYScale.gridLines as gridLine}
            {@const y = concreteYScale.apply(gridLine.value)}
            {#each concreteXScale?.rangeBands ?? [] as [x1, x2]}
              <line
                x1={x1}
                y1={y}
                x2={x2}
                y2={y}
                stroke={theme.gridColor}
                stroke-opacity={gridLine.level == 0 ? 1 : 0.4}
                stroke-linecap="square"
              />
            {/each}
          {/each}
          {#each yLabels as label}
            <Label label={label} dimension="y" proxy={proxy} color={theme.labelColor} />
          {/each}
        </g>
      {/if}
      {@render children?.(proxy)}
    </g>
  </svg>
</div>

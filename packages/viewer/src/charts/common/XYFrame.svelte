<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  import Label from "./Label.svelte";

  import { chartColors } from "./colors.js";
  import { makeScale, resolveLabelOverlap, unionExtents } from "./infer.js";
  import type { AxisSpec, Extents, ScaleSpec, XYFrameProxy } from "./types.js";

  const SAFE_MARGIN = 4;

  interface Props {
    xScale?: ScaleSpec;
    xAxis?: AxisSpec;
    yScale?: ScaleSpec;
    yAxis?: AxisSpec;

    width: number;
    height: number;

    extents?: Extents;

    colorScheme?: "light" | "dark";

    childrenBelow?: Snippet<[XYFrameProxy]>;
    children?: Snippet<[XYFrameProxy]>;
  }

  let {
    xScale,
    xAxis = {},
    yScale,
    yAxis = {},
    width,
    height,
    extents: additionalExtents,
    colorScheme,
    children,
    childrenBelow,
  }: Props = $props();

  // Infer intermediate scales
  let intermediateXScale = $derived(xScale ? makeScale(xScale, xAxis, "x") : null);
  let intermediateYScale = $derived(yScale ? makeScale(yScale, yAxis, "y") : null);

  // Combine extents
  let extents = $derived(
    unionExtents(
      [intermediateXScale?.extents, intermediateYScale?.extents, additionalExtents].filter((x) => x != null),
    ),
  );

  let plotRect = $derived({
    x: extents.left,
    y: extents.top,
    width: width - extents.left - extents.right,
    height: height - extents.top - extents.bottom,
  });

  let concreteXScale = $derived(intermediateXScale?.concrete([0, plotRect.width]));
  let concreteYScale = $derived(intermediateYScale?.concrete([plotRect.height, 0]));

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

  // Colors
  let colors = $derived(chartColors[colorScheme ?? "light"]);

  let proxy: XYFrameProxy = $derived({
    xScale: concreteXScale,
    yScale: concreteYScale,
    plotWidth: plotRect.width,
    plotHeight: plotRect.height,
  });
</script>

<div
  style:width="{width}py"
  style:height="{height}px"
  style:position="relative"
  style:user-select="none"
  style:-webkit-user-select="none"
  style:cursor="default"
>
  <svg
    width={width + SAFE_MARGIN * 2}
    height={height + SAFE_MARGIN * 2}
    style:position="absolute"
    style:left="-{SAFE_MARGIN}px"
    style:top="-{SAFE_MARGIN}px"
  >
    <g transform="translate({SAFE_MARGIN + plotRect.x},{SAFE_MARGIN + plotRect.y})">
      {@render childrenBelow?.(proxy)}
      {#if intermediateXScale && concreteXScale && xAxis}
        <g>
          {#each intermediateXScale.ticks as tick}
            {@const x = concreteXScale.apply(tick.value)}
            <line
              x1={x}
              y1={plotRect.height}
              x2={x}
              y2={plotRect.height + (tick.level == 0 ? 3 : 0)}
              stroke={colors.gridColor}
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
                stroke={colors.gridColor}
                stroke-opacity={gridLine.level == 0 ? 1 : 0.4}
                stroke-dasharray="1,3"
                stroke-linecap="square"
              />
            {/each}
          {/each}
          {#each xLabels as label}
            <Label label={label} dimension="x" proxy={proxy} color={colors.labelColor} />
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
              stroke={colors.gridColor}
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
                stroke={colors.gridColor}
                stroke-opacity={gridLine.level == 0 ? 1 : 0.4}
                stroke-linecap="square"
              />
            {/each}
          {/each}
          {#each yLabels as label}
            <Label label={label} dimension="y" proxy={proxy} color={colors.labelColor} />
          {/each}
        </g>
      {/if}
      {@render children?.(proxy)}
    </g>
  </svg>
</div>

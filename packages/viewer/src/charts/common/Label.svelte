<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { FONT_FAMILY, FONT_SIZE } from "./infer.js";
  import type { Label, XYFrameProxy } from "./types.js";

  interface Props {
    label: Label;
    color: string;
    proxy: XYFrameProxy;
    dimension: "x" | "y";
  }

  let { label, color, dimension, proxy }: Props = $props();

  let { px, py, anchorX, anchorY } = $derived(
    dimension == "x"
      ? {
          px: proxy.xScale?.apply(label.value) ?? 0,
          py: proxy.plotHeight + label.padding,
          anchorX: 0.5,
          anchorY: 0,
        }
      : {
          px: -label.padding,
          py: proxy.yScale?.apply(label.value) ?? 0,
          anchorX: 1,
          anchorY: 0.5,
        },
  );

  let { rotation, shiftX, shiftY, width, height } = $derived(
    label.orientation == "vertical"
      ? {
          rotation: 90,
          shiftX: anchorX * label.size.width,
          shiftY: -anchorY * label.size.height,
          width: label.size.height,
          height: label.size.width,
        }
      : {
          rotation: 0,
          shiftX: -anchorX * label.size.width,
          shiftY: -anchorY * label.size.height,
          width: label.size.width,
          height: label.size.height,
        },
  );

  let marginX = 4;
  let marginY = 4;
</script>

<g transform="translate({px + shiftX}, {py + shiftY}) rotate({rotation})">
  <foreignObject x={-marginX} y={-marginY} width={width + marginX * 2} height={height + marginY * 2}>
    <div
      style:width="{width + 2}px"
      style:height="{height + marginY * 2}px"
      style:line-height="{height + marginY * 2}px"
      style:font-family={FONT_FAMILY}
      style:font-size="{FONT_SIZE}px"
      style:margin-left="{marginX}px"
      style:color={color}
      style:overflow="hidden"
      style:white-space="nowrap"
      style:text-overflow="ellipsis"
      title={label.text}
    >
      {label.text}
    </div>
  </foreignObject>
</g>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { interactionHandler, type CursorValue } from "@embedding-atlas/utils";

  interface Props {
    class: string;
    axis: "x" | "y";
    scaler: number;
    value: number;
    min: number;
    max: number;
    onChange: (value: number) => void;
  }

  let props: Props = $props();

  function startDrag(e1: CursorValue) {
    let x1 = e1.pageX;
    let y1 = e1.pageY;
    let v0 = props.value;
    return {
      move: (e2: CursorValue) => {
        let dx = e2.pageX - x1;
        let dy = e2.pageY - y1;
        let v1 = v0 + (props.axis == "x" ? dx : dy) * props.scaler;
        if (v1 < props.min) v1 = props.min;
        if (v1 > props.max) v1 = props.max;
        props.onChange(v1);
      },
    };
  }
</script>

<div
  class="{props.class} {props.axis == 'x' ? 'cursor-col-resize' : 'cursor-row-resize'}"
  use:interactionHandler={{ drag: startDrag }}
></div>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
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

  function startDrag(e1: MouseEvent, callback: (dx: number, dy: number) => void) {
    e1.preventDefault();
    let x1 = e1.pageX;
    let y1 = e1.pageY;
    let mousemove = (e2: MouseEvent) => {
      e2.preventDefault();
      let dx = e2.pageX - x1;
      let dy = e2.pageY - y1;
      callback(dx, dy);
    };
    let mouseup = () => {
      window.removeEventListener("mousemove", mousemove);
      window.removeEventListener("mouseup", mouseup);
    };
    window.addEventListener("mousemove", mousemove);
    window.addEventListener("mouseup", mouseup);
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="{props.class} {props.axis == 'x' ? 'cursor-col-resize' : 'cursor-row-resize'}"
  onmousedown={(e1) => {
    let v0 = props.value;
    startDrag(e1, (dx, dy) => {
      let v1 = v0 + (props.axis == "x" ? dx : dy) * props.scaler;
      if (v1 < props.min) v1 = props.min;
      if (v1 > props.max) v1 = props.max;
      props.onChange(v1);
    });
  }}
></div>

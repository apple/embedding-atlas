<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    value?: number;
    min?: number;
    max?: number;
    step?: number | null | undefined;
    width?: number;
    /** Fill the width of the container (`width` is ignored). */
    fill?: boolean;
  }

  let { value = $bindable(0), min = 0, max = 100, step = undefined, width = 100, fill = false }: Props = $props();

  let height: number = 28;

  let trackHeight: number = 4;
  let knobSize: number = 15;

  let measuredWidth = $state(0);
  let effectiveWidth = $derived(fill ? Math.max(measuredWidth, knobSize + 1) : width);

  let scale = $derived((v: number) => ((v - min) / (max - min)) * (effectiveWidth - knobSize));
  let inverseScale = $derived((x: number) => (x / (effectiveWidth - knobSize)) * (max - min) + min);

  let knobLeft = $derived(scale(value));

  let container: HTMLDivElement | undefined = $state();

  function setValueWithPosition(x: number) {
    let v = inverseScale(x - knobSize / 2);
    v = Math.max(min, Math.min(max, v));
    if (step != null) {
      v = Math.round(v / step) * step;
    }
    value = v;
  }

  function onMouseDown(e1: MouseEvent) {
    setValueWithPosition(e1.clientX - container!.getBoundingClientRect().left);
    let onMove = (e2: MouseEvent) => {
      setValueWithPosition(e2.clientX - container!.getBoundingClientRect().left);
    };
    let onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key == "ArrowLeft") {
      value = Math.max(value - (step ?? 1), min);
    } else if (e.key == "ArrowRight") {
      value = Math.min(value + (step ?? 1), max);
    }
  }
</script>

<div
  bind:this={container}
  bind:clientWidth={measuredWidth}
  style:width={fill ? "100%" : `${width}px`}
  style:height="{height}px"
  class="group relative"
  onmousedown={onMouseDown}
  onkeydown={onKeyDown}
  role="slider"
  aria-valuenow={value}
  aria-valuemin={min}
  aria-valuemax={max}
  tabindex="0"
>
  <div
    class="bg-slate-400 dark:bg-slate-500 rounded-full absolute"
    style:left="0px"
    style:top="{(height - trackHeight) / 2}px"
    style:width="{effectiveWidth}px"
    style:height="{trackHeight}px"
  ></div>
  <div
    class="bg-blue-500 rounded-full absolute group-hover:bg-blue-600 dark:group-hover:bg-blue-400"
    style:left="{knobLeft}px"
    style:top="{(height - knobSize) / 2}px"
    style:width="{knobSize}px"
    style:height="{knobSize}px"
  ></div>
</div>

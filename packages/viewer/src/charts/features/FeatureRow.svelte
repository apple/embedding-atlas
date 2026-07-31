<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<!-- A single feature row, reused in both the pinned section and the main list. -->
<script lang="ts">
  import { IconPin, IconPinRegular, IconSelected, IconUnselected } from "../../assets/icons.js";
  import BinaryPredictBar from "./BinaryPredictBar.svelte";
  import CountBar from "./CountBar.svelte";
  import MulticlassPredictBar from "./MulticlassPredictBar.svelte";
  import type { ListItem } from "./features_list_store.js";

  interface Props {
    item: ListItem;
    selected: boolean;
    pinned: boolean;
    hasPredict: boolean;
    isBinary: boolean;
    maxCount: number;
    maxStrength: number;
    segmented: boolean;
    segmentColors: Record<string, string>;
    directionColors: { left: string; right: string };
    classNames: string[];
    markColor: string;
    markColorFade: string;
    /** Precomputed by the parent (depends on predict mode). */
    countLabel: string;
    tooltip: string;
    /** Plain vs shift click on the row body. */
    onRowClick: (shift: boolean) => void;
    onToggleSelect: () => void;
    onTogglePin: () => void;
  }

  let {
    item,
    selected,
    pinned,
    hasPredict,
    isBinary,
    maxCount,
    maxStrength,
    segmented,
    segmentColors,
    directionColors,
    classNames,
    markColor,
    markColorFade,
    countLabel,
    tooltip,
    onRowClick,
    onToggleSelect,
    onTogglePin,
  }: Props = $props();
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="col-span-5 grid grid-cols-subgrid items-center rounded hover:bg-gray-100 dark:hover:bg-gray-900 transition-colors duration-150 h-[24px]"
  class:!bg-blue-100={selected}
  class:dark:!bg-blue-800={selected}
  onclick={(e) => onRowClick(e.shiftKey)}
>
  <!-- Leading pin / select buttons. The container stretches to the full row
       height (self-stretch) so each button's clickable region spans the row;
       horizontal padding is unchanged so the buttons don't get wider. -->
  <div class="flex items-stretch gap-0.5 pl-0.5 self-stretch">
    <button
      type="button"
      class="flex items-center px-0.5 transition-colors duration-150 hover:text-blue-500"
      class:text-blue-500={pinned}
      class:text-slate-400={!pinned}
      class:dark:text-slate-500={!pinned}
      title={pinned ? "Unpin feature" : "Pin feature"}
      onclick={(e) => {
        e.stopPropagation();
        onTogglePin();
      }}
    >
      {#if pinned}
        <IconPin class="w-4 h-4" />
      {:else}
        <IconPinRegular class="w-4 h-4" />
      {/if}
    </button>
    <button
      type="button"
      class="flex items-center px-0.5 transition-colors duration-150 hover:text-blue-500"
      class:text-blue-500={selected}
      class:text-slate-400={!selected}
      class:dark:text-slate-500={!selected}
      title={selected ? "Remove from selection" : "Add to selection"}
      onclick={(e) => {
        e.stopPropagation();
        onToggleSelect();
      }}
    >
      {#if selected}
        <IconSelected class="w-4 h-4" />
      {:else}
        <IconUnselected class="w-4 h-4" />
      {/if}
    </button>
  </div>

  <!-- Feature name (marquee-scrolls on hover when it overflows) -->
  <div
    class="truncate"
    title={tooltip}
    onmouseenter={(e) => {
      const el = e.currentTarget;
      const overflow = el.scrollWidth - el.clientWidth;
      if (overflow > 0) {
        const span = el.firstElementChild as HTMLElement;
        const duration = Math.max(0.75, overflow / 70);
        el.style.textOverflow = "clip";
        span.style.display = "inline-block";
        span.style.transition = `transform ${duration}s ease-out`;
        span.style.transform = `translateX(-${overflow}px)`;
      }
    }}
    onmouseleave={(e) => {
      const el = e.currentTarget;
      const span = el.firstElementChild as HTMLElement;
      span.style.transition = "";
      span.style.transform = "";
      span.style.display = "";
      el.style.textOverflow = "";
    }}
  >
    <span>{item.feature}</span>
  </div>

  <div class="text-xs text-right text-slate-500 dark:text-slate-400">
    {countLabel}
  </div>

  <div>
    {#if hasPredict && item.predict != null}
      {#if isBinary}
        <BinaryPredictBar
          predict={item.predict}
          maxStrength={maxStrength}
          width={80}
          height={24}
          directionColors={directionColors}
          classNames={classNames}
        />
      {:else}
        <MulticlassPredictBar
          predict={item.predict}
          maxStrength={maxStrength}
          width={80}
          height={24}
          markColor={markColor}
        />
      {/if}
    {:else}
      <CountBar
        item={item}
        maxCount={maxCount}
        width={80}
        height={10}
        segmented={segmented}
        segmentColors={segmentColors}
        markColor={markColor}
        markColorFade={markColorFade}
      />
    {/if}
  </div>

  <div class="text-xs text-right font-medium">
    {#if hasPredict && item.predict != null}
      {#if isBinary}
        <span style:color={directionColors[item.predict?.direction == 0 ? "left" : "right"]}>
          {(item.predict.phi ?? 0).toFixed(2)}
        </span>
      {:else}
        <span style:color={markColor}>
          {(item.predict.strength ?? 0).toFixed(2)}
        </span>
      {/if}
    {/if}
  </div>
</div>

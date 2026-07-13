<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "./Button.svelte";

  interface Props {
    pointCount: number;
    totalCount: number;
    /** Whether a range-selection/cross-filter is currently active elsewhere in the app. 3D has
     *  no persistent brush outline to draw for it (unlike 2D's lasso/marquee overlay), so this
     *  drives a non-silent status-bar badge instead of leaving the active filter invisible. */
    activeFilterHidden?: boolean;
    onResetCamera: () => void;
    selectionMode?: "marquee" | "lasso" | "none";
    onSelectionMode?: (mode: "marquee" | "lasso" | "none") => void;
  }

  let {
    pointCount,
    totalCount,
    activeFilterHidden = false,
    onResetCamera,
    selectionMode = "none",
    onSelectionMode,
  }: Props = $props();
</script>

<div
  style:font-size="12px"
  style:line-height="20px"
  style:height="20px"
  style:color="white"
  style:position="absolute"
  style:bottom="0px"
  style:left="0px"
  style:right="0px"
  style:user-select="none"
  style:pointer-events="none"
  style:display="flex"
  style:flex-direction="row"
>
  <div
    style:flex="none"
    style:display="flex"
    style:flex-direction="row"
    style:align-items="center"
    style:gap="4px"
    style:padding="0px 4px"
    style:margin="4px"
    style:border-radius="2px"
    style:background="rgba(0,0,0,0.55)"
    style:pointer-events="auto"
  >
    <Button icon="reset" title="Reset camera to default view" onClick={onResetCamera} />
    {#if onSelectionMode}
      <Button
        icon="marquee"
        active={selectionMode == "marquee"}
        title="Toggle rectangle selection mode (drag to select points; disables camera orbit while dragging). This is a one-shot snapshot selection, not a persistent, re-evaluable region like 2D's lasso."
        onClick={() => onSelectionMode(selectionMode == "marquee" ? "none" : "marquee")}
      />
      <Button
        icon="lasso"
        active={selectionMode == "lasso"}
        title="Toggle lasso selection mode (drag to select points; disables camera orbit while dragging). This is a one-shot snapshot selection, not a persistent, re-evaluable region like 2D's lasso."
        onClick={() => onSelectionMode(selectionMode == "lasso" ? "none" : "lasso")}
      />
    {/if}
    <span>
      {#if pointCount < totalCount}
        {pointCount.toLocaleString()} / {totalCount.toLocaleString()} points (downsampled)
      {:else}
        {totalCount.toLocaleString()} points
      {/if}
    </span>
  </div>
  <div style:flex="1 1 0%"></div>
  {#if activeFilterHidden}
    <div
      style:flex="none"
      style:display="flex"
      style:flex-direction="row"
      style:align-items="center"
      style:gap="4px"
      style:padding="0px 6px"
      style:margin="4px"
      style:border-radius="2px"
      style:background="rgba(220,50,50,0.85)"
      style:pointer-events="auto"
    >
      <span title="A filter is active. Its outline can't be drawn in 3D, but it is still applied.">Filter active</span>
    </div>
  {/if}
</div>

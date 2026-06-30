<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "./Button.svelte";
  import MapScaleLegend from "./MapScaleLegend.svelte";

  import type { ThemeConfig } from "./theme.js";

  interface Props {
    resolvedTheme: ThemeConfig;
    statusMessage: string | null;
    pointCount: number;
    distancePerPoint: number;

    selectionMode: "marquee" | "lasso" | "none";
    onSelectionMode: (v: "marquee" | "lasso" | "none") => void;

    is3D?: boolean;
    onResetCamera?: () => void;
    // A selection brush is active but its outline is hidden (3D mode). Surfaced so the
    // filter is never silent, with a one-click clear.
    filterActive?: boolean;
    onClearFilter?: () => void;
  }

  let {
    resolvedTheme,
    statusMessage = null,
    pointCount,
    distancePerPoint,
    selectionMode,
    onSelectionMode,
    is3D = false,
    onResetCamera,
    filterActive = false,
    onClearFilter,
  }: Props = $props();
</script>

<div
  style:font-size="12px"
  style:line-height="20px"
  style:height="20px"
  style:color={resolvedTheme.statusBarTextColor}
  style:position="absolute"
  style:bottom="0px"
  style:left="0px"
  style:right="0px"
  style:user-select="none"
  style:font-family={resolvedTheme.fontFamily}
  style:display="flex"
  style:flex-direction="row"
>
  <div
    style:flex="none"
    style:display="flex"
    style:flex-direction="row"
    style:gap="4px"
    style:padding="0px 4px"
    style:border-radius="2px"
    style:background={resolvedTheme.statusBarBackgroundColor}
  >
    {#if statusMessage != null}
      <div style:display="inline-block">
        {statusMessage}
      </div>
    {/if}
  </div>
  <div style:flex="1 1 0%"></div>
  <div
    style:flex="none"
    style:display="flex"
    style:flex-direction="row"
    style:align-items="center"
    style:gap="4px"
    style:padding="0px 4px"
    style:border-radius="2px"
    style:background={resolvedTheme.statusBarBackgroundColor}
  >
    {#snippet divider()}
      <div style="border-right: 1px solid currentColor; margin: 4px 2px; opacity: 0.3; width: 0; height: 10px"></div>
    {/snippet}
    {#if resolvedTheme.brandingLink != null}
      <a
        href={resolvedTheme.brandingLink.href}
        target="_blank"
        rel="noopener noreferrer"
        style:color="currentColor"
        style:text-decoration="underline"
      >
        {resolvedTheme.brandingLink.text}
      </a>
      {@render divider()}
    {/if}
    {#if is3D}
      {#if filterActive}
        <!-- The brush outline is hidden in 3D, so surface the active filter explicitly
             (never a silent hidden filter) with a one-click clear. -->
        <Button
          icon="filter"
          active={true}
          title="A selection filter is active (its brush outline is hidden in 3D). Click to clear it."
          onClick={() => onClearFilter?.()}
        />
        {@render divider()}
      {/if}
      <Button
        icon="reset"
        title="Reset the 3D camera to fit all points. Drag to orbit, shift + drag to pan, scroll to zoom, double-click a point to focus."
        onClick={() => onResetCamera?.()}
      />
      {@render divider()}
    {:else}
      <Button
        icon="marquee"
        active={selectionMode == "marquee"}
        title="Toggle rectangle selection mode. In normal mode, use shift + drag for rectangle selection."
        onClick={() => onSelectionMode(selectionMode == "marquee" ? "none" : "marquee")}
      />
      <Button
        icon="lasso"
        active={selectionMode == "lasso"}
        title="Toggle lasso selection mode. In normal mode, use shift + meta + drag for lasso selection."
        onClick={() => onSelectionMode(selectionMode == "lasso" ? "none" : "lasso")}
      />
      {@render divider()}
      <MapScaleLegend distancePerPoint={distancePerPoint} />
      {@render divider()}
    {/if}
    <span>{pointCount.toLocaleString()} points</span>
  </div>
</div>

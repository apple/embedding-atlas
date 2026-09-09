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
  }

  let {
    resolvedTheme,
    statusMessage = null,
    pointCount,
    distancePerPoint,
    selectionMode,
    onSelectionMode,
  }: Props = $props();
</script>

<div
  style:font-size="calc(12px * var(--ea-toolbar-scale, 1))"
  style:line-height="calc(20px * var(--ea-toolbar-scale, 1))"
  style:height="calc(20px * var(--ea-toolbar-scale, 1))"
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
    style:gap="calc(4px * var(--ea-toolbar-scale, 1))"
    style:padding="0px calc(4px * var(--ea-toolbar-scale, 1))"
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
    style:gap="calc(4px * var(--ea-toolbar-scale, 1))"
    style:padding="0px calc(4px * var(--ea-toolbar-scale, 1))"
    style:border-radius="2px"
    style:background={resolvedTheme.statusBarBackgroundColor}
  >
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
      <div
        style="border-right: 1px solid currentColor; margin: calc(4px * var(--ea-toolbar-scale, 1)) calc(2px * var(--ea-toolbar-scale, 1)); opacity: 0.3; width: 0; height: calc(10px * var(--ea-toolbar-scale, 1))"
      ></div>
    {/if}
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
    <div
      style="border-right: 1px solid currentColor; margin: calc(4px * var(--ea-toolbar-scale, 1)) calc(2px * var(--ea-toolbar-scale, 1)); opacity: 0.3; width: 0; height: calc(10px * var(--ea-toolbar-scale, 1))"
    ></div>
    <MapScaleLegend distancePerPoint={distancePerPoint} />
    <div
      style="border-right: 1px solid currentColor; margin: calc(4px * var(--ea-toolbar-scale, 1)) calc(2px * var(--ea-toolbar-scale, 1)); opacity: 0.3; width: 0; height: calc(10px * var(--ea-toolbar-scale, 1))"
    ></div>
    <span>{pointCount.toLocaleString()} points</span>
  </div>
</div>

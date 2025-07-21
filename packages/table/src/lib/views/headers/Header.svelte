<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import HeaderTitle from "./HeaderTitle.svelte";
  import RowNumberTitle from "./RowNumberTitle.svelte";
  import SortButtons from "./SortButtons.svelte";

  import { ConfigContext } from "../../context/config.svelte";
  import { Context } from "../../context/context.svelte";
  import { CustomHeadersContext } from "../../context/custom-headers.svelte";
  import { OID } from "../../model/TableModel.svelte.js";
  import CustomHeaderContents from "./CustomHeaderContents.svelte";

  interface Props {
    col: string;
  }

  let { col }: Props = $props();

  const model = Context.model;
  const schema = Context.schema;
  const config = ConfigContext.config;
  const customHeadersConfig = CustomHeadersContext.config;

  let element: HTMLElement | null = $state(null);
  let contentWidth: number = $state(0);

  $effect(() => {
    config.minColumnWidths[col] = contentWidth + config.betweenColPadding;
  });

  const width = $derived(model.colWidths[col]);
  const isNumber = $derived((schema.dataType[col] ?? "string") === "number");
  const numberClass = $derived(isNumber || col === OID ? "number" : "");
  const isFirstCol = $derived(model.isFirstCol(col));
  const isLastCol = $derived(model.isLastCol(col));
  let headerHeight = $derived(config.headerHeight ? config.headerHeight + "px" : "auto");
</script>

<div
  class="header-cell {numberClass}"
  bind:this={element}
  style:--width={width + "px"}
  style:--height={headerHeight}
  style:--padding-x={config.betweenColPadding + "px"}
  style:--extra-padding-right={(isLastCol ? config.verticalScrollbarWidth : 0) + "px"}
  style:--extra-padding-left={(isFirstCol ? config.firstColLeftPadding : 0) + "px"}
>
  <div class="header-content" bind:clientWidth={contentWidth}>
    {#if customHeadersConfig[col]}
      <CustomHeaderContents col={col} customHeader={customHeadersConfig[col]} />
    {/if}
    <div class="header-title">
      {#if col !== OID}
        <HeaderTitle col={col} />
        <SortButtons col={col} />
      {:else}
        <RowNumberTitle />
      {/if}
    </div>
  </div>
</div>

<style>
  .header-cell {
    position: relative;
    display: flex;
    flex-direction: row;
    align-items: end;

    width: var(--width);
    min-height: var(--height);
    flex-shrink: 0;
    box-sizing: border-box;
    padding: 0.25em;
    padding-right: calc(calc(var(--padding-x) / 2) + var(--extra-padding-right));
    padding-left: calc(calc(var(--padding-x) / 2) + var(--extra-padding-left));

    color: var(--secondary-text-color);
    font-family: var(--header-font-family);
    font-size: var(--header-font-size);
  }

  .header-cell.number {
    justify-content: end;
  }

  .header-content {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }

  .header-title {
    height: 1.5em;
    align-items: center;
    display: flex;
    flex-direction: row;
    flex-shrink: 0;
  }
</style>

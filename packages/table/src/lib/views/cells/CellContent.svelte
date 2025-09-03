<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { JSType } from "@uwdata/mosaic-core";

  import BigIntContent from "./cell-contents/BigIntContent.svelte";
  import CustomCellContents from "./cell-contents/CustomCellContents.svelte";
  import ImageContent from "./cell-contents/ImageContent.svelte";
  import LinkContent from "./cell-contents/LinkContent.svelte";
  import NumberContent from "./cell-contents/NumberContent.svelte";
  import TextContent from "./cell-contents/TextContent.svelte";

  import { ConfigContext } from "../../context/config.svelte";
  import { Context } from "../../context/context.svelte";
  import { CustomCellsContext } from "../../context/custom-cells.svelte";

  interface Props {
    row: string;
    col: string;
    hovered: boolean;
  }

  const { row, col, hovered }: Props = $props();

  const model = Context.model;
  const controller = Context.controller;
  const schema = Context.schema;
  const config = ConfigContext.config;
  const customCellsConfig = CustomCellsContext.config;

  let height: number = $state(0);
  let contentHeight = $state(0);

  let clamped = $derived(contentHeight > height);

  const content: string | null = model.getContent({ row, col });
  const type: JSType = schema.dataType[col] ?? "string";
  const sqlType: string = schema.sqlType[col] ?? "TEXT";

  function isLink(value: any): boolean {
    return typeof value == "string" && (value.startsWith("http://") || value.startsWith("https://"));
  }

  function isImage(value: any): boolean {
    if (value == null) {
      return false;
    }
    if (typeof value == "string" && value.startsWith("data:image/")) {
      return true;
    }
    if (value.bytes && value.bytes instanceof Uint8Array) {
      // TODO: check if the bytes are actually an image.
      return true;
    }
    return false;
  }
</script>

<div
  class="cell-content clamp"
  bind:clientHeight={height}
  style:--lineHeight={config.lineHeight + "px"}
  style:--num-lines={config.textMaxLines}
>
  {#if customCellsConfig[col]}
    <CustomCellContents row={row} col={col} customCell={customCellsConfig[col]} bind:height={contentHeight} />
  {:else if type === "string"}
    {#if content && isLink(content)}
      <LinkContent url={content} bind:height={contentHeight} />
    {:else}
      <TextContent text={content} bind:height={contentHeight} clamped={clamped} parentHeight={height} />
    {/if}
  {:else if type === "number"}
    {#if sqlType === "BIGINT"}
      <BigIntContent bigint={BigInt((content as string) ?? "")} bind:height={contentHeight} />
    {:else}
      <NumberContent number={content as number | null} bind:height={contentHeight} />
    {/if}
  {:else if isImage(content)}
    <ImageContent image={content} bind:height={contentHeight} />
  {:else}
    <TextContent text={content} bind:height={contentHeight} clamped={clamped} parentHeight={height} />
  {/if}

  {#if clamped}
    <button
      class="expand-button {hovered ? 'show' : 'hide'}"
      onclick={() => {
        controller.addHeightToRow(row, contentHeight - height);
      }}>↘</button
    >
  {/if}
</div>

<style>
  .cell-content {
    position: relative;
    flex-grow: 1;
    line-height: var(--lineHeight);
    overflow-wrap: anywhere;
    overflow: hidden;
  }

  .expand-button {
    all: unset;
    visibility: hidden;
    position: absolute;
    bottom: 0;
    right: 0;
    cursor: pointer;
    font-size: 12px;
    line-height: 18px;
    padding-left: 4px;
    padding-right: 4px;
    border-radius: 2px;

    color: var(--secondary-text-color);
    background-color: var(--background-color);
    border: var(--outline);
  }

  .expand-button.show {
    visibility: visible;
  }
</style>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { slide } from "svelte/transition";

  import ConfirmButton from "../../charts/builder/ConfirmButton.svelte";
  import SpecEditor from "../../charts/builder/SpecEditor.svelte";
  import CornerButton from "../../widgets/CornerButton.svelte";

  import {
    IconCheck,
    IconChevronDown,
    IconChevronUp,
    IconClose,
    IconDown,
    IconEdit,
    IconUp,
  } from "../../assets/icons.js";

  import { findChartTypeOptions } from "../../charts/chart_types.js";
  import type { LayoutProps } from "../layout.js";

  interface Props {
    id: string;
    spec: any;

    isVisible: boolean;

    colorScheme: "light" | "dark";

    onIsVisibleChange?: (value: boolean) => void;

    onUp?: () => void;
    onDown?: () => void;
    onRemove?: () => void;

    onSpecChange?: (spec: any) => void;

    chartView: LayoutProps<unknown>["chartView"];
  }

  let { id, spec, isVisible, colorScheme, chartView, onIsVisibleChange, onUp, onDown, onRemove, onSpecChange }: Props =
    $props();

  let isEditing = $state(false);
  let supportsEditMode = $derived(findChartTypeOptions(spec).supportsEditMode ?? false);
  let chartMode: "edit" | "view" = $derived(supportsEditMode && isEditing ? "edit" : "view");

  let editingSpec = $state.raw<any | undefined>();
</script>

<div class="p-2 flex items-center">
  <button
    class="font-mono font-medium h-6 text-left flex flex-1 mr-2 overflow-hidden items-center"
    onclick={() => onIsVisibleChange?.(!isVisible)}
  >
    {#if isVisible}
      <div class="text-sm pr-0.5"><IconChevronUp /></div>
    {:else}
      <div class="text-sm pr-0.5"><IconChevronDown /></div>
    {/if}
    <div class="flex-1 whitespace-nowrap overflow-hidden text-ellipsis">
      {spec.title}
    </div>
  </button>
  <div class="flex-none flex gap-1 sm:opacity-0 group-hover:opacity-100">
    {#if onSpecChange}
      <CornerButton
        icon={chartMode == "edit" ? IconCheck : IconEdit}
        title="Edit"
        onClick={() => {
          isEditing = !isEditing;
          if (isEditing) {
            editingSpec = spec;
            onIsVisibleChange?.(true);
          }
        }}
      />
    {/if}
    {#if onUp}
      <CornerButton icon={IconUp} title="Move up" onClick={onUp} />
    {/if}
    {#if onDown}
      <CornerButton icon={IconDown} title="Move down" onClick={onDown} />
    {/if}
    {#if onRemove}
      <CornerButton icon={IconClose} title="Close" onClick={() => onRemove()} />
    {/if}
  </div>
</div>
<div
  style:display="grid"
  style:grid-template-rows={isVisible ? "1fr" : "0fr"}
  style:transition="grid-template-rows 300ms ease-in-out"
>
  <div class="overflow-hidden">
    {@render chartView({ id: id, width: "container", mode: chartMode })}
    {#if chartMode == "view" && isEditing && onSpecChange}
      <div transition:slide class="h-96">
        <div class="w-full h-96 p-2 flex flex-col gap-2">
          <SpecEditor
            class="w-full flex-1 min-h-0"
            initialValue={editingSpec}
            colorScheme={colorScheme}
            onChange={(s) => {
              editingSpec = s;
            }}
          />
          <ConfirmButton
            label="Confirm"
            disabled={editingSpec == undefined}
            onClick={() => {
              if (editingSpec != undefined) {
                onSpecChange(editingSpec);
                isEditing = false;
              }
            }}
          />
        </div>
      </div>
    {/if}
  </div>
</div>

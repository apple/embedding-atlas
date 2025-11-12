<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { slide } from "svelte/transition";

  import SpecEditor from "../../charts/builder/SpecEditor.svelte";
  import CornerButton from "../../widgets/CornerButton.svelte";

  import { IconChevronDown, IconChevronUp, IconClose, IconDown, IconEdit, IconUp } from "../../assets/icons.js";
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
</script>

<div class="px-2 pt-2 flex items-center">
  <button
    class="font-mono font-medium py-0.5 text-left flex flex-1 mr-2 overflow-hidden items-center"
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
  <div class="flex-none flex gap-1 sm:opacity-0 group-hover:opacity-100 pr-0.5">
    {#if onSpecChange}
      <CornerButton icon={IconEdit} title="Edit spec" onClick={() => (isEditing = !isEditing)} />
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
    {@render chartView({ id: id, width: "container" })}
    {#if isEditing && onSpecChange}
      <div transition:slide class="h-64">
        <div class="w-full h-64 p-2">
          <SpecEditor
            spec={spec}
            colorScheme={colorScheme}
            onSpecChange={(s) => {
              onSpecChange(s);
              isEditing = false;
            }}
          />
        </div>
      </div>
    {/if}
  </div>
</div>

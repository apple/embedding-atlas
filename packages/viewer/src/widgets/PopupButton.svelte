<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { type Snippet } from "svelte";

  import ToggleButton from "./ToggleButton.svelte";

  import { autoUpdate, computePosition, offset, shift } from "@floating-ui/dom";

  interface Props {
    title?: string;
    label?: string | null;
    icon?: any | null;
    anchor?: "left" | "right";
    children?: Snippet;
  }

  let { title = "", label = null, icon = null, anchor = "right", children }: Props = $props();

  let visible: boolean = $state(false);
  let container: HTMLDivElement;

  function onKeyDown(e: KeyboardEvent) {
    if (visible && e.key == "Escape") {
      visible = false;
      e.stopPropagation();
    }
  }

  let popoverElement: HTMLElement;

  $effect(() => {
    if (visible) {
      $effect(() => {
        popoverElement.showPopover();

        function updatePosition() {
          computePosition(container, popoverElement, {
            placement: "bottom",
            middleware: [offset(3), shift()],
          }).then(({ x, y }) => {
            popoverElement.style.left = x + "px";
            popoverElement.style.top = y + "px";
          });
        }
        return autoUpdate(container, popoverElement, updatePosition);
      });
    }
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="relative" bind:this={container} onkeydown={onKeyDown}>
  <ToggleButton icon={icon} title={title} label={label} bind:checked={visible} />
  <div
    bind:this={popoverElement}
    class="absolute px-3 py-3 rounded-md z-20 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 shadow-lg"
    style:width="max-content"
    popover
    ontoggle={(e) => {
      if (e.newState == "closed") {
        visible = false;
      }
    }}
  >
    {@render children?.()}
  </div>
</div>

<svelte:window />

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { type Snippet } from "svelte";

  import ToggleButton from "./ToggleButton.svelte";

  import { autoUpdate, computePosition, offset, shift, type Placement } from "@floating-ui/dom";

  interface Props {
    title?: string;
    label?: string | null;
    icon?: any | null;
    placement?: Placement;
    visible?: boolean;
    button?: Snippet<[{ visible: boolean; toggle: () => void }]>;
    children?: Snippet;
  }

  let {
    title = "",
    label = null,
    icon = null,
    placement = "bottom",
    visible = $bindable(false),
    children,
    button,
  }: Props = $props();

  function toggle() {
    visible = !visible;
  }
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
            placement: placement,
            middleware: [offset(3), shift()],
          }).then(({ x, y }) => {
            popoverElement.style.left = x + "px";
            popoverElement.style.top = y + "px";
          });
        }
        return autoUpdate(container, popoverElement, updatePosition);
      });
    } else {
      popoverElement.hidePopover();
    }
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="relative" bind:this={container} onkeydown={onKeyDown}>
  {#if button}
    {@render button({ visible, toggle })}
  {:else}
    <ToggleButton icon={icon} title={title} label={label} bind:checked={visible} />
  {/if}
  <div
    bind:this={popoverElement}
    class="absolute p-4 rounded-md z-20 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 shadow-lg"
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

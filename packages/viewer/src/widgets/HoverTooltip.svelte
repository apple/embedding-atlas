<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { autoUpdate, computePosition, flip, offset, shift, type Placement } from "@floating-ui/dom";
  import type { Snippet } from "svelte";

  interface Props {
    /** Trigger content the tooltip is anchored to. */
    children: Snippet;
    /** Tooltip body, rendered inside the floating panel. */
    content: Snippet;
    /** Preferred placement relative to the trigger. */
    placement?: Placement;
    /** Delay in milliseconds before the tooltip appears on hover. */
    delay?: number;
    /** Extra classes for the trigger wrapper. */
    class?: string;
  }

  let { children, content, placement = "top", delay = 100, class: className }: Props = $props();

  let trigger: HTMLSpanElement;
  let panel: HTMLDivElement;
  let visible = $state(false);
  let timer: ReturnType<typeof setTimeout> | undefined;

  function open() {
    clearTimeout(timer);
    timer = setTimeout(() => (visible = true), delay);
  }

  function close() {
    clearTimeout(timer);
    visible = false;
  }

  $effect(() => {
    if (visible) {
      panel.showPopover();
      function updatePosition() {
        computePosition(trigger, panel, {
          strategy: "fixed",
          placement: placement,
          middleware: [offset(6), flip(), shift({ padding: 6 })],
        }).then(({ x, y }) => {
          panel.style.left = `${x}px`;
          panel.style.top = `${y}px`;
        });
      }
      return autoUpdate(trigger, panel, updatePosition);
    } else {
      panel.hidePopover();
    }
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={trigger} class={className} onmouseenter={open} onmouseleave={close} onfocusin={open} onfocusout={close}>
  {@render children()}
</div>

<div
  bind:this={panel}
  popover="manual"
  class="fixed text-sm m-0 z-30 w-max max-w-xs rounded-md px-2.5 py-1.5 leading-relaxed text-slate-700 dark:text-slate-200 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-600 shadow-lg"
>
  {@render content()}
</div>

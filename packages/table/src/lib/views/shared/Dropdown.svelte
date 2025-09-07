<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";
  import TablePortal from "./TablePortal.svelte";

  interface Props {
    relativeTo: HTMLElement;
    children: Snippet;
    label: string;
  }

  let { relativeTo, children, label }: Props = $props();

  let open = $state(false);
  let element: HTMLElement | null = $state(null);
  let portalElement: HTMLElement | null = $state(null);
</script>

<button
  class="dropdown {open ? 'unclickable' : 'clickable'}"
  bind:this={element}
  onclick={(e) => {
    open = true;
  }}
>
  {label}
</button>
<svelte:window
  onclick={(e) => {
    if (open && e.target !== element) {
      open = false;
    }
  }}
/>

{#if open}
  <TablePortal
    bind:element={portalElement}
    relativeTo={relativeTo}
    anchor="outside"
    horizontalAlign="left"
    verticalAlign="top"
    stickyX={false}
  >
    {@render children()}
  </TablePortal>
{/if}

<style>
  .dropdown {
    all: unset;
    padding-left: 8px;
    padding-right: 8px;
    border-radius: 2px;
    cursor: pointer;
    color: var(--secondary-text-color);
    position: relative;
    user-select: none;
  }

  .dropdown::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    background-color: var(--primary-bg);
    z-index: -1;
  }

  .dropdown:hover {
    background-color: var(--hover-bg);
  }

  .unclickable {
    pointer-events: none;
  }
</style>

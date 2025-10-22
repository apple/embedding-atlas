<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script>
  import { onDestroy, onMount } from "svelte";

  import { Table as Component } from "../dist/index.js";

  /** @type import("../dist/index.js").TableProps */
  let props = $props();

  /** @type {HTMLElement} */
  let container;

  onMount(() => {
    let component = new Component(container, props);

    $effect(() => {
      component.update(props);
    });

    onDestroy(() => {
      component.destroy();
    });
  });
</script>

<div bind:this={container} style:width="100%" style:height="100%"></div>

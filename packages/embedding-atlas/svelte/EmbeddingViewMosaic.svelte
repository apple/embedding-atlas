<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script>
  import { onMount } from "svelte";

  import { EmbeddingViewMosaic as Component } from "../dist/index.js";

  /** @type import("../dist/index.js").EmbeddingViewMosaicProps */
  let props = $props();

  /** @type {HTMLElement} */
  let container;

  onMount(() => {
    let component = new Component(container, props);

    $effect(() => {
      component.update(props);
    });

    return () => {
      component.destroy();
    };
  });
</script>

<div bind:this={container} style:display="flex"></div>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script>
  import { onMount } from "svelte";

  import { EmbeddingView3DMosaic as Component } from "../dist/index.js";

  /** @type import("../dist/index.js").EmbeddingView3DMosaicProps */
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

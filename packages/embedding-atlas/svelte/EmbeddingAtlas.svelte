<script>
  import { onDestroy, onMount } from "svelte";

  import { EmbeddingAtlas as Component } from "../dist/index.js";

  /** @type import("../dist/index.js").EmbeddingAtlasProps */
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

<div bind:this={container} style:display="flex" style:width="100%" style:height="100%"></div>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script>
  import { onMount } from "svelte";

  import { EmbeddingAtlas as Component } from "../dist/index.js";

  /** @type import("../dist/index.js").EmbeddingAtlasProps */
  let props = $props();

  /** @type {HTMLElement} */
  let container;

  let viewerProps = $derived({ ...props, overlay: props.overlay, focusRequest: props.focusRequest });

  onMount(() => {
    let component = new Component(container, props);

    $effect(() => {
      component.update(viewerProps);
    });

    return () => {
      component.destroy();
    };
  });
</script>

<div bind:this={container} style:display="flex" style:width="100%" style:height="100%"></div>

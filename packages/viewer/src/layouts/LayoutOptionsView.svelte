<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { getStoreContext } from "../stores/embedding_atlas_store.js";
  import { findLayoutOptionsComponent } from "./layout_types.js";

  const store = getStoreContext();
  const { layouts, currentLayout } = store;

  let spec = $derived($layouts[$currentLayout] as any);
  let type = $derived(spec?.type);
  let LayoutClass = $derived(type != undefined ? findLayoutOptionsComponent(type) : undefined);
</script>

{#key $currentLayout}
  <LayoutClass layout={$currentLayout} />
{/key}

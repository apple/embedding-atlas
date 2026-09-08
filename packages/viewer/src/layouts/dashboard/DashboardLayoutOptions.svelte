<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "../../widgets/Button.svelte";

  import { IconAddChart } from "../../assets/icons.js";

  import { getStoreContext } from "../../stores/embedding_atlas_store.js";
  import type { LayoutOptionsProps } from "../layout.js";

  let { layout }: LayoutOptionsProps = $props();

  const store = getStoreContext();

  function newChart() {
    store.addChartToLayout(layout);
  }
</script>

<Button icon={IconAddChart} onClick={newChart} title="Add widget to dashboard" />

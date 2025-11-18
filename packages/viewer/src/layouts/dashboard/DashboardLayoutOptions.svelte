<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "../../widgets/Button.svelte";

  import { IconPlus } from "../../assets/icons.js";

  import { findUnusedId } from "../../utils/identifier.js";
  import type { LayoutOptionsProps } from "../layout.js";
  import { type DashboardLayoutState } from "./types.js";

  let { charts, onChartsChange }: LayoutOptionsProps<DashboardLayoutState> = $props();

  function newChart() {
    let id = findUnusedId(charts);
    onChartsChange({ [id]: { type: "builder", title: "New" } });
  }
</script>

<Button icon={IconPlus} onClick={newChart} title="Add" />

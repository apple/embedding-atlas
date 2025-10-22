<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { CustomCellProps } from "../lib/api/custom-cells.js";

  interface Props extends CustomCellProps {}

  let { value, rowData }: Props = $props();
</script>

<div>{JSON.stringify(rowData)}</div>

<style>
</style>

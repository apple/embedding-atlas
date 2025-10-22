<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { ChartViewProps } from "../chart.js";

  let props: ChartViewProps = $props();
</script>

<div class="bg-slate-100 p-2 rounded-md w-full h-full overflow-y-scroll">
  <pre>{JSON.stringify(props.spec, null, 2)}</pre>
</div>

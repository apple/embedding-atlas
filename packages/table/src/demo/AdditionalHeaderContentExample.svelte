<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { AdditionalHeaderContentProps } from "../lib/api/custom-headers";

  interface Props extends AdditionalHeaderContentProps {}

  let { column }: Props = $props();
</script>

<div class="additional-header-content">Additional Header Content: {column}</div>

<style>
  .additional-header-content {
    height: 100px;
  }
</style>

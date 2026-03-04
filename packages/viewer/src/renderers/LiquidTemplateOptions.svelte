<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import CodeEditor from "../widgets/CodeEditor.svelte";
  import type { RendererOptionsProps } from "./types.js";

  let { options, onChange }: RendererOptionsProps = $props();
</script>

<CodeEditor
  class="w-96 h-64 bg-white dark:bg-black rounded-md"
  value={options?.template ?? ""}
  onChange={(value) => {
    onChange?.({ template: value });
  }}
/>

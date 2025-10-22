<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    value?: string;
    placeholder?: string | null;
    className?: string;
  }

  let { value = $bindable(), placeholder = "", className = "" }: Props = $props();
</script>

<textarea
  class="form-textarea block w-full h-96 font-mono rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 {className ??
    ''}"
  bind:value={value}
  placeholder={placeholder}
></textarea>

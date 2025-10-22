<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    type?: string;
    value?: string;
    placeholder?: string;
    className?: string;
  }

  let { value = $bindable(), type = "text", placeholder = "", className = "" }: Props = $props();

  function onKeyDown(e: KeyboardEvent) {
    if (e.key == "Escape") {
      e.stopPropagation();
    }
  }
</script>

<input
  type={type}
  bind:value={value}
  placeholder={placeholder}
  onkeydown={onKeyDown}
  class="form-input rounded-md py-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 {className ??
    ''}"
/>

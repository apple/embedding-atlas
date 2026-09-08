<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Input from "./Input.svelte";

  interface Props {
    value: number;
    min?: number;
    max?: number;
    step?: number;
    placeholder?: string;
    className?: string;
  }

  let { value = $bindable(), min, max, step, placeholder = "", className = "" }: Props = $props();

  let text = $state(String(value));

  $effect(() => {
    text = String(value);
  });

  function commit() {
    let n = parseFloat(text);
    if (isNaN(n)) {
      text = String(value);
      return;
    }
    if (min != null) n = Math.max(min, n);
    if (max != null) n = Math.min(max, n);
    value = n;
    text = String(n);
  }
</script>

<Input type="text" bind:value={text} placeholder={placeholder} className={className} onEnter={commit} onBlur={commit} />

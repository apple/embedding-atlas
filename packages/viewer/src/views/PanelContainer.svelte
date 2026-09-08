<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";
  import Resizer from "../widgets/Resizer.svelte";

  let panelWidth = $state(400);

  interface Props {
    title: string;
    class?: string;
    children?: Snippet;
  }

  let { title, children, class: className }: Props = $props();
</script>

<div class="h-full relative {className ?? ''}" style:width="{panelWidth}px">
  <h2 class="text-slate-400 dark:text-slate-500 text-sm uppercase select-none">{title}</h2>

  {@render children?.()}

  <Resizer
    class="absolute right-0 top-0 bottom-0 w-2"
    axis="x"
    value={panelWidth}
    onChange={(v) => (panelWidth = v)}
    min={300}
    max={800}
    scaler={1}
  />
</div>

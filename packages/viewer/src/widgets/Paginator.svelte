<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";
  import PaginatorControls from "./PaginatorControls.svelte";

  interface Props {
    count: number;
    pageSize?: number;

    children: Snippet<[{ start: number; end: number }]>;
  }

  let { count, pageSize = 50, children }: Props = $props();

  let currentPage = $state.raw(0);

  let pageCount = $derived(Math.ceil(count / pageSize));
</script>

{#if pageCount > 1}
  <PaginatorControls
    currentPage={currentPage}
    pageCount={pageCount}
    onChange={(p) => {
      currentPage = p;
    }}
  />
{/if}

{@render children({ start: currentPage * pageSize, end: Math.min(count, currentPage * pageSize + pageSize) })}

{#if pageCount > 1}
  <PaginatorControls
    currentPage={currentPage}
    pageCount={pageCount}
    onChange={(p) => {
      currentPage = p;
    }}
  />
{/if}

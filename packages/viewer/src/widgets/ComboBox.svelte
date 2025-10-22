<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    value?: string | null;
    placeholder?: string | null;
    options?: string[] | null;
    onChange?: (value: string) => void;
    className?: string | null;
  }

  let { value = "", placeholder, options = [], onChange, className }: Props = $props();

  let inputElement: HTMLInputElement;
  let container: HTMLDivElement;

  let isOpen = $state(false);
  let filteredItems = $derived(
    value != null && options != null
      ? options.filter((item) => item.toLowerCase().includes(value.toLowerCase()))
      : (options ?? []),
  );

  function emit(value: string) {
    onChange?.(value);
  }

  function selectItem(item: string) {
    emit(item);
    isOpen = false;
  }
</script>

<div bind:this={container} class="relative {className ?? ''}">
  <input
    bind:this={inputElement}
    type="text"
    class="form-input rounded-md py-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 w-full"
    placeholder={placeholder}
    value={value}
    onfocus={() => (isOpen = true)}
    onblur={(e) => {
      if (e.relatedTarget && e.relatedTarget instanceof Node && container.contains(e.relatedTarget)) {
        return;
      }
      isOpen = false;
    }}
    onchange={() => emit(inputElement.value ?? "")}
    oninput={() => emit(inputElement.value ?? "")}
  />

  {#if isOpen && filteredItems.length > 0}
    <ul
      class="absolute mt-1 w-full rounded-md shadow-md p-1 z-10 flex flex-col bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
    >
      {#each filteredItems as item (item)}
        <button
          class="px-2 py-1 text-left rounded-md text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700"
          onclick={() => selectItem(item)}
        >
          {item}
        </button>
      {/each}
    </ul>
  {/if}
</div>

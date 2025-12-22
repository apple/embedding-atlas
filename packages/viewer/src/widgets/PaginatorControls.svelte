<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconLeft, IconRight } from "../assets/icons";

  interface Props {
    currentPage: number;
    pageCount: number;
    onChange: (page: number) => void;
  }

  let { currentPage, pageCount, onChange }: Props = $props();

  let inputElement: HTMLInputElement;

  function update(p: number) {
    if (!isFinite(p)) {
      return currentPage;
    }
    p = Math.max(0, Math.min(pageCount - 1, p));

    onChange(p);
    inputElement.value = (p + 1).toString();
  }
</script>

<div class="flex gap-2 select-none mx-2 my-1 items-center">
  <button
    class="border border-slate-300 bg-white/50 dark:border-slate-700 dark:bg-black/50 px-3 py-1 rounded-md"
    class:text-slate-300={currentPage <= 0}
    class:dark:text-slate-700={currentPage <= 0}
    disabled={currentPage <= 0}
    onclick={() => {
      update(currentPage - 1);
    }}
  >
    <IconLeft class="w-5 h-5" />
  </button>
  <div class="flex items-center gap-2">
    <div>Page</div>
    <input
      bind:this={inputElement}
      value={currentPage + 1}
      class="form-input w-16 h-7.5 text-right border border-slate-300 bg-white/50 dark:border-slate-700 dark:bg-black/50 rounded-md px-2 py-0 my-1"
      onblur={() => {
        let p = +inputElement.value;
        update(Math.round(p) - 1);
      }}
      onkeydown={(e) => {
        if (e.key == "Enter") {
          let p = +inputElement.value;
          update(Math.round(p) - 1);
        }
      }}
    />
    <div>of {pageCount}</div>
  </div>
  <button
    class="border border-slate-300 bg-white/50 dark:border-slate-700 dark:bg-black/50 px-3 py-1 rounded-md"
    class:text-slate-300={currentPage >= pageCount - 1}
    class:dark:text-slate-700={currentPage >= pageCount - 1}
    disabled={currentPage >= pageCount - 1}
    onclick={() => {
      update(currentPage + 1);
    }}
  >
    <IconRight class="w-5 h-5" />
  </button>
</div>

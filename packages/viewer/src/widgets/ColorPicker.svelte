<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { defaultCategoryColors } from "@embedding-atlas/component";

  import PopupButton from "./PopupButton.svelte";

  interface Props {
    value: string;
    /** Preset colors shown as swatches. Defaults to the default category colors. */
    presets?: string[];
    title?: string;
    onChange?: (value: string) => void;
  }

  let { value, presets = defaultCategoryColors(10), title = "Choose color", onChange }: Props = $props();

  let visible = $state(false);
  let hexInput = $state("");

  $effect.pre(() => {
    hexInput = value;
  });

  function isValidColor(color: string): boolean {
    return /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(color);
  }

  function select(color: string) {
    if (color.toLowerCase() != value.toLowerCase()) {
      onChange?.(color);
    }
  }

  function commitHexInput() {
    let color = hexInput.trim();
    if (!color.startsWith("#")) {
      color = "#" + color;
    }
    if (isValidColor(color)) {
      select(color);
    } else {
      hexInput = value;
    }
  }
</script>

<PopupButton bind:visible={visible} placement="bottom-end" class="p-2">
  {#snippet button({ toggle })}
    <button
      class="flex-none rounded-md p-1 h-[28px] w-[28px] bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 focus-visible:outline-2 outline-blue-600 -outline-offset-1"
      title={title}
      onclick={toggle}
    >
      <div class="w-full h-full rounded-sm" style:background={value}></div>
    </button>
  {/snippet}
  <div class="flex flex-col gap-2">
    <div class="grid grid-cols-5 gap-1">
      {#each presets as color}
        {@const isSelected = color.toLowerCase() == value.toLowerCase()}
        <button
          class="w-6 h-6 rounded-md border-2 focus-visible:outline-2 outline-blue-600 outline-offset-1 {isSelected
            ? 'border-slate-700 dark:border-slate-200'
            : 'border-transparent'}"
          style:background={color}
          title={color}
          onclick={() => {
            select(color);
            visible = false;
          }}
        ></button>
      {/each}
    </div>
    <div class="flex items-center gap-1">
      <label
        class="relative flex-none w-[28px] h-[28px] rounded-md cursor-pointer border border-slate-300 dark:border-slate-600 bg-[conic-gradient(red,yellow,lime,aqua,blue,magenta,red)]"
        title="Custom color"
      >
        <input
          type="color"
          class="absolute inset-0 opacity-0 cursor-pointer"
          value={isValidColor(value) && value.length == 7 ? value : "#000000"}
          onchange={(e) => select(e.currentTarget.value)}
        />
      </label>
      <input
        type="text"
        class="form-input w-[100px] rounded-md py-1 px-2 text-sm font-mono bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 dark:text-slate-300"
        bind:value={hexInput}
        onkeydown={(e) => {
          if (e.key == "Enter") {
            commitHexInput();
          } else if (e.key == "Escape") {
            hexInput = value;
          }
        }}
        onblur={commitHexInput}
      />
    </div>
  </div>
</PopupButton>

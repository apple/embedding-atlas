<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import PopupButton from "../widgets/PopupButton.svelte";
  import Select from "../widgets/Select.svelte";

  import {
    IconArrayType,
    IconDateType,
    IconNumberType,
    IconOtherType,
    IconSettings,
    IconStringType,
  } from "../assets/icons.js";

  import { rendererOptions, renderersList } from "../renderers/renderer_types.js";
  import { type ColumnStyle } from "../renderers/types.js";
  import type { ColumnDesc, JSType } from "../utils/database.js";

  interface Props {
    column: ColumnDesc;
    style: ColumnStyle;
    onChange: (value: ColumnStyle) => void;
  }

  let { column, style, onChange }: Props = $props();

  const jsTypeIcons: Record<JSType, any> = {
    string: IconStringType,
    number: IconNumberType,
    Date: IconDateType,
    "string[]": IconArrayType,
  };

  let TypeIcon = $derived(column.jsType != null ? (jsTypeIcons[column.jsType] ?? IconOtherType) : IconOtherType);

  function change(fields: Partial<ColumnStyle>) {
    onChange({ ...style, ...fields });
  }

  let optionsAction = $derived(style.renderer != null ? rendererOptions[style.renderer] : undefined);
</script>

<div
  class="bg-white dark:bg-slate-800 p-2 rounded-md border border-slate-200 dark:border-slate-700 flex flex-col gap-2"
>
  <div class="flex gap-2 items-center">
    <div
      class="flex-1 font-medium text-slate-700 dark:text-slate-300 truncate flex items-center gap-2"
      title={column.name}
    >
      <TypeIcon class="w-5 h-5 text-slate-400 dark:text-slate-500 flex-none" />
      {column.name}
    </div>
    <div class="text-sm text-slate-400 dark:text-slate-500 truncate max-w-40" title={column.type}>{column.type}</div>
  </div>
  <div class="flex flex-wrap items-center gap-x-4 gap-y-2 select-none">
    <div class="flex items-center gap-2">
      <Select
        label="Style"
        value={style.display ?? "badge"}
        onChange={(v) => {
          change({ display: v });
        }}
        options={[
          { value: "full", label: "Full" },
          { value: "badge", label: "Badge" },
          { value: "hidden", label: "Hidden" },
        ]}
      />
    </div>
    <div class="flex items-center gap-2">
      <Select
        label="Format"
        value={style.renderer ?? null}
        class="max-w-32"
        onChange={(v) => change({ renderer: v })}
        options={[
          { value: null, label: "(default)" },
          ...renderersList.map((x) => ({ value: x.renderer, label: x.label })),
        ]}
      />

      {#if optionsAction}
        {#key optionsAction}
          <PopupButton icon={IconSettings}>
            <div
              use:optionsAction={{
                options: style.options,
                onChange: (value) => {
                  change({ options: value });
                },
              }}
            ></div>
          </PopupButton>
        {/key}
      {/if}
    </div>
  </div>
</div>

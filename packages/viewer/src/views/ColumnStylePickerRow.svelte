<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import PopupButton from "../widgets/PopupButton.svelte";
  import Select from "../widgets/Select.svelte";

  import { IconSettings } from "../assets/icons";

  import { rendererOptions, renderersList } from "../renderers/renderer_types.js";
  import { type ColumnStyle } from "../renderers/types.js";
  import type { ColumnDesc } from "../utils/database.js";

  interface Props {
    column: ColumnDesc;
    style: ColumnStyle;
    onChange: (value: ColumnStyle) => void;
  }

  let { column, style, onChange }: Props = $props();

  function change(fields: Partial<ColumnStyle>) {
    onChange({ ...style, ...fields });
  }

  let optionsAction = $derived(style.renderer != null ? rendererOptions[style.renderer] : undefined);
</script>

<tr class="leading-10">
  <td class="w-full">
    <div class="max-w-80 whitespace-nowrap text-ellipsis overflow-x-hidden">
      {column.name}
    </div>
  </td>
  <td class="pr-2">
    <div class="flex items-center gap-1">
      <Select
        value={style.renderer ?? null}
        onChange={(v) => change({ renderer: v })}
        options={[
          { value: null, label: "(default)" },
          ...renderersList.map((x) => ({ value: x.renderer, label: x.label })),
        ]}
      />

      {#if optionsAction}
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
      {/if}
    </div>
  </td>
  <td>
    <div class="flex items-center gap-2">
      <Select
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
  </td>
</tr>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Select from "./widgets/Select.svelte";
  import Slider from "./widgets/Slider.svelte";

  import type { ColumnDesc } from "./database_utils.js";
  import { renderersList, type ColumnStyle } from "./renderers/index.js";

  interface Props {
    column: ColumnDesc;
    style: ColumnStyle;
    onChange: (value: ColumnStyle) => void;
  }

  let { column, style, onChange }: Props = $props();

  function change(fields: Partial<ColumnStyle>) {
    onChange({ ...style, ...fields });
  }
</script>

<tr class="leading-10">
  <td class="w-full">
    <div class="max-w-80 whitespace-nowrap text-ellipsis overflow-x-hidden">
      {column.name}
    </div>
  </td>
  <td class="pr-2">
    <div class="flex items-center gap-2">
      <Select
        value={style.renderer ?? null}
        onChange={(v) => change({ renderer: v })}
        options={[
          { value: null, label: "(default)" },
          ...renderersList.map((x) => ({ value: x.renderer, label: x.label })),
        ]}
      />

      {#if style.renderer == "image"}
        <Slider
          bind:value={
            () => style.rendererOptions?.size ?? 100,
            (v) => {
              change({ rendererOptions: { size: v } });
            }
          }
          width={72}
          min={16}
          max={400}
        />
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

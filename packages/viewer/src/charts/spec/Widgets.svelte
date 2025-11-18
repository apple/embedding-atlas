<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import InlineSelect from "../../widgets/InlineSelect.svelte";
  import ScaleTypePicker from "../controls/ScaleTypePicker.svelte";

  import type { ChartViewProps } from "../chart.js";
  import type { ChartOutputs } from "./runtime.js";
  import type { ChartSpec } from "./spec.js";

  interface Props {
    spec: ChartSpec;
    outputs: ChartOutputs;
    onSpecChange: ChartViewProps<ChartSpec>["onSpecChange"];
  }

  let { spec, outputs, onSpecChange }: Props = $props();
</script>

<div class="flex flex-wrap gap-2 justify-end flex-wrap">
  {#each spec.widgets ?? [] as widget}
    {#if widget.type == "scale.type"}
      <ScaleTypePicker
        label={widget.channel.toUpperCase()}
        bind:value={
          () => outputs.scale[widget.channel].type ?? "linear",
          (v) => {
            onSpecChange({ scale: { [widget.channel]: { type: v } } });
          }
        }
      />
    {:else if widget.type == "encoding.normalize"}
      {@const layers = typeof widget.layer == "number" ? [widget.layer] : widget.layer}
      <span class="flex gap-1 select-none">
        <span class="text-slate-400 dark:text-slate-500 text-sm">Normalize:</span>
        <InlineSelect
          options={[
            { value: null, label: "off" },
            ...(widget.options ?? ["x", "y"]).map((x) => ({ value: x, label: x.toUpperCase() })),
          ]}
          value={(spec.layers?.[layers[0]]?.encoding?.[widget.attribute] as any)?.normalize ?? null}
          onChange={(v) => {
            let newSpec: ChartSpec = JSON.parse(JSON.stringify(spec));
            for (let layer of layers) {
              if (newSpec.layers?.[layer]?.encoding?.[widget.attribute] != undefined) {
                let encoding: any = newSpec.layers?.[layer]?.encoding?.[widget.attribute];
                encoding.normalize = v ?? undefined;
              }
            }
            onSpecChange(newSpec, "replace");
          }}
        />
      </span>
    {/if}
  {/each}
</div>

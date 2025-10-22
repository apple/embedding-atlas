<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { MosaicClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import type { ChartViewProps } from "../chart.js";

  interface Item {
    label: string;
    color: string;
    count: number;
    predicate: any;
  }

  interface Props {
    items: Item[];
  }

  interface State {
    selection?: string[];
  }

  let {
    context,
    spec,
    state: chartState,
    onStateChange,
  }: Omit<ChartViewProps<Props, State>, "width" | "height"> = $props();

  let selectedItems = $derived.by(() => {
    let set = new Set(chartState.selection ?? []);
    return new Set(spec.items.filter((x) => set.has(x.label)));
  });

  class Client extends MosaicClient {
    reset() {
      onStateChange({ selection: undefined });
    }
  }

  const client = new Client();

  function onClickItem(item: Item, event: MouseEvent) {
    let newSelection = new Set(chartState.selection ?? []);
    if (event.shiftKey || event.metaKey) {
      if (newSelection.has(item.label)) {
        newSelection.delete(item.label);
      } else {
        newSelection.add(item.label);
      }
      onStateChange({ selection: Array.from(newSelection) });
    } else {
      if (newSelection.has(item.label) && newSelection.size == 1) {
        onStateChange({ selection: undefined });
      } else {
        onStateChange({ selection: [item.label] });
      }
    }
  }

  $effect.pre(() => {
    $effect.pre(() => {
      let set = new Set(chartState.selection ?? []);
      let items = spec.items.filter((x) => set.has(x.label));

      let predicate = items.length != 0 ? SQL.or(items.map((x) => x.predicate)) : null;
      let clause = {
        source: client,
        clients: new Set([client]),
        value: items.length == 0 ? null : items,
        predicate: predicate,
      };
      context.filter.update(clause);
    });

    return () => {
      context.filter.update({ source: client, clients: new Set([client]), value: null, predicate: null });
    };
  });
</script>

<table>
  <tbody>
    {#each spec.items as item}
      {@const selected = selectedItems.has(item) || selectedItems.size == 0}
      <tr
        class="hover:bg-slate-200 dark:hover:bg-slate-700 select-none leading-7"
        class:opacity-20={!selected}
        onclick={(e) => {
          onClickItem(item, e);
        }}
      >
        <td class="first:rounded-tl-md first:rounded-bl-md">
          <div class="block w-4 h-4 mx-2 rounded-full" style:background-color={item.color}></div>
        </td>
        <td>
          <div class="whitespace-nowrap nowrap max-w-72 text-ellipsis overflow-hidden" title={item.label}>
            {item.label}
          </div>
        </td>
        <td class="text-slate-400 px-2 text-xs text-right last:rounded-tr-md last:rounded-br-md" title="Count"
          >{item.count.toLocaleString()}</td
        >
      </tr>
    {/each}
  </tbody>
</table>

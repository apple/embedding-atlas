<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { MosaicClient, type Selection } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import type { PlotStateStore } from "./plots/plot_state_store.js";

  interface Item {
    label: string;
    color: string;
    count: number;
    predicate: any;
  }

  interface Props {
    items: Item[];
    selection?: Selection | null;
    stateStore?: PlotStateStore<{ selectedItems: string[] } | null> | null;
  }

  let { items, selection = null, stateStore }: Props = $props();

  let selectedItems = $state.raw(new Set<Item>());

  class Client extends MosaicClient {
    reset() {
      selectedItems = new Set();
    }
  }

  const client = new Client();

  function onClickItem(item: Item, event: MouseEvent) {
    if (event.shiftKey || event.metaKey) {
      let newSelection = new Set(selectedItems);
      if (newSelection.has(item)) {
        newSelection.delete(item);
      } else {
        newSelection.add(item);
      }
      selectedItems = newSelection;
    } else {
      if (selectedItems.has(item) && selectedItems.size == 1) {
        selectedItems = new Set();
      } else {
        selectedItems = new Set([item]);
      }
    }
  }

  $effect.pre(() => {
    let captured = selection;
    if (captured == null) {
      return;
    }

    $effect.pre(() => {
      let items = selectedItems;

      let predicate = items.size != 0 ? SQL.or(Array.from(items).map((x) => x.predicate)) : null;
      let clause = {
        source: client,
        clients: new Set([client]),
        value: items.size == 0 ? null : items,
        predicate: predicate,
      };
      captured.update(clause);
    });

    return () => {
      captured.update({ source: client, clients: new Set([client]), value: null, predicate: null });
    };
  });

  // Sync with state store
  $effect.pre(() => {
    let store = stateStore;
    if (!store) {
      return;
    }
    let cleanup = store.subscribe((s) => {
      if (s == null) {
        return;
      }
      selectedItems = new Set(items.filter((item) => s.selectedItems.indexOf(item.label) >= 0));
    });
    $effect.pre(() => {
      store.set({
        selectedItems: Array.from(selectedItems).map((item) => item.label),
      });
    });
    return cleanup;
  });
</script>

<div class="absolute right-0 top-0 p-2 m-2 rounded-md bg-slate-100/75 dark:bg-slate-800/75 backdrop-blur-sm">
  <table>
    <tbody>
      {#each items as item}
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
</div>

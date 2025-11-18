<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<!-- A "plot" that shows user-specified selections -->
<script lang="ts">
  import { makeClient, MosaicClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import Button from "../../widgets/Button.svelte";
  import CodeEditor from "../../widgets/CodeEditor.svelte";
  import Input from "../../widgets/Input.svelte";
  import Container from "../common/Container.svelte";

  import { IconClose, IconEdit } from "../../assets/icons.js";
  import { predicateToString } from "../../utils/database.js";
  import type { ChartViewProps } from "../chart.js";
  import type { PredicatesSpec } from "./types.js";

  interface State {
    selectedPredicates?: string[];
  }

  interface Item {
    name: string;
    predicate: string;
  }

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onSpecChange,
    onStateChange,
  }: ChartViewProps<PredicatesSpec, State> = $props();

  let { colorScheme } = context;

  let items = $derived(spec.items ?? []);
  let selectedPredicates = $derived(chartState.selectedPredicates ?? []);

  let editingEnabled = $state(false);
  let editingItem = $state.raw<Item | null>(null);
  let editingPredicate = $state("");
  let editingName = $state("");

  function resetEditor() {
    editingEnabled = false;
    editingItem = null;
    editingName = "";
    editingPredicate = "";
  }

  function setItems(newItems: Item[]) {
    onSpecChange({
      items: newItems,
    });
    onStateChange({
      selectedPredicates: (chartState.selectedPredicates ?? []).filter(
        (x) => newItems.find((item) => item.predicate == x) != null,
      ),
    });
  }

  async function validate(): Promise<Item | null> {
    let name = editingName.trim();
    if (name == "") {
      name = "Predicate";
    }
    let predicate = editingPredicate.trim();
    if (predicate == "") {
      return null;
    }
    // Test the perdicate.
    try {
      await context.coordinator.query(SQL.Query.from(context.table).select({ count: SQL.count() }).where(predicate));
    } catch (e: any) {
      alert(e.toString());
      return null;
    }
    return { name, predicate };
  }

  // Mosaic client
  $effect.pre(() => {
    let client = makeClient({
      coordinator: context.coordinator,
      selection: context.filter,
      query: (): any => SQL.sql`SELECT 1`,
    });

    let source = {
      reset: () => {
        onStateChange({ selectedPredicates: [] });
      },
    };

    // Sync selection predicate.
    $effect(() => {
      if (selectedPredicates.length == 0) {
        context.filter.update({
          source: source,
          clients: new Set<MosaicClient>([client]),
          predicate: null,
          value: null,
        });
      } else {
        let preds = selectedPredicates.map((x) => "(" + x + ")").join(" OR ");
        context.filter.update({
          source: source,
          clients: new Set<MosaicClient>([client]),
          predicate: SQL.asVerbatim(preds),
          value: preds,
        });
      }
    });

    return () => {
      client.destroy();
    };
  });
</script>

<Container width={width} height={height} scrollY={true}>
  <div class="flex flex-col gap-1">
    {#each items as item}
      {@const isSelected = selectedPredicates.indexOf(item.predicate) >= 0}
      <div
        class="flex gap-4 w-full bg-white dark:bg-slate-800 p-2 rounded-md border border-slate-200 dark:border-slate-600 select-none"
        class:!bg-blue-100={isSelected}
        class:!border-blue-400={isSelected}
        class:dark:!bg-blue-800={isSelected}
        class:dark:!border-blue-600={isSelected}
      >
        <button
          class="flex-1 overflow-hidden text-left"
          onclick={(e) => {
            if (e.shiftKey) {
              selectedPredicates = isSelected
                ? selectedPredicates.filter((x) => x != item.predicate)
                : [...selectedPredicates, item.predicate];
            } else {
              if (isSelected) {
                selectedPredicates = [];
              } else {
                selectedPredicates = [item.predicate];
              }
            }
          }}
        >
          <div class="text-ellipsis overflow-hidden w-full">{item.name}</div>
          <div class="text-ellipsis overflow-hidden w-full">
            <code class="text-xs whitespace-nowrap" title={item.predicate}>{item.predicate}</code>
          </div>
        </button>
        <div class="flex-none flex gap-1">
          <Button
            icon={IconEdit}
            onClick={() => {
              editingItem = item;
              editingPredicate = item.predicate;
              editingName = item.name;
              editingEnabled = true;
            }}
          />
          <Button
            icon={IconClose}
            onClick={() => {
              setItems(items.filter((x) => x !== item));
            }}
          />
        </div>
      </div>
    {/each}
    <button
      class="text-left text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 whitespace-nowrap text-ellipsis w-full overflow-hidden select-none"
      onclick={() => {
        editingEnabled = true;
        editingItem = null;
      }}>+ Add Predicate</button
    >
  </div>

  {#if editingEnabled}
    <div class="mt-4">
      <Input bind:value={editingName} placeholder="predicate name" className="w-full mb-2" />
      <div class="text-slate-500 dark:text-slate-400 text-sm mb-1">SQL Predicate</div>
      <div class="w-full !h-32 mb-2">
        <CodeEditor
          language="sql"
          sql={{ table: context.table, columns: context.columns }}
          class="w-full h-full"
          colorScheme={$colorScheme}
          value={editingPredicate}
          onChange={(v) => (editingPredicate = v)}
        />
      </div>
      <div class="flex gap-2">
        {#if editingItem != null}
          <Button
            label="Update"
            onClick={async () => {
              let newItem = await validate();
              if (newItem) {
                setItems(items.map((x) => (x === editingItem ? newItem : x)));
                resetEditor();
              }
            }}
          />
        {:else}
          <Button
            label="Add"
            onClick={async () => {
              let newItem = await validate();
              if (newItem) {
                setItems([...items, newItem]);
                resetEditor();
              }
            }}
          />
        {/if}

        <Button
          label="Cancel"
          onClick={() => {
            editingItem = null;
            editingEnabled = false;
          }}
        />
        <div class="flex-1"></div>
        <button
          class="text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
          onclick={() => (editingPredicate = predicateToString(context.filter.predicate(null)) ?? "")}
        >
          Current Predicate
        </button>
      </div>
    </div>
  {/if}
</Container>

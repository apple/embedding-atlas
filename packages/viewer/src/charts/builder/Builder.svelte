<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { applyUpdatesIfNeeded } from "@embedding-atlas/utils";
  import { untrack } from "svelte";

  import CodeEditor from "../../widgets/CodeEditor.svelte";
  import Select from "../../widgets/Select.svelte";
  import ChartView from "../ChartView.svelte";
  import Container from "../common/Container.svelte";
  import ChartIcon from "./ChartIcon.svelte";

  import { type ColumnDesc, type JSType } from "../../utils/database.js";
  import type { ChartBuilderDescription, ChartViewProps } from "../chart.js";
  import { chartBuilderDescriptions } from "../chart_types.js";
  import type { UIElement } from "./builder_description.js";

  let { context, width, height, onSpecChange, onStateChange }: ChartViewProps<{}, {}> = $props();

  let { columns, colorScheme } = context;

  let chartBuilders = chartBuilderDescriptions();

  let builder: ChartBuilderDescription<any, UIElement[]> = $state.raw(chartBuilders[0]);

  let values: Record<string, any> = $state({});
  let validateResult: string | boolean = $state(false);
  let localChartState = $state.raw<any>(null);
  let localChartSpec = $state.raw<any>(null);

  // If builder change, refresh the values.
  $effect.pre(() => {
    let _ = builder;
    // Update values so we keep the fields selected if the keys are the same.
    let currentValues = untrack(() => values);
    let newValues: Record<string, any> = {};
    for (let item of builder.ui) {
      if ("field" in item) {
        let allowedColumns = filteredColumns(columns, item.field.types);
        let current = currentValues[item.field.key];
        if (current != null && allowedColumns.findIndex(({ name }) => name == current) >= 0) {
          newValues[item.field.key] = current;
        }
      }
    }
    // Setting values should trigger validation and update the spec.
    values = newValues;
  });

  $effect.pre(() => {
    try {
      let input = getInput();
      if (input == undefined) {
        validateResult = false;
        localChartSpec = null;
        localChartState = null;
        return;
      }
      let r = builder.create(input, { table: context.table, id: context.id });
      validateResult = r != null;
      localChartSpec = r;
      localChartState = null;
    } catch (e: any) {
      validateResult = e.toString();
    }
  });

  function confirm() {
    if (localChartSpec != null) {
      onSpecChange(localChartSpec, "replace");
      onStateChange(localChartState ?? {}, "replace");
    }
  }

  function getInput(): Record<string, any> | undefined {
    let input = { ...values };
    for (let item of builder.ui) {
      if ("field" in item) {
        let value = getField(input[item.field.key]);
        if (item.field.required && value == undefined) {
          return undefined;
        }
        input[item.field.key] = value;
      }
    }
    return input;
  }

  function getField(name: string): { name: string; type: "continuous" | "discrete" | "discrete[]" } | null {
    let c = columns.find((x) => x.name == name);
    if (c == null || c.jsType == null) {
      return null;
    }
    switch (c.jsType) {
      case "number":
        return {
          name: c.name,
          type: "continuous",
        };
      case "string":
        return {
          name: c.name,
          type: "discrete",
        };
      case "string[]":
        return {
          name: c.name,
          type: "discrete[]",
        };
      default:
        return null;
    }
  }

  function filteredColumns(columns: ColumnDesc[], types: JSType[] | null | undefined): ColumnDesc[] {
    if (types == null) {
      return columns.filter((c) => c.jsType != null);
    }
    return columns.filter((c) => c.jsType != null && types.indexOf(c.jsType) >= 0);
  }
</script>

<Container width={width} height={height} scrollY={true} class="flex flex-col gap-2">
  <div class="flex flex-wrap gap-2">
    {#each chartBuilders as type}
      {@const selected = builder == type}
      <button
        onclick={() => {
          builder = type;
        }}
        title={type.description}
        class="rounded-md border border-slate-200 dark:border-slate-700"
        class:!border-slate-600={selected}
        class:dark:!border-slate-400={selected}
        class:!bg-slate-100={selected}
        class:dark:!bg-slate-700={selected}
      >
        <ChartIcon type={type.icon} colorScheme={$colorScheme} />
      </button>
    {/each}
  </div>

  <div>{builder.description}</div>

  {#each builder.ui as elem}
    {#if "field" in elem}
      {@const key = elem.field.key}
      <span class="text-slate-500 dark:text-slate-400">{elem.field.label}</span>
      <Select
        value={values[key] ?? null}
        onChange={(v) => (values[key] = v)}
        placeholder="(select field)"
        class="w-full"
        options={filteredColumns(columns, elem.field.types).map((c) => ({
          value: c.name,
          label: `${c.name} (${c.type})`,
        }))}
      />
    {/if}
    {#if "code" in elem}
      {@const key = elem.code.key}
      <div class="w-full h-64">
        <CodeEditor
          class="w-full h-full"
          value={values[key]}
          onChange={(v) => (values[key] = v)}
          colorScheme={$colorScheme}
          language={elem.code.language ?? "plain"}
        />
      </div>
    {/if}
  {/each}
  {#if localChartSpec != null && builder.preview !== false}
    {#key localChartSpec}
      <div class="border-t border-b border-dotted border-slate-400 dark:border-slate-500 py-2 my-1">
        <div class="-m-2 rounded-md">
          <ChartView
            context={context}
            spec={localChartSpec}
            state={localChartState ?? {}}
            width={"container"}
            mode="view"
            onStateChange={(update, mode = "merge") => {
              applyUpdatesIfNeeded(localChartState ?? {}, update, mode, (r) => (localChartState = r));
            }}
            onSpecChange={(update, mode = "merge") => {
              applyUpdatesIfNeeded(localChartSpec ?? {}, update, mode, (r) => (localChartSpec = r));
            }}
          />
        </div>
      </div>
    {/key}
  {/if}
  <div>
    <button
      class="px-2 h-8 w-24 rounded-md text-white text-sm"
      class:bg-blue-500={validateResult === true}
      class:bg-gray-300={validateResult !== true}
      class:dark:text-gray-500={validateResult !== true}
      class:dark:bg-gray-700={validateResult !== true}
      disabled={validateResult !== true}
      onclick={confirm}
    >
      Confirm
    </button>
  </div>
  {#if typeof validateResult == "string" && validateResult.trim() != ""}
    <div>{validateResult}</div>
  {/if}
</Container>

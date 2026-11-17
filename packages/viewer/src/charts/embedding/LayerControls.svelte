<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "../../widgets/Button.svelte";
  import CheckBox from "../../widgets/CheckBox.svelte";
  import ColorPicker from "../../widgets/ColorPicker.svelte";
  import CornerButton from "../../widgets/CornerButton.svelte";
  import Select from "../../widgets/Select.svelte";

  import { IconClose, IconDensityLayer, IconLabelsLayer, IconPointsLayer } from "../../assets/icons.js";
  import type { ChartContext } from "../chart.js";
  import type { EmbeddingLegend } from "./color_scale.js";
  import {
    defaultLabelsLayer,
    defaultLayers,
    isFilteredByDefault,
    resolveLayerData,
    type DefaultLabelsOptions,
    type LayerData,
    type LayerSpec,
    type LayerValues,
  } from "./layer_data.js";
  import SliderRow from "./SliderRow.svelte";
  import type { EmbeddingField, EmbeddingSpec } from "./types.js";

  type LayerType = LayerSpec["type"];

  interface Props {
    context: ChartContext;
    spec: EmbeddingSpec;
    /** The table of the main data. */
    dataTable: string;
    /** The color used when a layer has no category or color. */
    defaultColor: string;
    /** The legend of the main category field, used for the default fixed color. */
    mainCategoryLegend: EmbeddingLegend | null;
    /** The view's layers (the spec's layers, or the default layers) with their resolved data. */
    resolvedLayers: { layer: LayerSpec; data: LayerData }[];
    /** The viewer's options for default labels layers (see `defaultLabelsLayer`). */
    labelsOptions: DefaultLabelsOptions;
    /** Effective values of the layers (including auto values), matched to layers by id (or index). */
    layerValues: LayerValues[];
    onSpecChange: (fn: (draft: EmbeddingSpec) => void) => void;
  }

  let {
    context,
    spec,
    dataTable,
    defaultColor,
    mainCategoryLegend,
    resolvedLayers,
    labelsOptions,
    layerValues,
    onSpecChange,
  }: Props = $props();

  /** The labels provided to the viewer (e.g., with the CLI's --labels option), referred to as "$labels". */
  let providedLabels = $derived(context.embeddingViewLabels ?? null);

  // Layers with `visible: false` stay in the spec but are not listed. For a spec without `layers`, these are the
  // default layers until the controls are touched. `index` is the layer's index (for edits and `layerValues`).
  let listedLayers = $derived(
    resolvedLayers
      .map(({ layer, data }, index) => ({ layer, data, index }))
      .filter(({ layer }) => layer.visible !== false),
  );

  /** Edit the layers, converting the spec to explicit layers first if needed. */
  function editLayers(fn: (layers: LayerSpec[], draft: EmbeddingSpec) => void) {
    onSpecChange((draft) => {
      if (draft.layers == null) {
        draft.layers = defaultLayers(draft, labelsOptions);
        // These are now part of the layers. (`minimumDensity` stays, it applies to all layers.)
        delete draft.mode;
        delete draft.pointSize;
      }
      fn(draft.layers, draft);
    });
  }

  const typeLabels: Record<LayerType, string> = { points: "Points", density: "Density", labels: "Labels" };
  const typeIcons: Record<LayerType, any> = {
    points: IconPointsLayer,
    density: IconDensityLayer,
    labels: IconLabelsLayer,
  };

  function updateLayer(index: number, fn: (layer: any) => void) {
    editLayers((layers) => {
      let layer = layers[index];
      if (layer != null) {
        fn(layer);
      }
    });
  }

  function addLayer(type: LayerType) {
    editLayers((layers) => {
      // A labels layer gets the default labels (the provided labels on the main table), or automatic labels.
      layers.push(type == "labels" ? defaultLabelsLayer({ ...labelsOptions, autoLabels: true })! : { type });
    });
  }

  /** Set a field of a layer's data (`undefined` removes it, and an empty `data`). */
  function setData(layer: any, key: string, value: any) {
    if (value === undefined) {
      if (layer.data != null) {
        delete layer.data[key];
        if (Object.keys(layer.data).length == 0) {
          delete layer.data;
        }
      }
    } else {
      layer.data ??= {};
      layer.data[key] = value;
    }
  }

  /** Set the category column of a layer. If the layer is on the main table and there is no main category yet,
   *  the column becomes the main category. Other layers that would then inherit it keep their fixed color. */
  function setLayerCategory(index: number, column: string, isMainTable: boolean) {
    editLayers((layers, draft) => {
      let layer: any = layers[index];
      if (layer == null) {
        return;
      }
      delete layer.color;
      // The layer's color scale is for its previous category.
      delete layer.colorScale;
      if (isMainTable && draft.data.category == null) {
        // Layers on the main table without a category or color would follow the new main category:
        // opt them out (`category: null`), so only this layer changes.
        layers.forEach((other, i) => {
          if (
            i != index &&
            other.type != "labels" &&
            other.color == null &&
            other.data?.category === undefined &&
            resolveLayerData(draft, other, dataTable).isDataTable
          ) {
            setData(other, "category", null);
          }
        });
        draft.data.category = column;
        delete draft.colorScale;
        setData(layer, "category", undefined);
      } else {
        setData(layer, "category", column);
      }
    });
  }

  function removeLayer(index: number) {
    editLayers((layers) => {
      layers.splice(index, 1);
    });
  }

  function columnOptions(table: string, filter: (jsType: string | null) => boolean) {
    return (context.tables[table]?.columns ?? [])
      .filter((c) => filter(c.jsType))
      .map((c) => ({ value: c.name, label: `${c.name} (${c.type})` }));
  }

  let tableOptions = $derived(Object.keys(context.tables).map((t) => ({ value: t, label: t })));

  const isNumeric = (t: string | null) => t == "number";
  const isCategorical = (t: string | null) => t == "string" || t == "number" || t == "Date";
  const isString = (t: string | null) => t == "string";

  // Special values of the color select.
  const colorMain = "$main";
  const colorFixed = "$fixed";
  /** The value of field selects for a SQL expression field. */
  const customField = "$sql";

  /** The value of a field select: the column name, or `customField` for a SQL expression. */
  function fieldValue(field: EmbeddingField | null | undefined): string | null | undefined {
    return field == null || typeof field == "string" ? field : customField;
  }

  /** The option of a field select for a SQL expression field (if the field is one). */
  function customFieldOption(field: EmbeddingField | null | undefined) {
    return field != null && typeof field != "string" ? [{ value: customField, label: "Custom (SQL)" }] : [];
  }

  /** The value of the color select: the main category field, a fixed color, or a custom category column. */
  function colorMode(data: LayerData): string {
    if (data.categoryIsMain) {
      return colorMain;
    }
    return fieldValue(data.category) ?? colorFixed;
  }

  /** The label of a field (for the main category option). */
  function fieldLabel(field: EmbeddingField): string {
    return typeof field == "string" ? field : "Custom (SQL)";
  }

  function layerValue(layer: LayerSpec, index: number) {
    let id = layer.id ?? String(index);
    return layerValues.find((v) => v.id == id);
  }
</script>

{#snippet rowLabel(label: string)}
  <div class="w-20 flex-none text-sm text-slate-500 dark:text-slate-400 select-none">{label}</div>
{/snippet}

<div class="flex flex-col gap-2 max-h-[60vh] overflow-y-auto">
  {#each listedLayers as { layer, data, index }}
    {@const table = data.table}
    {@const isMainTable = data.isDataTable}
    {@const values = layerValue(layer, index)}
    {@const TypeIcon = typeIcons[layer.type]}
    <div
      class="group bg-white dark:bg-slate-900 p-2 rounded-md border border-slate-200 dark:border-slate-700 flex flex-col gap-2"
    >
      <div class="flex items-center gap-2">
        <div class="flex-1 font-medium text-slate-700 dark:text-slate-300 select-none flex items-center gap-2">
          <TypeIcon class="w-5 h-5 text-slate-400 dark:text-slate-500 flex-none" />
          {typeLabels[layer.type]}
        </div>
        <div class="flex-none flex gap-1 sm:opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
          <CornerButton icon={IconClose} title="Remove layer" onClick={() => removeLayer(index)} />
        </div>
      </div>
      <div class="grid grid-cols-2 gap-x-4 gap-y-2 items-center">
        {#if tableOptions.length > 1 || !isMainTable}
          <div class="col-span-2 flex gap-2 items-center">
            {@render rowLabel("Table")}
            <Select
              class="flex-1 min-w-0"
              value={table}
              onChange={(v) =>
                updateLayer(index, (l) => {
                  // Fields and colors from the previous table don't apply to the new one.
                  delete l.data;
                  delete l.color;
                  delete l.colorScale;
                  // The provided labels are in the main table's coordinates: use automatic labels on other tables.
                  if (l.labels == "$labels" && v != context.table) {
                    delete l.labels;
                  }
                  if (v != dataTable) {
                    setData(l, "table", v);
                    // Pre-fill x and y if the new table has columns with the same names as the main data.
                    let names = new Set((context.tables[v]?.columns ?? []).map((c) => c.name));
                    let { x, y } = spec.data;
                    if (typeof x == "string" && typeof y == "string" && names.has(x) && names.has(y)) {
                      setData(l, "x", x);
                      setData(l, "y", y);
                    }
                  }
                })}
              options={tableOptions}
            />
            {#if !isMainTable}
              {#each ["x", "y"] as const as axis}
                <Select
                  class="flex-1 min-w-0"
                  label={axis.toUpperCase()}
                  value={fieldValue(data[axis])}
                  placeholder="(select)"
                  onChange={(v) => {
                    if (v == customField) {
                      return;
                    }
                    updateLayer(index, (l) => {
                      setData(l, axis, v);
                    });
                  }}
                  options={[...customFieldOption(data[axis]), ...columnOptions(table, isNumeric)]}
                />
              {/each}
            {/if}
          </div>
          {#if !isMainTable && (data.x == null || data.y == null)}
            <div class="col-span-2 text-xs text-amber-600 dark:text-amber-400 select-none">
              Choose the X and Y columns to show this layer.
            </div>
          {/if}
        {/if}
        {#if layer.type == "points" || layer.type == "density"}
          {@const mode = colorMode(data)}
          <div class="flex gap-2 items-center">
            {@render rowLabel("Color")}
            <Select
              class="flex-1 min-w-0"
              value={mode}
              onChange={(v) => {
                if (v == mode || v == customField) {
                  return;
                }
                if (v != colorMain && v != colorFixed) {
                  setLayerCategory(index, v, isMainTable);
                  return;
                }
                updateLayer(index, (l) => {
                  setData(l, "category", undefined);
                  delete l.colorScale;
                  if (v == colorMain) {
                    delete l.color;
                  } else {
                    l.color = l.color ?? mainCategoryLegend?.items[0]?.color ?? defaultColor;
                  }
                });
              }}
              options={[
                ...(isMainTable && spec.data.category != null
                  ? [{ value: colorMain, label: `Main: ${fieldLabel(spec.data.category)}` }]
                  : []),
                { value: colorFixed, label: "Fixed color" },
                "---",
                ...(mode != colorMain ? customFieldOption(data.category) : []),
                ...columnOptions(table, isCategorical),
              ]}
            />
            {#if mode == colorFixed}
              <ColorPicker
                value={layer.color ?? defaultColor}
                onChange={(color) =>
                  updateLayer(index, (l) => {
                    setData(l, "category", undefined);
                    l.color = color;
                  })}
              />
            {/if}
          </div>
        {/if}
        {#if layer.type == "labels"}
          {#if Array.isArray(layer.labels)}
            <div class="text-sm text-slate-500 dark:text-slate-400 select-none">
              {layer.labels.length} static labels
            </div>
          {:else if (providedLabels != null && table == context.table) || layer.labels == "$labels"}
            <div class="flex gap-2 items-center">
              {@render rowLabel("Source")}
              <Select
                class="flex-1 min-w-0"
                value={layer.labels == "$labels" ? "$labels" : "automatic"}
                onChange={(v) =>
                  updateLayer(index, (l) => {
                    if (v == "$labels") {
                      l.labels = "$labels";
                    } else {
                      delete l.labels;
                    }
                  })}
                options={[
                  { value: "automatic", label: "Automatic" },
                  { value: "$labels", label: `Provided (${providedLabels?.length ?? 0})` },
                ]}
              />
            </div>
          {/if}
          {#if layer.labels == null}
            <div class="flex gap-2 items-center">
              {@render rowLabel("Text")}
              <Select
                class="flex-1 min-w-0"
                value={fieldValue(data.text)}
                onChange={(v) => {
                  if (v == customField) {
                    return;
                  }
                  updateLayer(index, (l) => {
                    setData(l, "text", v);
                  });
                }}
                options={[
                  { value: null, label: "--" },
                  ...customFieldOption(data.text),
                  ...columnOptions(table, isString),
                ]}
              />
            </div>
          {/if}
        {/if}
        <!-- Labels have a default opacity; points and density layers adjust it with the zoom level. -->
        <SliderRow
          label="Opacity"
          value={typeof layer.opacity == "number" ? layer.opacity : (values?.opacity ?? 1)}
          min={0}
          max={1}
          step={0.01}
          toggle={{
            label: layer.type == "labels" ? "Default" : "Auto",
            active: layer.opacity == null || layer.opacity == "auto",
            title:
              layer.type == "labels"
                ? "Use the default opacity"
                : "Adjust the opacity automatically with the zoom level",
            onClick: () =>
              updateLayer(index, (l) => {
                delete l.opacity;
              }),
          }}
          onChange={(v) =>
            updateLayer(index, (l) => {
              l.opacity = v;
            })}
        />
        {#if layer.type == "points"}
          <SliderRow
            label="Point Size"
            value={layer.pointSize ?? (values?.type == "points" ? values.pointSize : 1)}
            min={0.1}
            max={10}
            step={0.05}
            toggle={{
              label: "Auto",
              active: layer.pointSize == null,
              title: "Adjust the point size automatically with the zoom level",
              onClick: () =>
                updateLayer(index, (l) => {
                  delete l.pointSize;
                }),
            }}
            onChange={(v) =>
              updateLayer(index, (l) => {
                l.pointSize = v;
              })}
          />
        {/if}
        {#if layer.type == "density"}
          <div class="flex gap-3 items-center">
            <CheckBox
              label="Fill"
              bind:checked={
                () => layer.fill !== false,
                (v) =>
                  updateLayer(index, (l) => {
                    l.fill = v;
                  })
              }
            />
            <CheckBox
              label="Contours"
              bind:checked={
                () => layer.contours !== false,
                (v) =>
                  updateLayer(index, (l) => {
                    l.contours = v;
                  })
              }
            />
          </div>
        {/if}
        <!-- Static and provided labels don't depend on the data, so they have no filter. -->
        {#if !(layer.type == "labels" && layer.labels != null)}
          <CheckBox
            label={layer.type == "labels" ? "Update with filter" : "Apply filter"}
            bind:checked={
              () => data.filtered,
              (v) =>
                updateLayer(index, (l) => {
                  // Keep the spec minimal: omit `filter` when it matches the default.
                  if (v == isFilteredByDefault(l.type, isMainTable)) {
                    delete l.filter;
                  } else {
                    l.filter = v ? "$filter" : null;
                  }
                })
            }
          />
        {/if}
      </div>
    </div>
  {/each}
  <div class="flex gap-2 items-center">
    <span class="text-sm text-slate-500 dark:text-slate-400 select-none">Add:</span>
    <Button label="Points" onClick={() => addLayer("points")} />
    <Button label="Density" onClick={() => addLayer("density")} />
    <Button label="Labels" onClick={() => addLayer("labels")} />
  </div>
</div>

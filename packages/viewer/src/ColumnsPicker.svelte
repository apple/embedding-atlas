<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { untrack } from "svelte";

  import Button from "./lib/widgets/Button.svelte";
  import ComboBox from "./lib/widgets/ComboBox.svelte";
  import Select from "./lib/widgets/Select.svelte";

  import { jsTypeFromDBType } from "./lib/database_utils.js";

  // Predefined embedding models. The default is the first model.
  const textModels = [
    "Xenova/all-MiniLM-L6-v2",
    "Xenova/multilingual-e5-small",
    "Xenova/multilingual-e5-base",
    "Xenova/multilingual-e5-large",
  ];
  const imageModels = [
    "Xenova/dinov2-small",
    "Xenova/dinov2-base",
    "Xenova/dinov2-large",
    "Xenova/dino-vitb8",
    "Xenova/dino-vits8",
    "Xenova/dino-vitb16",
    "Xenova/dino-vits16",
  ];

  export interface Value {
    text: string | null;
    embedding:
      | {
          precomputed: { x: string; y: string };
        }
      | { compute: { column: string; type: "text" | "image"; model: string } }
      | null;
  }

  interface Props {
    columns: { column_name: string; column_type: string }[];
    onConfirm: (value: Value) => void;
  }

  let { columns, onConfirm }: Props = $props();

  let embeddingMode = $state<"precomputed" | "from-text" | "from-image" | "none">("precomputed");

  let embeddingXColumn: string | null = $state(null);
  let embeddingYColumn: string | null = $state(null);
  let embeddingTextColumn: string | null = $state(null);
  let embeddingTextModel: string | null = $state(null);
  let embeddingImageColumn: string | null = $state(null);
  let embeddingImageModel: string | null = $state(null);

  let textColumn: string | null = $state(null);

  let numericalColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "number"));
  let stringColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "string"));

  $effect.pre(() => {
    let c = textColumn;
    if (untrack(() => embeddingTextColumn == null)) {
      embeddingTextColumn = c;
    }
  });

  function confirm() {
    let value: Value = { text: textColumn, embedding: null };
    if (embeddingMode == "precomputed" && embeddingXColumn != null && embeddingYColumn != null) {
      value.embedding = { precomputed: { x: embeddingXColumn, y: embeddingYColumn } };
    }
    if (embeddingMode == "from-text" && embeddingTextColumn != null) {
      let model = embeddingTextModel?.trim() ?? "";
      if (model == null || model == "") {
        model = textModels[0];
      }
      value.embedding = { compute: { column: embeddingTextColumn, type: "text", model: model } };
    }
    if (embeddingMode == "from-image" && embeddingImageColumn != null) {
      let model = embeddingImageModel?.trim() ?? "";
      if (model == null || model == "") {
        model = imageModels[0];
      }
      value.embedding = { compute: { column: embeddingImageColumn, type: "image", model: model } };
    }
    onConfirm?.(value);
  }
</script>

<div
  class="flex flex-col p-4 w-[40rem] border rounded-md bg-slate-50 border-slate-300 dark:bg-slate-900 dark:border-slate-700"
>
  <div class="flex flex-col gap-2 pb-4">
    <!-- Text column -->
    <h2 class="text-slate-500 dark:text-slate-500">Search and Tooltip (optional)</h2>
    <p class="text-sm text-slate-400 dark:text-slate-600">
      The selected column, if any, will be used for full-text search and tooltips. Choose a column with freeform text,
      such as a description, chat messages, or a summary.
    </p>
    <div class="w-full flex flex-row items-center">
      <div class="w-[4rem] dark:text-slate-400">Text</div>
      <Select
        class="flex-1"
        value={textColumn}
        onChange={(v) => (textColumn = v)}
        options={[
          { value: null, label: "(none)" },
          ...stringColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
        ]}
      />
    </div>
    <div class="my-2"></div>
    <!-- Embedding Config -->
    <h2 class="text-slate-500 dark:text-slate-500">Embedding View (optional)</h2>
    <p class="text-sm text-slate-400 dark:text-slate-600">
      To enable the embedding view, you can either (a) pick a pair of pre-computed X and Y columns; or (b) pick a text
      column and compute the embedding projection in browser. For large data, it's recommended to pre-compute the
      embedding and its 2D projection.
    </p>
    <div class="flex rounded overflow-hidden size-fit gap-[1px] bg-slate-100 dark:bg-slate-800 mb-2">
      {#each [["precomputed", "Precomputed"], ["from-text", "From Text"], ["from-image", "From Image"], ["none", "Disabled"]] as [mode, label]}
        <button
          class="bg-slate-200 dark:bg-slate-700 dark:text-slate-400 px-2 py-1"
          class:!bg-slate-500={mode == embeddingMode}
          class:!text-slate-100={mode == embeddingMode}
          onclick={() => (embeddingMode = mode as any)}
        >
          {label}
        </button>
      {/each}
    </div>
    {#if embeddingMode == "precomputed"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[4rem] dark:text-slate-400">X</div>
        <Select
          class="flex-1"
          value={embeddingXColumn}
          onChange={(v) => (embeddingXColumn = v)}
          options={[
            { value: null, label: "(none)" },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[4rem] dark:text-slate-400">Y</div>
        <Select
          class="flex-1"
          value={embeddingYColumn}
          onChange={(v) => (embeddingYColumn = v)}
          options={[
            { value: null, label: "(none)" },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
    {:else if embeddingMode == "from-text"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[4rem] dark:text-slate-400">Text</div>
        <Select
          class="flex-1"
          value={embeddingTextColumn}
          onChange={(v) => (embeddingTextColumn = v)}
          options={[
            { value: null, label: "(none)" },
            ...stringColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[4rem] dark:text-slate-400">Model</div>
        <ComboBox
          className="flex-1"
          value={embeddingTextModel}
          placeholder="(default {textModels[0]})"
          onChange={(v) => (embeddingTextModel = v)}
          options={textModels}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        Computing the embedding and 2D projection in browser may take a while.
      </p>
    {:else if embeddingMode == "from-image"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[4rem] dark:text-slate-400">Image</div>
        <Select
          class="flex-1"
          value={embeddingImageColumn}
          onChange={(v) => (embeddingImageColumn = v)}
          options={[
            { value: null, label: "(none)" },
            ...columns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[4rem] dark:text-slate-400">Model</div>
        <ComboBox
          className="flex-1"
          value={embeddingImageModel}
          placeholder="(default {imageModels[0]})"
          onChange={(v) => (embeddingImageModel = v)}
          options={imageModels}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        Computing the embedding and 2D projection in browser may take a while.
      </p>
    {/if}
  </div>
  <div class="w-full flex flex-row items-center mt-4">
    <div class="flex-1"></div>
    <Button
      label="Confirm"
      class="w-40 justify-center"
      onClick={() => {
        confirm();
      }}
    />
  </div>
</div>

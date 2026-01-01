<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { untrack } from "svelte";

  import Button from "../../widgets/Button.svelte";
  import ComboBox from "../../widgets/ComboBox.svelte";
  import SegmentedControl from "../../widgets/SegmentedControl.svelte";
  import Select from "../../widgets/Select.svelte";

  import { EMBEDDING_ATLAS_VERSION } from "../../constants.js";
  import { jsTypeFromDBType } from "../../utils/database.js";

  // Predefined embedding models. The default is the first model.
  const textModels = [
    "Xenova/all-MiniLM-L6-v2",
    "Xenova/paraphrase-multilingual-mpnet-base-v2",
    "Xenova/multilingual-e5-small",
    "Xenova/multilingual-e5-base",
    "Xenova/multilingual-e5-large",
    "Xenova/bge-small-en-v1.5",
    "Xenova/bge-base-en-v1.5",
  ];
  const imageModels = [
    // DINOv2 models (recommended for qualitative analysis)
    "Xenova/dinov2-small",
    "Xenova/dinov2-base",
    "Xenova/dinov2-large",
    // Original DINO models
    "Xenova/dino-vitb8",
    "Xenova/dino-vits8",
    "Xenova/dino-vitb16",
    "Xenova/dino-vits16",
    // CLIP models (good for image-text alignment)
    "Xenova/clip-vit-base-patch32",
    "Xenova/clip-vit-base-patch16",
    // SigLIP models
    "Xenova/siglip-base-patch16-224",
    // Note: I-JEPA models require custom implementation
    // Add "facebook/ijepa_vith14_1k" when available in Transformers.js
  ];

  export interface Settings {
    version: string;
    text?: string;
    embedding?:
      | {
          precomputed: { x: string; y: string; neighbors?: string };
        }
      | { compute: { column: string; type: "text" | "image"; model: string } };
  }

  interface Props {
    columns: { column_name: string; column_type: string }[];
    onConfirm: (value: Settings) => void;
  }

  let { columns, onConfirm }: Props = $props();

  let embeddingMode = $state<"precomputed" | "from-text" | "from-image" | "none">("precomputed");

  let textColumn: string | undefined = $state(undefined);

  let embeddingXColumn: string | undefined = $state(undefined);
  let embeddingYColumn: string | undefined = $state(undefined);
  let embeddingNeighborsColumn: string | undefined = $state(undefined);
  let embeddingTextColumn: string | undefined = $state(undefined);
  let embeddingTextModel: string | undefined = $state(undefined);
  let embeddingImageColumn: string | undefined = $state(undefined);
  let embeddingImageModel: string | undefined = $state(undefined);

  let numericalColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "number"));
  let stringColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "string"));

  $effect.pre(() => {
    let c = textColumn;
    if (untrack(() => embeddingTextColumn == undefined)) {
      embeddingTextColumn = c;
    }
  });

  function confirm() {
    let value: Settings = { version: EMBEDDING_ATLAS_VERSION, text: textColumn };
    if (embeddingMode == "precomputed" && embeddingXColumn != undefined && embeddingYColumn != undefined) {
      value.embedding = {
        precomputed: {
          x: embeddingXColumn,
          y: embeddingYColumn,
          neighbors: embeddingNeighborsColumn != undefined ? embeddingNeighborsColumn : undefined,
        },
      };
    }
    if (embeddingMode == "from-text" && embeddingTextColumn != undefined) {
      let model = embeddingTextModel?.trim() ?? "";
      if (model == undefined || model == "") {
        model = textModels[0];
      }
      value.embedding = { compute: { column: embeddingTextColumn, type: "text", model: model } };
    }
    if (embeddingMode == "from-image" && embeddingImageColumn != undefined) {
      let model = embeddingImageModel?.trim() ?? "";
      if (model == undefined || model == "") {
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
      <div class="w-[6rem] dark:text-slate-400">Text</div>
      <Select
        class="flex-1 min-w-0"
        value={textColumn}
        onChange={(v) => (textColumn = v)}
        options={[
          { value: undefined, label: "(none)" },
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
    <div class="flex items-start">
      <SegmentedControl
        value={embeddingMode}
        onChange={(v) => (embeddingMode = v as any)}
        options={[
          { value: "precomputed", label: "Pre-computed" },
          { value: "from-text", label: "From Text" },
          { value: "from-image", label: "From Image" },
          { value: "none", label: "None" },
        ]}
      />
    </div>
    {#if embeddingMode == "precomputed"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">X</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingXColumn}
          onChange={(v) => (embeddingXColumn = v)}
          options={[
            { value: undefined, label: "(none)" },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Y</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingYColumn}
          onChange={(v) => (embeddingYColumn = v)}
          options={[
            { value: undefined, label: "(none)" },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Neighbors</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingNeighborsColumn}
          onChange={(v) => (embeddingNeighborsColumn = v)}
          options={[
            { value: undefined, label: "(none)" },
            ...columns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        Neighbors column should contain pre-computed nearest neighbors in format: <code
          >{`{ "ids": [n1, n2, ...], "distances": [d1, d2, ...] }`}</code
        >. IDs should be zero-based row indices.
      </p>
    {:else if embeddingMode == "from-text"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Text</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingTextColumn}
          onChange={(v) => (embeddingTextColumn = v)}
          options={[
            { value: undefined, label: "(none)" },
            ...stringColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Model</div>
        <ComboBox
          className="flex-1"
          value={embeddingTextModel}
          placeholder="(default {textModels[0]})"
          onChange={(v) => (embeddingTextModel = v)}
          options={textModels}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        Computing the embedding and 2D projection in browser may take a while. The model will be loaded with
        Transformers.js.
      </p>
    {:else if embeddingMode == "from-image"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Image</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingImageColumn}
          onChange={(v) => (embeddingImageColumn = v)}
          options={[
            { value: undefined, label: "(none)" },
            ...columns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Model</div>
        <ComboBox
          className="flex-1"
          value={embeddingImageModel}
          placeholder="(default {imageModels[0]})"
          onChange={(v) => (embeddingImageModel = v)}
          options={imageModels}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        Computing the embedding and 2D projection in browser may take a while. The model will be loaded with
        Transformers.js.
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

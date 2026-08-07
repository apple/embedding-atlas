<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { UMAPOptions } from "@embedding-atlas/umap-wasm";
  import { untrack } from "svelte";
  import { get } from "svelte/store";

  import ModelNameInput from "../../views/ModelNameInput.svelte";
  import ProviderConfigForm from "../../views/ProviderConfigForm.svelte";
  import Button from "../../widgets/Button.svelte";
  import CheckBox from "../../widgets/CheckBox.svelte";
  import DisclosureButton from "../../widgets/DisclosureButton.svelte";
  import NumberInput from "../../widgets/NumberInput.svelte";
  import SegmentedControl from "../../widgets/SegmentedControl.svelte";
  import Select from "../../widgets/Select.svelte";

  import { EMBEDDING_ATLAS_VERSION } from "../../constants.js";
  import { defaultModels } from "../../inference/model_config_store.js";
  import { inferProvider } from "../../inference/resolve.js";
  import { jsTypeFromDBType } from "../../utils/database.js";

  // Fallbacks when the user clears the model field on confirm.
  const DEFAULT_TEXT_MODEL = "Xenova/all-MiniLM-L6-v2";
  const DEFAULT_IMAGE_MODEL = "Xenova/dinov2-small";

  export interface Settings {
    version: string;
    text?: string;
    embedding?:
      | {
          precomputed: { x: string; y: string; z?: string; neighbors?: string };
        }
      | {
          compute: {
            column: string;
            type: "text" | "image";
            model: string;
            umapOptions?: UMAPOptions;
            /** Number of output dimensions for the UMAP projection (default: 2). 3 computes a
             *  3D projection and opens a navigable 3D embedding view instead of a flat plane. */
            dimensions?: 2 | 3;
          };
        };
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
  let embeddingZColumn: string | undefined = $state(undefined);
  let embeddingNeighborsColumn: string | undefined = $state(undefined);
  let embeddingTextColumn: string | undefined = $state(undefined);
  let embeddingImageColumn: string | undefined = $state(undefined);

  // Per-file model picks. Initialised once from the global default; edits stay local —
  // we don't want a per-file pick to silently overwrite the user's saved default.
  let embeddingTextModel: string = $state(get(defaultModels).embedding);
  let embeddingImageModel: string = $state(DEFAULT_IMAGE_MODEL);

  let umapMinDist = $state(0.1);
  let umapNNeighbors = $state(15);
  let umapGpu = $state(true);
  let compute3D = $state(false);

  let umapOptions = $derived<UMAPOptions>({
    minDist: umapMinDist,
    nNeighbors: umapNNeighbors,
    gpu: umapGpu,
  });

  let numericalColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "number"));
  let stringColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "string"));

  // Provider that the currently selected embedding model routes to, so we can surface
  // the relevant connection settings (e.g. OpenAI endpoint / API key) inline.
  let activeModel = $derived(embeddingMode === "from-image" ? embeddingImageModel : embeddingTextModel);
  let activeProvider = $derived(activeModel.trim() === "" ? null : inferProvider(activeModel.trim()));

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
          z: embeddingZColumn != undefined ? embeddingZColumn : undefined,
          neighbors: embeddingNeighborsColumn != undefined ? embeddingNeighborsColumn : undefined,
        },
      };
    }
    if (embeddingMode == "from-text" && embeddingTextColumn != undefined) {
      let model = embeddingTextModel.trim();
      if (model == "") {
        model = DEFAULT_TEXT_MODEL;
      }
      value.embedding = {
        compute: {
          column: embeddingTextColumn,
          type: "text",
          model: model,
          umapOptions,
          dimensions: compute3D ? 3 : 2,
        },
      };
    }
    if (embeddingMode == "from-image" && embeddingImageColumn != undefined) {
      let model = embeddingImageModel.trim();
      if (model == "") {
        model = DEFAULT_IMAGE_MODEL;
      }
      value.embedding = {
        compute: {
          column: embeddingImageColumn,
          type: "image",
          model: model,
          umapOptions,
          dimensions: compute3D ? 3 : 2,
        },
      };
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
        <div class="w-[6rem] dark:text-slate-400">Z (optional)</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingZColumn}
          onChange={(v) => (embeddingZColumn = v)}
          options={[
            { value: undefined, label: "(none)" },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        Selecting a Z column opens a navigable 3D embedding view instead of a flat 2D plane.
      </p>
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
    {/if}
    {#if embeddingMode == "from-text" || embeddingMode == "from-image"}
      <!-- Model -->
      <div class="w-full flex flex-row items-start">
        <div class="w-[6rem] dark:text-slate-400 mt-1">Model</div>
        <div class="flex-1 min-w-0">
          {#if embeddingMode == "from-text"}
            <ModelNameInput bind:value={embeddingTextModel} modality="text" />
          {:else}
            <ModelNameInput bind:value={embeddingImageModel} modality="image" />
          {/if}
        </div>
      </div>
      <!-- Provider connection / inference settings for the selected model -->
      {#if activeProvider != null}
        <ProviderConfigForm providerType={activeProvider} />
      {/if}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">Dimensions</div>
        <CheckBox bind:checked={compute3D} label="Compute a 3D projection (navigable 3D view)" />
      </div>
      <!-- UMAP settings -->
      <DisclosureButton label="UMAP Settings" class="mt-1">
        <div class="w-full flex flex-row items-center">
          <div class="w-[6rem] dark:text-slate-400">Min Dist</div>
          <NumberInput className="flex-1 min-w-0" bind:value={umapMinDist} min={0} max={1} step={0.01} />
        </div>
        <div class="w-full flex flex-row items-center">
          <div class="w-[6rem] dark:text-slate-400">Neighbors</div>
          <NumberInput className="flex-1 min-w-0" bind:value={umapNNeighbors} min={2} max={200} step={1} />
        </div>
        <div class="w-full flex flex-row items-center">
          <div class="w-[6rem] dark:text-slate-400">GPU</div>
          <CheckBox bind:checked={umapGpu} label="Use WebGPU if available" />
        </div>
      </DisclosureButton>
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

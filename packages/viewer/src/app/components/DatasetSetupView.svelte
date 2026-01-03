<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { untrack } from "svelte";
  import { onMount } from "svelte";

  import Button from "../../widgets/Button.svelte";
  import ComboBox from "../../widgets/ComboBox.svelte";
  import Input from "../../widgets/Input.svelte";
  import SegmentedControl from "../../widgets/SegmentedControl.svelte";
  import Select from "../../widgets/Select.svelte";

  import { IconImport } from "../../assets/icons.js";
  import { EMBEDDING_ATLAS_VERSION } from "../../constants.js";
  import { jsTypeFromDBType } from "../../utils/database.js";
  import { checkBackendCapabilities, type BackendCapabilities } from "../../embedding/backend.js";

  // Browser-compatible embedding models (via Transformers.js)
  const browserTextModels = [
    "Xenova/all-MiniLM-L6-v2",
    "Xenova/paraphrase-multilingual-mpnet-base-v2",
    "Xenova/multilingual-e5-small",
    "Xenova/multilingual-e5-base",
    "Xenova/multilingual-e5-large",
    "Xenova/bge-small-en-v1.5",
    "Xenova/bge-base-en-v1.5",
  ];

  const browserImageModels = [
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
  ];

  // Backend-only models (require Python backend)
  const backendImageModels = [
    "facebook/ijepa_vith14_1k",  // I-JEPA (recommended for video/image analysis)
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "google/vit-base-patch16-384",
    "openai/clip-vit-base-patch32",
  ];

  // Backend capabilities
  let backendCapabilities = $state<BackendCapabilities | null>(null);
  let backendAvailable = $derived(backendCapabilities !== null && backendCapabilities.embedding_computation);

  // Combined model lists based on backend availability
  let textModels = $derived(browserTextModels);
  let imageModels = $derived(
    backendAvailable
      ? [...browserImageModels, ...backendImageModels]
      : browserImageModels
  );

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
    columns?: { column_name: string; column_type: string }[];
    onLoadData: (inputs: (File | { url: string })[]) => void;
    onConfirm: (value: Settings) => void;
    onChangeDataset?: () => void;
    dataLoaded?: boolean;
  }

  let { columns, onLoadData, onConfirm, onChangeDataset, dataLoaded = false }: Props = $props();

  // File upload state
  let isDragging = $state(false);
  let fileInput: HTMLInputElement;
  let urlInputValue = $state("");

  // Column mapping state
  let embeddingMode = $state<"precomputed" | "from-text" | "from-image" | "none">("precomputed");
  let textColumn: string | undefined = $state(undefined);
  let embeddingXColumn: string | undefined = $state(undefined);
  let embeddingYColumn: string | undefined = $state(undefined);
  let embeddingNeighborsColumn: string | undefined = $state(undefined);
  let embeddingTextColumn: string | undefined = $state(undefined);
  let embeddingTextModel: string | undefined = $state(undefined);
  let embeddingImageColumn: string | undefined = $state(undefined);
  let embeddingImageModel: string | undefined = $state(undefined);

  let numericalColumns = $derived(
    columns?.filter((x) => jsTypeFromDBType(x.column_type) == "number") ?? []
  );
  let stringColumns = $derived(
    columns?.filter((x) => jsTypeFromDBType(x.column_type) == "string") ?? []
  );

  $effect.pre(() => {
    let c = textColumn;
    if (untrack(() => embeddingTextColumn == undefined)) {
      embeddingTextColumn = c;
    }
  });

  // Check backend capabilities on mount
  onMount(async () => {
    backendCapabilities = await checkBackendCapabilities();
  });

  // Helper function to determine if a model requires backend
  function isBackendModel(model: string): boolean {
    return backendImageModels.includes(model);
  }

  // Helper function to get model label with backend indicator
  function getModelLabel(model: string): string {
    const isBackend = isBackendModel(model);
    if (model === "facebook/ijepa_vith14_1k") {
      return isBackend ? "I-JEPA ViT-H/14 (backend)" : "I-JEPA ViT-H/14";
    }
    return isBackend ? `${model} (backend)` : model;
  }

  // File upload handlers
  function handleDragOver(event: any) {
    event.preventDefault();
    isDragging = true;
  }

  function handleDragLeave() {
    isDragging = false;
  }

  function handleDrop(event: any) {
    event.preventDefault();
    isDragging = false;
    if (event.dataTransfer.files) {
      let files: File[] = Array.from(event.dataTransfer.files);
      if (files.length > 0 && files.every(isValidFile)) {
        onLoadData(files);
      }
    }
  }

  function handleFileSelect(event: any) {
    let files: File[] = Array.from(event.target.files);
    if (files.length > 0 && files.every(isValidFile)) {
      onLoadData(files);
    }
  }

  function isValidFile(file: File) {
    const extensions = [".csv", ".parquet", ".json", ".jsonl"];
    for (let ext of extensions) {
      if (file.name.toLowerCase().endsWith(ext.toLowerCase())) {
        return true;
      }
    }
    return false;
  }

  function handleUrlSubmit() {
    let url = urlInputValue.trim();
    if (url == "") {
      return;
    }
    onLoadData([{ url: url }]);
  }

  function handleChangeDataset() {
    // Reset the view to allow selecting a new dataset
    urlInputValue = "";
    onChangeDataset?.();
  }

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
  class="flex flex-col gap-4 p-6 max-w-4xl w-full border rounded-md bg-slate-50 border-slate-300 dark:bg-slate-900 dark:border-slate-700"
>
  <!-- Dataset Selection Section -->
  <div class="flex flex-col gap-3">
    <h2 class="text-lg font-semibold text-slate-700 dark:text-slate-300">Select Dataset</h2>

    {#if !dataLoaded}
      <!-- File Upload -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <button
        class="flex flex-col items-center w-full justify-center py-16 border-2 border-dashed rounded-md transition-all
          {isDragging
          ? 'bg-white border-slate-500 dark:bg-slate-800 dark:border-slate-500'
          : 'bg-white border-slate-300 dark:bg-slate-800 dark:border-slate-700'}"
        ondragover={handleDragOver}
        ondragleave={handleDragLeave}
        ondrop={handleDrop}
        onclick={() => {
          fileInput.click();
        }}
      >
        <div class="text-center space-y-2">
          <p class="text-slate-600 dark:text-slate-400">
            {isDragging ? "Drop your files here" : "Drag & drop files here or click to select"}
          </p>
          <p class="text-sm text-slate-400 dark:text-slate-600">Accepted file types: JSON, CSV, Parquet</p>
        </div>
        <input
          bind:this={fileInput}
          type="file"
          class="hidden"
          accept=".csv,.parquet,.json,.jsonl"
          multiple={true}
          onchange={handleFileSelect}
        />
      </button>

      <!-- URL Input -->
      <div class="w-full text-center text-slate-400 dark:text-slate-500">&mdash; or &mdash;</div>
      <div class="flex gap-2">
        <Input className="flex-1" bind:value={urlInputValue} placeholder="Enter data URL" onEnter={handleUrlSubmit} />
        <Button onClick={handleUrlSubmit} icon={IconImport} title="Load from URL" />
      </div>

      <div class="text-sm text-slate-400 dark:text-slate-500 mt-2">
        All data remains confined to the browser and is not transmitted elsewhere.
      </div>
    {:else}
      <!-- Data loaded - show compact status with change option -->
      <div class="flex items-center justify-between p-3 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md">
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 bg-green-500 rounded-full"></div>
          <span class="text-sm text-slate-600 dark:text-slate-400">Dataset loaded ({columns?.length ?? 0} columns)</span>
        </div>
        <Button label="Change Dataset" onClick={handleChangeDataset} class="text-sm" />
      </div>
    {/if}
  </div>

  <!-- Column Mapping Section - Only shown when data is loaded -->
  {#if dataLoaded && columns && columns.length > 0}
    <div class="border-t border-slate-300 dark:border-slate-700 pt-4"></div>

    <div class="flex flex-col gap-3">
      <h2 class="text-lg font-semibold text-slate-700 dark:text-slate-300">Configure Columns</h2>

      <!-- Text column -->
      <div class="flex flex-col gap-2">
        <h3 class="text-sm font-medium text-slate-600 dark:text-slate-400">Search and Tooltip (optional)</h3>
        <p class="text-sm text-slate-400 dark:text-slate-600">
          The selected column, if any, will be used for full-text search and tooltips. Choose a column with freeform text,
          such as a description, chat messages, or a summary.
        </p>
        <div class="w-full flex flex-row items-center gap-3">
          <div class="w-24 text-sm dark:text-slate-400">Text</div>
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
      </div>

      <!-- Embedding Config -->
      <div class="flex flex-col gap-2 mt-2">
        <h3 class="text-sm font-medium text-slate-600 dark:text-slate-400">Embedding View (optional)</h3>
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

        <!-- Embedding mode options -->
        <div class="mt-2 flex flex-col gap-2">
          {#if embeddingMode == "precomputed"}
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">X</div>
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
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">Y</div>
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
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">Neighbors</div>
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
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">Text</div>
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
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">Model</div>
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
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">Image</div>
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
            <div class="w-full flex flex-row items-center gap-3">
              <div class="w-24 text-sm dark:text-slate-400">Model</div>
              <ComboBox
                className="flex-1"
                value={embeddingImageModel}
                placeholder="(default {imageModels[0]})"
                onChange={(v) => (embeddingImageModel = v)}
                options={imageModels.map(m => ({
                  value: m,
                  label: getModelLabel(m)
                }))}
              />
            </div>
            {#if embeddingImageModel && isBackendModel(embeddingImageModel)}
              <p class="text-sm text-amber-600 dark:text-amber-400">
                This model requires the Python backend for computation. Ensure the backend server is running.
              </p>
            {:else}
              <p class="text-sm text-slate-400 dark:text-slate-600">
                Computing the embedding and 2D projection in browser may take a while. The model will be loaded with
                Transformers.js.
              </p>
            {/if}
            {#if backendAvailable && backendImageModels.length > 0}
              <p class="text-sm text-green-600 dark:text-green-400">
                ✓ Backend available - I-JEPA and other advanced models are enabled
              </p>
            {/if}
          {/if}
        </div>
      </div>
    </div>

    <!-- Confirm button -->
    <div class="w-full flex flex-row items-center justify-end mt-4 pt-4 border-t border-slate-300 dark:border-slate-700">
      <Button
        label="Start Visualization"
        class="px-6 py-2 justify-center bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-600 dark:hover:bg-blue-700"
        onClick={confirm}
      />
    </div>
  {/if}
</div>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ActionButton from "../widgets/ActionButton.svelte";
  import CodeEditor from "../widgets/CodeEditor.svelte";
  import Select from "../widgets/Select.svelte";
  import PanelContainer from "./PanelContainer.svelte";
  import PanelSection from "./PanelSection.svelte";

  import { IconCopy, IconDownload, IconImport } from "../assets/icons.js";

  import type { EmbeddingAtlasState } from "../api.js";
  import { getStoreContext } from "../stores/embedding_atlas_store.js";
  import { deserializePayload, serializePayload } from "../utils/query_payload.js";

  const store = getStoreContext();
  const { colorScheme } = store;

  let exportFormat: "json" | "jsonl" | "csv" | "parquet" = $state("parquet");
  let textValue = $state("");

  async function onCopyStateJSON() {
    let text = JSON.stringify(store.getCurrentState(), null, 2);
    await navigator.clipboard.writeText(text);
  }

  async function onCopyStateBase64() {
    let base64Text = "state=" + (await serializePayload(store.getCurrentState()));
    await navigator.clipboard.writeText(base64Text);
  }

  async function onImportState() {
    let text = textValue;
    let objectValue: EmbeddingAtlasState | undefined = undefined;

    // Try parse as JSON
    try {
      objectValue = JSON.parse(text);
    } catch (error) {}

    if (objectValue == undefined) {
      try {
        objectValue = await deserializePayload(text.replace("state=", ""));
      } catch (error) {}
    }

    if (objectValue == undefined) {
      throw new Error("invalid value");
    }

    textValue = JSON.stringify(objectValue, null, 2);

    store.importState(objectValue);
  }
</script>

<PanelContainer title="Import / Export" class="p-2 flex flex-col gap-2">
  <PanelSection title="Export the current view state">
    <div class="flex gap-2">
      <ActionButton
        icon={IconCopy}
        label="Copy JSON"
        title="Copy the current Embedding Atlas state as JSON to clipboard."
        class="w-48"
        onClick={onCopyStateJSON}
      />
      <ActionButton
        icon={IconCopy}
        label="Copy Base64"
        title="Copy the current Embedding Atlas state as base64 to clipboard."
        class="w-48"
        onClick={onCopyStateBase64}
      />
    </div>
    <div class="text-slate-400 dark:text-slate-500 text-sm">
      Export as JSON to use with the <code>initial_state</code> parameter in the notebook widget, or as base64 to share
      via a URL's <code>state=</code> query parameter.
    </div>
  </PanelSection>
  <PanelSection title="Import a view state">
    <CodeEditor class="w-full h-64" value={textValue} onChange={(v) => (textValue = v)} colorScheme={$colorScheme} />
    <ActionButton
      icon={IconImport}
      label="Import State"
      title="Import the state from the text editor above."
      class="w-48"
      onClick={onImportState}
    />
  </PanelSection>

  {#if store.props.onExportSelection}
    {@const onExportSelection = store.props.onExportSelection}
    <PanelSection title="Export the selected data points">
      <div class="flex gap-2">
        <ActionButton
          icon={IconDownload}
          label="Export Selection"
          title="Export the selected points"
          class="w-48"
          onClick={() => onExportSelection(store.getCurrentPredicate(), exportFormat)}
        />
        <Select
          label="Format"
          value={exportFormat}
          onChange={(v) => (exportFormat = v)}
          options={[
            { value: "parquet", label: "Parquet" },
            { value: "jsonl", label: "JSONL" },
            { value: "json", label: "JSON" },
            { value: "csv", label: "CSV" },
          ]}
        />
      </div>
    </PanelSection>
  {/if}

  {#if store.props.onExportApplication}
    {@const onExportApplication = store.props.onExportApplication}
    <PanelSection title="Export application">
      <div class="flex flex-col gap-2">
        <ActionButton
          icon={IconDownload}
          label="Export Application"
          title="Download a self-contained static web application"
          class="w-48"
          onClick={onExportApplication}
        />
      </div>
      <div class="text-slate-500 dark:text-slate-400 text-sm">
        Export as a self-contained static website. Extract the downloaded zip file and serve it with
        <code>npx http-server</code> or <code>python -m http.server</code>.
      </div>
    </PanelSection>
  {/if}
</PanelContainer>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ComboBox from "../widgets/ComboBox.svelte";

  import { inferProvider } from "../inference/resolve.js";

  interface Props {
    value: string;
    /** When set, drives the preset list (text models vs. image models). */
    modality?: "text" | "image";
  }

  let { value = $bindable(""), modality }: Props = $props();

  const TRANSFORMERS_TEXT_MODELS = [
    "Xenova/all-MiniLM-L6-v2",
    "Xenova/all-MiniLM-L6-v2:q4",
    "Xenova/paraphrase-multilingual-mpnet-base-v2",
    "Xenova/multilingual-e5-small",
    "Xenova/multilingual-e5-base",
    "Xenova/multilingual-e5-large",
  ];
  const TRANSFORMERS_IMAGE_MODELS = [
    "Xenova/dinov2-small",
    "Xenova/dinov2-base",
    "Xenova/dinov2-large",
    "Xenova/dino-vitb8",
    "Xenova/dino-vits8",
    "Xenova/dino-vitb16",
    "Xenova/dino-vits16",
  ];

  const PROVIDER_LABEL: Record<string, string> = {
    "transformers.js": "Transformers.js",
    openai: "OpenAI-compatible",
  };

  let presets = $derived(modality === "image" ? TRANSFORMERS_IMAGE_MODELS : TRANSFORMERS_TEXT_MODELS);

  let provider = $derived(value.trim() === "" ? null : inferProvider(value.trim()));
</script>

<div class="flex flex-col gap-1">
  <ComboBox className="w-full" value={value} placeholder="Model name" options={presets} onChange={(v) => (value = v)} />
  {#if provider}
    <div class="text-sm text-slate-500 dark:text-slate-500">
      Routes to: <span class="font-medium">{PROVIDER_LABEL[provider] ?? provider}</span>
    </div>
  {/if}
</div>

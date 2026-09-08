<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Input from "../widgets/Input.svelte";

  import { providerConfigs, type ProviderType } from "../inference/model_config_store.js";
  import { type ProviderConfig } from "../inference/provider_config.js";

  interface Props {
    providerType: ProviderType;
  }

  let { providerType }: Props = $props();

  // The form owns its binding: read/write this provider's slice of the global
  // config store, so call sites only need to pass `providerType`.
  let config = $derived($providerConfigs[providerType] ?? {});

  function update(patch: Partial<ProviderConfig>) {
    providerConfigs.update((c) => ({ ...c, [providerType]: { ...(c[providerType] ?? {}), ...patch } }));
  }

  // Batch size is an optional positive integer; blank or invalid input clears the
  // override so the provider's built-in default applies.
  function parseBatchSize(v: string): number | undefined {
    const n = Math.floor(Number(v));
    return v.trim() === "" || !Number.isFinite(n) || n < 1 ? undefined : n;
  }
</script>

<div class="flex flex-col gap-2">
  {#if providerType === "openai"}
    <div class="w-full flex flex-row items-center">
      <div class="w-[6rem] dark:text-slate-400">Endpoint</div>
      <Input
        className="flex-1 min-w-0"
        bind:value={() => config.endpoint ?? "", (v) => update({ endpoint: v === "" ? undefined : v })}
        placeholder="https://api.openai.com/v1"
      />
    </div>
    <div class="w-full flex flex-row items-center">
      <div class="w-[6rem] dark:text-slate-400">API key</div>
      <Input
        type="password"
        className="flex-1 min-w-0"
        bind:value={() => config.apiKey ?? "", (v) => update({ apiKey: v === "" ? undefined : v })}
        placeholder="sk-..."
      />
    </div>
    <div class="w-full flex flex-row items-center">
      <div class="w-[6rem] dark:text-slate-400">Batch size</div>
      <Input
        className="flex-1 min-w-0"
        bind:value={
          () => (config.embeddingsMaxBatchSize != null ? String(config.embeddingsMaxBatchSize) : ""),
          (v) => update({ embeddingsMaxBatchSize: parseBatchSize(v) })
        }
        placeholder="128"
      />
    </div>
    <p class="text-sm text-slate-400 dark:text-slate-600">
      API key is stored in this browser's localStorage. Avoid sharing the device or clear it when done.
    </p>
  {:else if providerType === "transformers.js"}
    <div class="w-full flex flex-row items-center">
      <div class="w-[6rem] dark:text-slate-400">Version</div>
      <Input
        className="flex-1 min-w-0"
        bind:value={() => config.version ?? "", (v) => update({ version: v === "" ? undefined : v })}
        placeholder="4.2.0"
      />
    </div>
    <div class="w-full flex flex-row items-center">
      <div class="w-[6rem] dark:text-slate-400">Batch size</div>
      <Input
        className="flex-1 min-w-0"
        bind:value={
          () => (config.embeddingsMaxBatchSize != null ? String(config.embeddingsMaxBatchSize) : ""),
          (v) => update({ embeddingsMaxBatchSize: parseBatchSize(v) })
        }
        placeholder="64"
      />
    </div>
    <p class="text-sm text-slate-400 dark:text-slate-600">
      Version pins the transformers.js library loaded from CDN. Batch size caps the number of inputs per inference call;
      leave blank to use the default.
    </p>
  {/if}
</div>

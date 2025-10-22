<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { nanoid } from "nanoid";

  interface Props {
    label?: string | null | undefined;
    value: any;
    disabled?: boolean;
    placeholder?: string | null;
    class?: string | null;
    options?: (
      | {
          value: any;
          label: string;
        }
      | "---"
    )[];
    onChange?: ((value: any) => void) | null | undefined;
  }

  let {
    label = undefined,
    value,
    disabled = false,
    placeholder = null,
    options = [],
    onChange = undefined,
    class: additionalClasses,
  }: Props = $props();

  let selectElement: HTMLSelectElement | undefined = $state();

  const randomPrefix = nanoid();
  const nullValue = randomPrefix + "_null";
  const undefinedValue = randomPrefix + "_undefined";
  const toSelectValue = (v: any) => (v === null ? nullValue : v === undefined ? undefinedValue : v.toString());
  const fromSelectValue = (v: string) => (v === nullValue ? null : v === undefinedValue ? undefined : v);

  function onChangeRaw(e: any) {
    if (!onChange || !selectElement) {
      return;
    }
    onChange(fromSelectValue(selectElement.value));
  }
</script>

{#if label != null}
  <label class="select-none flex items-center gap-2">
    <span class="text-slate-500 dark:text-slate-400 whitespace-nowrap">{label}</span>
    <select
      class="form-select rounded-md py-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 dark:text-slate-400 text-ellipsis {additionalClasses ??
        ''}"
      bind:this={selectElement}
      value={toSelectValue(value)}
      onchange={onChangeRaw}
      disabled={disabled}
    >
      {#if placeholder != null}
        <option value={null} disabled selected>{placeholder}</option>
      {/if}
      {#each options as option}
        {#if option === "---"}
          <hr />
        {:else}
          <option value={toSelectValue(option.value)}>{option.label}</option>
        {/if}
      {/each}
    </select>
  </label>
{:else}
  <select
    class="form-select rounded-md py-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 dark:text-slate-400 select-none text-ellipsis {additionalClasses ??
      ''}"
    bind:this={selectElement}
    value={toSelectValue(value)}
    onchange={onChangeRaw}
    disabled={disabled}
  >
    {#if placeholder != null}
      <option value={null} disabled selected>{placeholder}</option>
    {/if}
    {#each options as option}
      {#if option === "---"}
        <hr />
      {:else}
        <option value={toSelectValue(option.value)}>{option.label}</option>
      {/if}
    {/each}
  </select>
{/if}

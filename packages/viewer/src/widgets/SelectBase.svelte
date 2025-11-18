<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  export type Option =
    | {
        label: string;
        value: any;
      }
    | "---";

  export interface Props {
    class?: string | null;
    style?: string | null;
    options?: Option[];
    value: any;
    placeholder?: string | null;
    disabled?: boolean;
    onChange?: ((value: any) => void) | null | undefined;
  }

  function toOptionValue(value: any): string {
    if (value === undefined) {
      return "undefined";
    } else {
      return JSON.stringify(value);
    }
  }

  function fromOptionValue(value: string): any {
    if (value === "undefined") {
      return undefined;
    } else {
      return JSON.parse(value);
    }
  }
</script>

<script lang="ts">
  let {
    value,
    disabled = false,
    placeholder = null,
    options = [],
    onChange = undefined,
    class: className,
    style: style,
  }: Props = $props();

  let selectElement: HTMLSelectElement;

  let optionForUndefined = $derived(options.filter((x) => x !== "---" && x.value === undefined)[0]);
</script>

<select
  bind:this={selectElement}
  class={className}
  style={style}
  value={toOptionValue(value)}
  onchange={() => {
    if (selectElement && onChange) {
      onChange(fromOptionValue(selectElement.value));
    }
  }}
  disabled={disabled}
>
  {#if placeholder != null && !optionForUndefined}
    <option value="undefined" disabled selected>{placeholder}</option>
  {/if}
  {#each options as option}
    {#if option === "---"}
      <hr />
    {:else}
      <option value={toOptionValue(option.value)}>{option.label}</option>
    {/if}
  {/each}
</select>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { nanoid } from "nanoid";
  import colors from "tailwindcss/colors";

  interface Props {
    options: { value: any; label: string }[];
    value: any;
    onChange: (value: any) => void;
  }

  let { value, options, onChange }: Props = $props();

  let selectElement: HTMLSelectElement;

  const randomPrefix = nanoid();
  const nullValue = randomPrefix + "_null";
  const undefinedValue = randomPrefix + "_undefined";
  const toSelectValue = (v: any) => (v === null ? nullValue : v === undefined ? undefinedValue : v.toString());
  const fromSelectValue = (v: string) => (v === nullValue ? null : v === undefinedValue ? undefined : v);

  const svgCode = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 20">
    <path d="M 2,8 L 5,12 L 8,8" style="stroke:${colors.gray[500]};stroke-opacity:0.7;stroke-width:1.5;fill:none;stroke-linecap:round;stroke-linejoin:round" />
  </svg>`;
</script>

<select
  bind:this={selectElement}
  class="form-select text-center pl-[4px] pr-[16px] py-0 my-0 border-0 rounded text-sm! text-slate-500 bg-slate-100 dark:text-slate-400 dark:bg-slate-900"
  style:background-image="url(data:image/svg+xml;base64,{btoa(svgCode)})"
  style:background-position="right center"
  value={toSelectValue(value)}
  onchange={() => {
    onChange(fromSelectValue(selectElement.value));
  }}
>
  {#each options as option}
    <option value={toSelectValue(option.value)}>{option.label}</option>
  {/each}
</select>

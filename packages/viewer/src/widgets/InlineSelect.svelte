<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import colors from "tailwindcss/colors";

  import SelectBase, { type Props } from "./SelectBase.svelte";

  let props: Props = $props();

  const svgDataUrl =
    "data:image/svg+xml;base64," +
    btoa(`<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 20">
      <path d="M 2,8 L 5,12 L 8,8" style="stroke:${colors.gray[500]};stroke-opacity:0.7;stroke-width:1.5;fill:none;stroke-linecap:round;stroke-linejoin:round" />
    </svg>`);
</script>

<SelectBase
  {...props}
  class="form-select select-none text-center pl-[4px] pr-[16px] py-0 my-0 border-0 rounded text-sm! text-slate-500 bg-slate-100 dark:text-slate-400 dark:bg-slate-900"
  style="background-image:url({svgDataUrl});background-position:right center"
/>

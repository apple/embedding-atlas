<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import colors from "tailwindcss/colors";

  import { chartIcons } from "../../assets/chart_icons.js";

  let { type, colorScheme }: { type: string; colorScheme: "light" | "dark" } = $props();

  function process(source: string, colorScheme: "light" | "dark"): string {
    source = source.replace("xmlns", 'style="width:100%;height:100%" xmlns');
    let map: Record<string, string> =
      colorScheme == "dark" ? { black: colors.slate[400] } : { black: colors.slate[500] };
    for (let key in map) {
      source = source.replaceAll(key, map[key]);
    }
    return source;
  }
</script>

<div class="w-12 h-12">{@html process(chartIcons[type] ?? "", colorScheme ?? "light")}</div>

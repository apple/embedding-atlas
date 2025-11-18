<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import JSON5 from "json5";

  import CodeEditor from "../../widgets/CodeEditor.svelte";
  import type { ChartSpec } from "../spec/spec.js";

  import jsonSchema from "../chart_types.ts?type=BuiltinChartSpec&json-schema";

  interface Props {
    class?: string;
    initialValue?: ChartSpec;
    colorScheme?: "light" | "dark";
    onChange?: (spec: ChartSpec | undefined) => void;
  }

  let { class: className, initialValue, colorScheme, onChange }: Props = $props();

  let value = JSON5.stringify(initialValue, { space: 2, quote: '"' });

  function parse(textValue: string): ChartSpec | undefined {
    try {
      let newSpec = JSON.parse(JSON.stringify(JSON5.parse(textValue)));
      return newSpec as ChartSpec;
    } catch (e: any) {
      return undefined;
    }
  }
</script>

<CodeEditor
  class={className}
  colorScheme={colorScheme}
  language="json5"
  json={{ schema: jsonSchema }}
  value={value}
  onChange={(newValue) => {
    let spec = parse(newValue);
    if (spec) {
      onChange?.(spec);
    } else {
      onChange?.(undefined);
    }
  }}
/>

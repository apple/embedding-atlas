<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import CodeEditor from "../../widgets/CodeEditor.svelte";
  import Input from "../../widgets/Input.svelte";
  import Container from "../common/Container.svelte";

  import { renderMarkdown } from "../../renderers/markdown.js";
  import type { ChartViewProps } from "../chart.js";
  import type { MarkdownSpec } from "./types.js";

  let { spec, width, height, mode, onSpecChange }: ChartViewProps<MarkdownSpec, {}> = $props();
</script>

{#if mode == "view"}
  <Container width={width} height={height} scrollY={true} class="prose dark:prose-invert max-w-none">
    {@html renderMarkdown(spec.content ?? "")}
  </Container>
{:else}
  <Container width={width} height={height} class="flex flex-col gap-2">
    <Input bind:value={() => spec.title ?? "", (value) => onSpecChange({ title: value })} />
    <CodeEditor
      value={spec.content}
      class={height == null ? "h-64" : "flex-1"}
      language="markdown"
      onChange={(value) => {
        onSpecChange({ content: value });
      }}
    />
  </Container>
{/if}

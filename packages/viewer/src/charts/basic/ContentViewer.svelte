<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import * as SQL from "@uwdata/mosaic-sql";

  import ContentRenderer from "../../renderers/ContentRenderer.svelte";
  import Container from "../common/Container.svelte";

  import type { ChartViewProps } from "../chart.js";
  import type { ContentViewerSpec } from "./types.js";

  let { context, width, height, spec }: ChartViewProps<ContentViewerSpec, {}> = $props();

  let { columnStyles } = context;

  let value = $state<any>(undefined);

  $effect.pre(() =>
    context.highlight.subscribe(async (id) => {
      if (id == null) {
        return;
      }
      let r = await context.coordinator.query(
        SQL.Query.from(context.table)
          .select({ value: spec.field })
          .where(SQL.eq(SQL.column(context.id, context.table), SQL.literal(id))),
      );
      value = r.get(0)?.value ?? undefined;
    }),
  );

  let renderer = $derived($columnStyles[spec.field]?.renderer);
</script>

<Container width={width} height={height} scrollY={true}>
  {#if value != null}
    <ContentRenderer value={value} renderer={renderer} />
  {:else}
    (null)
  {/if}
</Container>

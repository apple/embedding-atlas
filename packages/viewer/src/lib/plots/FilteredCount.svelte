<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { type Selection, makeClient } from "@uwdata/mosaic-core";
  import { Query, sql } from "@uwdata/mosaic-sql";

  import { Context } from "../contexts.js";

  interface Props {
    table: string;
    filter: Selection;
  }

  let { table, filter }: Props = $props();
  const coordinator = Context.coordinator;

  let totalCount: number | null = $state(null);
  let count: number | null = $state(null);

  $effect(() => {
    let props = { coordinator, table, filter };

    totalCount = null;
    count = null;

    props.coordinator.query(Query.from(props.table).select({ count: sql`COUNT(*)::INT` })).then((result: any) => {
      totalCount = result.getChild("count").get(0);
    });

    let client = makeClient({
      coordinator: props.coordinator,
      selection: props.filter,
      query: (predicate) =>
        Query.from(props.table)
          .select({ count: sql`COUNT(*)::INT` })
          .where(predicate),
      queryResult: (result: any) => {
        count = result.getChild("count").get(0);
      },
    });
    return () => {
      client.destroy();
    };
  });
</script>

<div>
  {count?.toLocaleString() ?? ""} <span class="text-slate-500">/ {totalCount?.toLocaleString() ?? ""} points</span>
</div>

<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Coordinator, type Selection, makeClient } from "@uwdata/mosaic-core";
  import { Query, sql } from "@uwdata/mosaic-sql";

  interface Props {
    coordinator: Coordinator;
    table: string;
    filter: Selection;
  }

  let { coordinator, table, filter }: Props = $props();

  let totalCount: number | null = $state(null);
  let count: number | null = $state(null);

  $effect(() => {
    totalCount = null;
    count = null;

    let client = makeClient({
      coordinator: coordinator,
      selection: filter,
      prepare: async () => {
        let result = await coordinator.query(Query.from(table).select({ count: sql`COUNT(*)::INT` }));
        totalCount = result.get(0).count;
      },
      query: (predicate) =>
        Query.from(table)
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

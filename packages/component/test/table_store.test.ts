import { describe, expect, test } from "vitest";

import {
  applyMask,
  chunkQuery,
  MaskedArrays,
  maskPositions,
  TableStore,
  type ColumnSpec,
  type ColumnType,
  type FilterMask,
} from "../src/lib/embedding_view/data/table_store.js";
import * as SQLModule from "@uwdata/mosaic-sql";

function col(name: string, type: ColumnType): ColumnSpec {
  return { expr: SQLModule.column(name), type };
}

/** A fake Mosaic coordinator over an in-memory table (rows in random order, so ORDER BY matters). */
function fakeCoordinator(rows: Record<string, any>[]) {
  let queries: string[] = [];
  let coordinator: any = {
    async query(q: any) {
      let sql = String(q);
      queries.push(sql);
      let order = /ORDER BY "?(\w+)"?/.exec(sql)![1];
      let sorted = [...rows].sort((a, b) => (a[order] < b[order] ? -1 : a[order] > b[order] ? 1 : 0));
      let columns: Record<string, any[]> = {};
      for (let m of sql.matchAll(/(?:\("(\w+)"\)::\w+|"(\w+)"|(rowid)) AS "(\w+)"/g)) {
        let source = m[1] ?? m[2] ?? m[3];
        columns[m[4]] = sorted.map((r) => r[source]);
      }
      return { getChild: (name: string) => ({ toArray: () => columns[name] }) };
    },
  };
  return { coordinator, queries };
}

const rows = [
  { id: 2, x: 2.5, y: -2, c: 1 },
  { id: 0, x: 0.5, y: 0, c: 0 },
  { id: 1, x: 1.5, y: -1, c: 2 },
];

describe("TableStore", () => {
  test("columns requested together are loaded in one query, ordered by the key", async () => {
    let { coordinator, queries } = fakeCoordinator(rows);
    let store = new TableStore(coordinator);
    let [[x], [y, c]] = await Promise.all([
      store.columns("t", "id", [col("x", "float32")]),
      store.columns("t", "id", [col("y", "float32"), col("c", "uint8")]),
    ]);
    expect(queries.length).toBe(1);
    expect(queries[0]).toContain("ORDER BY");
    expect(Array.from(x)).toEqual([0.5, 1.5, 2.5]);
    expect(Array.from(y)).toEqual([0, -1, -2]);
    expect(c).toBeInstanceOf(Uint8Array);
    expect(Array.from(c)).toEqual([0, 2, 1]);
    let index = await store.keyIndex("t", "id");
    expect(index.positionOf(2)).toBe(2);
    expect(queries.length).toBe(1);
  });

  test("the same column in two roles is loaded once", async () => {
    let { coordinator, queries } = fakeCoordinator(rows);
    let store = new TableStore(coordinator);
    let [a, b] = await store.columns("t", "id", [col("x", "float32"), col("x", "float32")]);
    expect(a).toBe(b);
    expect(queries.length).toBe(1);
  });

  test("columns are cached; later columns don't reload the keys", async () => {
    let { coordinator, queries } = fakeCoordinator(rows);
    let store = new TableStore(coordinator);
    await store.columns("t", "id", [col("x", "float32")]);
    let [again] = await store.columns("t", "id", [col("x", "float32")]);
    expect(queries.length).toBe(1);
    expect(Array.from(again)).toEqual([0.5, 1.5, 2.5]);
    await store.columns("t", "id", [col("y", "float32")]);
    expect(queries.length).toBe(2);
    expect(queries[1]).not.toContain('AS "k"');
  });

  test("a failed query can be retried", async () => {
    let { coordinator } = fakeCoordinator(rows);
    let fail = true;
    let query = coordinator.query;
    coordinator.query = async (q: any) => {
      if (fail) {
        throw new Error("boom");
      }
      return query(q);
    };
    let store = new TableStore(coordinator);
    await expect(store.columns("t", "id", [col("x", "float32")])).rejects.toThrow("boom");
    fail = false;
    let [r] = await store.columns("t", "id", [col("x", "float32")]);
    expect(Array.from(r)).toEqual([0.5, 1.5, 2.5]);
  });
});

describe("masks", () => {
  let bits = new Uint32Array([0b1010]);
  let mask: FilterMask = { predicate: "p", predicateString: "p", bits, count: 2 };

  test("applyMask keeps the masked rows", () => {
    expect(Array.from(applyMask(new Float32Array([0, 1, 2, 3]), mask))).toEqual([1, 3]);
    expect(Array.from(applyMask(new Uint8Array([5, 6, 7, 8]), mask))).toEqual([6, 8]);
    let all = new Float32Array([1, 2]);
    expect(applyMask(all, null)).toBe(all);
    expect(applyMask(all, { predicate: null, predicateString: "", bits: null, count: null })).toBe(all);
  });

  test("maskPositions", () => {
    expect(Array.from(maskPositions(mask)!)).toEqual([1, 3]);
    expect(maskPositions(null)).toBe(null);
  });

  test("MaskedArrays memoizes by array and mask", () => {
    let cache = new MaskedArrays();
    let x = new Float32Array([0, 1, 2, 3]);
    let a = cache.get(x, mask);
    expect(cache.get(x, mask)).toBe(a);
    let other: FilterMask = { ...mask };
    expect(cache.get(x, other)).not.toBe(a);
    expect(cache.get(x, null)).toBe(x);
  });
});

describe("TableStore failures", () => {
  test("a bad column doesn't fail the other columns loaded with it", async () => {
    let { coordinator, queries } = fakeCoordinator(rows);
    let query = coordinator.query;
    coordinator.query = async (q: any) => {
      if (String(q).includes('"nope"')) {
        throw new Error("no such column");
      }
      return query(q);
    };
    let store = new TableStore(coordinator);
    let good = store.columns("t", "id", [col("x", "float32")]);
    let bad = store.columns("t", "id", [col("nope", "float32")]);
    await expect(bad).rejects.toThrow("no such column");
    let [x] = await good;
    expect(Array.from(x)).toEqual([0.5, 1.5, 2.5]);
    expect((await store.keyIndex("t", "id")).length).toBe(3);
    // The batch fails (not recorded), then the keys and x succeed on their own (the bad column fails again).
    expect(queries.length).toBe(2);
  });
});

describe("TableStore keys", () => {
  test("a non-unique identifier falls back to rowid", async () => {
    let dup = [
      { rowid: 0, id: 7, x: 0.5 },
      { rowid: 1, id: 7, x: 1.5 },
      { rowid: 2, id: 3, x: 2.5 },
    ];
    let { coordinator, queries } = fakeCoordinator(dup);
    let store = new TableStore(coordinator);
    let warn = console.warn;
    let warnings: string[] = [];
    console.warn = (m: string) => warnings.push(m);
    try {
      let [x] = await store.columns("t", "id", [col("x", "float32")]);
      expect(Array.from(x)).toEqual([0.5, 1.5, 2.5]);
      expect(warnings.length).toBe(1);
      expect(queries.at(-1)).toContain("ORDER BY rowid");
      let index = await store.keyIndex("t", "id");
      expect(index.unique).toBe(true);
      expect(String(await store.keyExpressionOf("t", "id"))).toBe("rowid");
    } finally {
      console.warn = warn;
    }
  });

  test("a failing key column fails the columns with its own error", async () => {
    let { coordinator } = fakeCoordinator(rows);
    let query = coordinator.query;
    coordinator.query = async (q: any) => {
      if (String(q).includes('"bad_id"')) {
        throw new Error("no column bad_id");
      }
      return query(q);
    };
    let store = new TableStore(coordinator);
    let a = store.columns("t", "bad_id", [col("x", "float32")]);
    let b = store.columns("t", "bad_id", [col("y", "float32")]);
    await expect(a).rejects.toThrow("no column bad_id");
    await expect(b).rejects.toThrow("no column bad_id");
  });

  test("tables without rowid get a clear error", async () => {
    let { coordinator } = fakeCoordinator(rows);
    coordinator.query = async () => {
      throw new Error('Binder Error: Referenced column "rowid" not found');
    };
    let store = new TableStore(coordinator);
    await expect(store.columns("v", null, [col("x", "float32")])).rejects.toThrow(/identifier column/);
  });
});

describe("TableStore retain", () => {
  test("columns that aren't retained are loaded again when requested", async () => {
    let { coordinator, queries } = fakeCoordinator(rows);
    let store = new TableStore(coordinator);
    await store.columns("t", "id", [col("x", "float32"), col("y", "float32")]);
    store.retain([{ table: "t", key: "id", columns: [col("x", "float32")] }]);
    await store.columns("t", "id", [col("x", "float32")]);
    expect(queries.length).toBe(1);
    await store.columns("t", "id", [col("y", "float32")]);
    expect(queries.length).toBe(2);
    expect(queries[1]).not.toContain('AS "k"');
  });
});

describe("chunkQuery", () => {
  test("predicates with several clauses are ANDed", async () => {
    let SQL = await import("@uwdata/mosaic-sql");
    let predicate = [SQL.eq(SQL.column("a"), SQL.literal(1)), SQL.eq(SQL.column("b"), SQL.literal(2))];
    let q = String(
      chunkQuery("t", SQL.column("id"), { text: SQL.column("text") }, SQL.literal(0), SQL.literal(9), predicate),
    );
    expect(q).toBe('SELECT "id" AS "k", "text" FROM "t" WHERE ("id" BETWEEN 0 AND 9) AND ("a" = 1) AND ("b" = 2)');
    let single = String(chunkQuery("t", SQL.column("id"), {}, SQL.literal(0), SQL.literal(9), predicate[0]));
    expect(single).toBe('SELECT "id" AS "k" FROM "t" WHERE ("id" BETWEEN 0 AND 9) AND ("a" = 1)');
    let none = String(chunkQuery("t", SQL.column("id"), {}, SQL.literal(0), SQL.literal(9), null));
    expect(none).toBe('SELECT "id" AS "k" FROM "t" WHERE ("id" BETWEEN 0 AND 9)');
  });
});

describe("TableStore in-flight columns", () => {
  test("retain doesn't drop columns that are being loaded", async () => {
    let { coordinator, queries } = fakeCoordinator(rows);
    let store = new TableStore(coordinator);
    let loading = store.columns("t", "id", [col("x", "float32")]);
    await Promise.resolve();
    await Promise.resolve();
    // The query is in flight; x isn't needed at this moment.
    store.retain([]);
    await loading;
    await store.columns("t", "id", [col("x", "float32")]);
    expect(queries.length).toBe(1);
  });
});

describe("filter masks", () => {
  test("a table whose keys fail reports the error once, and doesn't query masks", async () => {
    let { Selection } = await import("@uwdata/mosaic-core");
    let SQL = await import("@uwdata/mosaic-sql");
    let client: any = null;
    let queries: string[] = [];
    let coordinator: any = {
      connect: (c: any) => (client = c),
      query: async (q: any) => {
        queries.push(String(q));
        throw new Error('Binder Error: Referenced column "rowid" not found');
      },
    };
    let store = new TableStore(coordinator);
    let errors: unknown[] = [];
    let masks: unknown[] = [];
    store.filterMask("v", null, Selection.crossfilter(), {
      onMask: (m) => masks.push(m),
      onError: (e) => errors.push(e),
    });
    await client.prepare();
    let predicate = SQL.eq(SQL.column("a"), SQL.literal(1));
    let q1 = String(client.query(predicate));
    let q2 = String(client.query([predicate, predicate]));
    expect(errors.length).toBe(1);
    expect(String(errors[0])).toMatch(/identifier column/);
    expect(q1).not.toContain("FROM");
    expect(q2).not.toContain("FROM");
    expect(masks.length).toBe(0);
  });
});

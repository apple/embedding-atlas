// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { makeClient, type Coordinator, type MosaicClient, type Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { bitsetForEach, KeyIndex } from "./key_index.js";

// The data layer of EmbeddingViewMosaic.
//
// Each column of a table is loaded once (for all layers that use it), ordered by the table's key column (its
// identifier column, or DuckDB's `rowid` when there's none), so all columns of a table are aligned by position.
// Filters don't reload the columns: a filter mask is one query of the keys of the matching rows, applied to the
// columns locally. Columns and masks are immutable: a change produces new arrays, so a consumer that holds them
// (e.g., a label generation) always sees consistent data.

export type ColumnType = "float32" | "float64" | "uint8";

export type ColumnArray = Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;

/** A column to load: a SQL expression (e.g., a column, or a category index expression), cast to `type`. */
export interface ColumnSpec {
  expr: SQL.ExprNode;
  type: ColumnType;
}

/** Identifies a column spec (the cast type and the SQL). */
export function columnId(column: ColumnSpec): string {
  return `${column.type}:${String(column.expr)}`;
}

/** A filter state: the rows matching a predicate. `bits` is null when the predicate is empty (all rows match). */
export interface FilterMask {
  /** The predicate (a SQL expression), or null. */
  predicate: any;
  /** The predicate as a string ("" for no predicate). */
  predicateString: string;
  /** A bitset of the matching positions, or null if all rows match. */
  bits: Uint32Array<ArrayBuffer> | null;
  /** The number of matching rows (null if all rows match). */
  count: number | null;
}

interface PendingColumn {
  id: string;
  expr: SQL.ExprNode;
  type: ColumnType;
  resolve: (array: ColumnArray) => void;
  reject: (error: unknown) => void;
}

interface TableEntry {
  table: string;
  /** The key column (an identifier column, or "rowid"). */
  key: string;
  keyIndex: Promise<KeyIndex> | null;
  resolveKeyIndex: ((index: KeyIndex) => void) | null;
  rejectKeyIndex: ((error: unknown) => void) | null;
  columns: Map<string, Promise<ColumnArray>>;
  pending: PendingColumn[];
  /** The columns being loaded (queried, not resolved yet), by id. */
  loading: Set<string>;
  scheduled: boolean;
}

/** Loads and caches the columns of tables for EmbeddingViewMosaic, and creates filter masks. */
export class TableStore {
  private coordinator: Coordinator;
  private tables = new Map<string, TableEntry>();
  private destroyed = false;

  constructor(coordinator: Coordinator) {
    this.coordinator = coordinator;
  }

  /** Load columns of a table; the arrays are in the same order (cached; requests in the same tick are loaded together
   *  in one query). */
  columns(table: string, key: string | null, columns: ColumnSpec[]): Promise<ColumnArray[]> {
    let entry = this.entry(table, key);
    return Promise.all(columns.map((column) => this.column(entry, column)));
  }

  /** The key index of a table (loads the key column if needed). */
  keyIndex(table: string, key: string | null): Promise<KeyIndex> {
    let entry = this.entry(table, key);
    this.ensureKeyIndex(entry);
    return entry.keyIndex!;
  }

  /** The SQL expression of a key column. */
  static keyExpression(key: string): SQL.ExprNode {
    return key == "rowid" ? SQL.sql`rowid` : SQL.column(key);
  }

  /** The SQL expression of a table's key column, once its keys are loaded: the identifier column, or `rowid` if
   *  there's none or its values aren't unique (see `load`). */
  async keyExpressionOf(table: string, key: string | null): Promise<SQL.ExprNode> {
    await this.keyIndex(table, key);
    return TableStore.keyExpression(this.entry(table, key).key);
  }

  /** Create a Mosaic client that keeps a filter mask of a table up to date with a selection. */
  filterMask(
    table: string,
    key: string | null,
    selection: Selection,
    callbacks: { onMask: (mask: FilterMask) => void; onError: (error: unknown) => void },
  ): MosaicClient {
    let entry = this.entry(table, key);
    let predicateIds = new PredicateIds();
    // If the table's keys can't be loaded, there's nothing to mask (the error is reported once, in `prepare`).
    let keysFailed = false;
    // The latest predicate requested: results for older predicates are dropped (e.g., the filter was cleared
    // while a mask query was running).
    let latest: string | null = null;
    return makeClient({
      coordinator: this.coordinator,
      selection: selection,
      // Wait for the keys before the first query: the key column is known then (see `load`).
      prepare: async () => {
        try {
          await this.keyIndex(table, key);
        } catch (error) {
          keysFailed = true;
          callbacks.onError(error);
        }
      },
      query: (predicate) => {
        let predicateString = toPredicateString(predicate);
        latest = predicateString;
        let id = { [predicateIds.columnName(predicate)]: SQL.sql`NULL::BOOLEAN` };
        if (predicateString == "" || keysFailed) {
          // No filter: all rows. (A trivial query, since Mosaic needs one; its result is cached.)
          return SQL.Query.select(id);
        }
        return SQL.Query.from(table)
          .select({ k: TableStore.keyExpression(entry.key), ...id })
          .where(predicate);
      },
      queryResult: async (data: any) => {
        if (keysFailed) {
          return;
        }
        let predicate = predicateIds.predicateOfResult(data);
        if (predicate === undefined) {
          return;
        }
        let predicateString = toPredicateString(predicate);
        if (predicateString == "") {
          if (predicateString == latest && !this.destroyed) {
            callbacks.onMask({ predicate: null, predicateString: "", bits: null, count: null });
          }
          return;
        }
        let keys = data.getChild("k").toArray();
        let index = await this.keyIndex(table, key);
        if (predicateString != latest || this.destroyed) {
          return;
        }
        let { bits, count } = index.bitsetOf(keys);
        callbacks.onMask({ predicate, predicateString, bits, count });
      },
      queryError: callbacks.onError,
    });
  }

  /** Forget loaded columns that aren't needed anymore (so their arrays can be freed; they're loaded again if
   *  they're requested later). The keys of tables are kept. */
  retain(needed: { table: string; key: string | null; columns: ColumnSpec[] }[]) {
    let keep = new Map<TableEntry, Set<string>>();
    for (let n of needed) {
      let entry = this.entry(n.table, n.key);
      let ids = keep.get(entry) ?? new Set<string>();
      for (let column of n.columns) {
        ids.add(columnId(column));
      }
      keep.set(entry, ids);
    }
    for (let entry of this.tables.values()) {
      let ids = keep.get(entry);
      for (let id of [...entry.columns.keys()]) {
        if (!ids?.has(id) && !entry.loading.has(id) && !entry.pending.some((c) => c.id == id)) {
          entry.columns.delete(id);
        }
      }
    }
  }

  destroy() {
    this.destroyed = true;
    this.tables.clear();
  }

  private entry(table: string, key: string | null): TableEntry {
    let k = key ?? "rowid";
    let id = JSON.stringify([table, k]);
    let entry = this.tables.get(id);
    if (entry == null) {
      entry = {
        table,
        key: k,
        keyIndex: null,
        resolveKeyIndex: null,
        rejectKeyIndex: null,
        columns: new Map(),
        pending: [],
        loading: new Set(),
        scheduled: false,
      };
      this.tables.set(id, entry);
    }
    return entry;
  }

  private ensureKeyIndex(entry: TableEntry) {
    if (entry.keyIndex == null) {
      entry.keyIndex = new Promise((resolve, reject) => {
        entry.resolveKeyIndex = resolve;
        entry.rejectKeyIndex = reject;
      });
      // The error is reported to the callers of `columns` / `keyIndex`; nothing else may be waiting for the keys.
      entry.keyIndex.catch(() => {});
      this.schedule(entry);
    }
  }

  private column(entry: TableEntry, column: ColumnSpec): Promise<ColumnArray> {
    let id = columnId(column);
    let existing = entry.columns.get(id);
    if (existing != null) {
      return existing;
    }
    let promise = new Promise<ColumnArray>((resolve, reject) => {
      entry.pending.push({ id, expr: column.expr, type: column.type, resolve, reject });
    });
    entry.columns.set(id, promise);
    // The first query of a table also loads its keys.
    this.ensureKeyIndex(entry);
    this.schedule(entry);
    return promise;
  }

  private schedule(entry: TableEntry) {
    if (entry.scheduled) {
      return;
    }
    entry.scheduled = true;
    queueMicrotask(() => {
      entry.scheduled = false;
      this.flush(entry);
    });
  }

  private async flush(entry: TableEntry) {
    let pending = entry.pending;
    entry.pending = [];
    let ids = pending.map((c) => c.id);
    ids.forEach((id) => entry.loading.add(id));
    let key = entry.resolveKeyIndex != null ? { resolve: entry.resolveKeyIndex, reject: entry.rejectKeyIndex! } : null;
    entry.resolveKeyIndex = null;
    entry.rejectKeyIndex = null;
    if (pending.length == 0 && key == null) {
      return;
    }
    try {
      await this.load(entry, pending, key);
    } finally {
      ids.forEach((id) => entry.loading.delete(id));
    }
  }

  /** Load columns (and the keys, if `key` is given) in one query. If it fails, the keys and the columns are loaded
   *  separately, so that one bad column (e.g., a layer with a column that doesn't exist) doesn't fail the others.
   *  Returns the error if the query failed. */
  private async load(
    entry: TableEntry,
    pending: PendingColumn[],
    key: { resolve: (index: KeyIndex) => void; reject: (error: unknown) => void } | null,
  ): Promise<unknown> {
    // The keys loaded by an earlier query, if they're not loaded by this one.
    let keyIndex = entry.keyIndex;
    let keyColumn = entry.key;
    let keyExpr = TableStore.keyExpression(keyColumn);
    let select: Record<string, any> = {};
    if (key != null) {
      select["k"] = keyExpr;
    }
    pending.forEach((c, i) => {
      let cast = c.type == "float32" ? "FLOAT" : c.type == "float64" ? "DOUBLE" : "UTINYINT";
      select[`c${i}`] = SQL.cast(c.expr, cast);
    });
    let result: any;
    try {
      result = await this.coordinator.query(SQL.Query.from(entry.table).select(select).orderby(keyExpr));
    } catch (error) {
      let reported = describeError(error, entry.table, keyColumn);
      if (pending.length + (key != null ? 1 : 0) > 1) {
        // Isolate the failure: the keys first (columns can't be aligned without them), then each column.
        let keyError = key != null ? await this.load(entry, [], key) : null;
        if (keyError != null) {
          this.rejectColumns(entry, pending, keyError);
          return keyError;
        }
        await Promise.all(pending.map((c) => this.load(entry, [c], null)));
        return null;
      }
      this.rejectColumns(entry, pending, reported);
      if (key != null) {
        entry.keyIndex = null;
        key.reject(reported);
      }
      return reported;
    }

    let index: KeyIndex;
    if (key != null) {
      index = new KeyIndex(result.getChild("k").toArray());
      if (!index.unique && keyColumn != "rowid") {
        // Rows must be identified by their keys (to align columns and apply masks): use rowid instead.
        console.warn(
          `Embedding view: the identifier column "${keyColumn}" of "${entry.table}" isn't unique (or has nulls), ` +
            `using rowid instead.`,
        );
        entry.key = "rowid";
        return await this.load(entry, pending, key);
      }
    } else {
      try {
        index = await keyIndex!;
      } catch (error) {
        this.rejectColumns(entry, pending, error);
        return error;
      }
    }
    pending.forEach((c, i) => {
      let array = toTypedArray(result.getChild(`c${i}`).toArray(), c.type);
      if (array.length != index.length) {
        c.reject(new Error(`Column ${String(c.expr)} of "${entry.table}" doesn't have one value per row.`));
      } else {
        c.resolve(array);
      }
    });
    key?.resolve(index);
    return null;
  }

  /** Reject columns (and forget them, so they can be loaded again later). */
  private rejectColumns(entry: TableEntry, pending: PendingColumn[], error: unknown) {
    for (let c of pending) {
      entry.columns.delete(c.id);
      c.reject(error);
    }
  }
}

/** The query of a chunk of a table's rows (keys from `low` to `high`, in key order) that match a predicate: the key
 *  (as `k`) and the given columns. The predicate may be a single expression or a list of clauses (ANDed). */
export function chunkQuery(
  table: string,
  keyExpr: SQL.ExprNode,
  columns: Record<string, SQL.ExprNode>,
  low: SQL.ExprNode,
  high: SQL.ExprNode,
  predicate: any,
): SQL.Query {
  // Separate arguments: `where` flattens one level of lists, and the predicate may be a list of clauses.
  return SQL.Query.from(table)
    .select({ k: keyExpr, ...columns })
    .where(SQL.isBetween(keyExpr, [low, high]), predicate ?? null);
}

/** A clearer error for tables without rowid (e.g., views), which need an identifier column. */
function describeError(error: unknown, table: string, keyColumn: string): unknown {
  if (keyColumn == "rowid" && String(error).toLowerCase().includes("rowid")) {
    return new Error(
      `Table "${table}" has no rowid (e.g., it's a view): specify an identifier column (a unique row id) for it.`,
      { cause: error },
    );
  }
  return error;
}

function toTypedArray(values: any, type: ColumnType): ColumnArray {
  switch (type) {
    case "float32":
      return values instanceof Float32Array ? (values as Float32Array<ArrayBuffer>) : new Float32Array(values);
    case "float64":
      return values instanceof Float64Array ? (values as Float64Array<ArrayBuffer>) : new Float64Array(values);
    case "uint8":
      return values instanceof Uint8Array ? (values as Uint8Array<ArrayBuffer>) : new Uint8Array(values);
  }
}

/** The rows of arrays at the positions set in a mask (the arrays themselves if all rows match). */
export function applyMask<T extends ColumnArray>(array: T, mask: FilterMask | null): T {
  if (mask == null || mask.bits == null) {
    return array;
  }
  let result = new (array.constructor as any)(mask.count!) as T;
  let j = 0;
  bitsetForEach(mask.bits, (p) => {
    result[j++] = array[p];
  });
  return result;
}

/** Positions set in a mask, or null if all rows match. */
export function maskPositions(mask: FilterMask | null): Uint32Array | null {
  if (mask == null || mask.bits == null) {
    return null;
  }
  let result = new Uint32Array(mask.count!);
  let j = 0;
  bitsetForEach(mask.bits, (p) => {
    result[j++] = p;
  });
  return result;
}

/** Memoizes masked arrays by (array, mask), so layers sharing data and a filter share the masked arrays. */
export class MaskedArrays {
  private cache = new WeakMap<ColumnArray, WeakMap<FilterMask, ColumnArray>>();

  get<T extends ColumnArray>(array: T, mask: FilterMask | null): T {
    if (mask == null || mask.bits == null) {
      return array;
    }
    let byMask = this.cache.get(array);
    if (byMask == null) {
      byMask = new WeakMap();
      this.cache.set(array, byMask);
    }
    let result = byMask.get(mask);
    if (result == null) {
      result = applyMask(array, mask);
      byMask.set(mask, result);
    }
    return result as T;
  }
}

export function toPredicateString(predicate: any): string {
  if (predicate == null) {
    return "";
  }
  if (Array.isArray(predicate)) {
    return predicate.map((x) => String(x)).join(" AND ");
  }
  return String(predicate);
}

/** Assigns each distinct filter predicate an id, which the query embeds in the name of a column, so that each
 *  result can be paired with the predicate it was queried with. The query stays a pure function of the predicate
 *  (a recent predicate gives the same SQL, so Mosaic's query cache still works), and the id is in the result's
 *  schema even if the result has no rows. */
class PredicateIds {
  /** The number of recent predicates to remember. */
  static readonly capacity = 64;
  static readonly prefix = "_predicate_";

  private ids = new Map<string, number>();
  private predicates = new Map<number, any>();
  private nextId = 1;

  /** The name of the column that identifies the predicate. */
  columnName(predicate: any): string {
    let key = toPredicateString(predicate);
    let id = this.ids.get(key) ?? this.nextId++;
    // Re-insert to mark the predicate as most recently used.
    this.ids.delete(key);
    this.ids.set(key, id);
    this.predicates.delete(id);
    this.predicates.set(id, predicate ?? null);
    while (this.ids.size > PredicateIds.capacity) {
      let [oldKey, oldId] = this.ids.entries().next().value!;
      this.ids.delete(oldKey);
      this.predicates.delete(oldId);
    }
    return PredicateIds.prefix + id;
  }

  /** The predicate of a query result, or undefined if it's unknown. */
  predicateOfResult(data: any): any {
    for (let field of data.schema.fields) {
      if (field.name.startsWith(PredicateIds.prefix)) {
        return this.predicates.get(parseInt(field.name.slice(PredicateIds.prefix.length)));
      }
    }
    return undefined;
  }
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { connectWorker, type WorkerProxy } from "@embedding-atlas/utils";
import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import type { Searcher } from "../api.js";
import type { SearchIndex } from "./search.worker.js";

async function createSearchIndex(): Promise<WorkerProxy<SearchIndex>> {
  let worker = new Worker(new URL("./search.worker.js", import.meta.url), { type: "module" });
  let conn = await connectWorker(worker);
  return conn.create<SearchIndex>("SearchIndex");
}

export class FullTextSearcher implements Searcher {
  coordinator: Coordinator;
  table: string;
  columns: { text: string; id: string };

  backend: Promise<WorkerProxy<SearchIndex>>;
  currentIndex: { predicate: string | null; promise: Promise<void> } | null = null;

  constructor(
    coordinator: Coordinator,
    table: string,
    columns: {
      text: string;
      id: string;
    },
  ) {
    this.coordinator = coordinator;
    this.table = table;
    this.columns = columns;
    this.currentIndex = null;
    this.backend = createSearchIndex();
  }

  predicateString(predicate: any | null): string | null {
    if (predicate != null && predicate.toString() != "") {
      return predicate.toString();
    } else {
      return null;
    }
  }

  buildIndexIfNeeded(predicate: any | null): Promise<void> {
    let builder = async () => {
      let result: any;
      if (predicateString != null) {
        result = await this.coordinator.query(`
        SELECT
          ${SQL.column(this.columns.id)} AS id,
          ${SQL.column(this.columns.text)} AS text
        FROM ${this.table}
        WHERE ${predicateString}
      `);
      } else {
        result = await this.coordinator.query(`
        SELECT
          ${SQL.column(this.columns.id)} AS id,
          ${SQL.column(this.columns.text)} AS text
        FROM ${this.table}
      `);
      }
      let backend = await this.backend;
      await backend.clear();
      await backend.addPoints(Array.from(result));
    };

    let predicateString = this.predicateString(predicate);
    if (this.currentIndex != null) {
      if (this.currentIndex.predicate != predicateString) {
        let promise = this.currentIndex.promise.then(() => builder());
        this.currentIndex = { predicate: predicateString, promise: promise };
      }
    } else {
      let promise = builder();
      this.currentIndex = { predicate: predicateString, promise: promise };
    }
    return this.currentIndex.promise;
  }

  async fullTextSearch(
    query: string,
    options: { limit?: number; predicate?: any; onStatus?: (status: string) => void } = {},
  ): Promise<{ id: any }[]> {
    let limit = options.limit ?? 100;
    let predicate = options.predicate;
    options?.onStatus?.("Indexing...");
    await this.buildIndexIfNeeded(predicate);
    options?.onStatus?.("Searching...");
    let backend = await this.backend;
    let resultIDs = await backend.query(query, limit);
    return resultIDs.map((id) => ({ id: id }));
  }
}

export interface SearchResultItem {
  id: any;
  fields: Record<string, any>;
  distance?: number;
  x?: number;
  y?: number;
  text?: string;
}

export async function querySearchResultItems(
  coordinator: Coordinator,
  table: string,
  idColumn: string,
  additionalFields: Record<string, any> | null,
  predicate: string | null,
  items: { id: any; distance?: number }[],
): Promise<SearchResultItem[]> {
  let fieldExpressions: string[] = [`${SQL.column(idColumn, table)} AS id`];

  let fields = additionalFields ?? {};
  for (let key in fields) {
    let spec = fields[key];
    if (typeof spec == "string") {
      fieldExpressions.push(`${SQL.column(spec, table)} AS "field_${key}"`);
    } else {
      fieldExpressions.push(`${SQL.sql(spec.sql)} AS "field_${key}"`);
    }
  }

  let ids = items.map((x) => x.id);
  let id2order = new Map<any, number>();
  let id2item = new Map<any, { id: any; distance?: number }>();
  for (let i = 0; i < ids.length; i++) {
    id2order.set(ids[i], i);
    id2item.set(ids[i], items[i]);
  }
  let r = await coordinator.query(`
    SELECT
      ${fieldExpressions.join(", ")}
    FROM (
      SELECT ${SQL.column(idColumn, table)} AS __search_result_id__
      FROM ${table}
      WHERE
        ${SQL.column(idColumn, table)} IN [${ids.map((x) => SQL.literal(x)).join(", ")}]
        ${predicate ? `AND (${predicate})` : ``}
    )
    LEFT JOIN ${table} ON ${SQL.column(idColumn, table)} = __search_result_id__
  `);

  let result = Array.from(r).map((x: any): any => {
    let r: Record<string, any> = { id: x.id, distance: id2item.get(x.id)?.distance, fields: {} };
    for (let key in x) {
      if (key.startsWith("field_")) {
        r.fields[key.substring(6)] = x[key];
      } else {
        r[key] = x[key];
      }
    }
    return r;
  });
  result = result.sort((a, b) => (id2order.get(a.id) ?? 0) - (id2order.get(b.id) ?? 0));
  return result;
}

export function resolveSearcher(options: {
  coordinator: Coordinator;
  table: string;
  searcher?: Searcher | null;
  idColumn: string;
  textColumn?: string | null;
  neighborsColumn?: string | null;
}): Searcher {
  let { coordinator, table, idColumn, searcher, textColumn, neighborsColumn } = options;

  if (searcher === null) {
    return {};
  }

  let result: Searcher = {};

  if (searcher?.fullTextSearch != null) {
    result.fullTextSearch = searcher.fullTextSearch.bind(searcher);
  } else if (textColumn != null) {
    // FullTextSearcher on the text column.
    let fts = new FullTextSearcher(coordinator, table, { id: idColumn, text: textColumn });
    result.fullTextSearch = fts.fullTextSearch.bind(fts);
  }

  if (searcher?.vectorSearch != null) {
    result.vectorSearch = searcher.vectorSearch.bind(searcher);
  }

  return result;
}

export async function performSearch({
  searcher,
  predicate,
  query,
  mode,
  limit,
  onStatus,
}: {
  searcher: Searcher;
  predicate: string | null;
  query: any;
  mode: string;
  limit: number;
  onStatus: (status: string) => void;
}): Promise<{ id: any; distance?: number }[]> {
  onStatus("Searching...");
  if (mode == "full-text" && searcher.fullTextSearch != null) {
    query = query.trim();
    return await searcher.fullTextSearch(query, { limit: limit, predicate: predicate, onStatus: onStatus });
  } else if (mode == "vector" && searcher.vectorSearch != null) {
    query = query.trim();
    return await searcher.vectorSearch(query, { limit: limit, predicate: predicate, onStatus: onStatus });
  } else if (mode == "raw") {
    return query.items;
  } else {
    return [];
  }
}

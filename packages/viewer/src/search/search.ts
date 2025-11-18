// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import type { Searcher } from "../api.js";

class SearchWorkerAPI {
  worker: Worker;
  callbacks: Map<string, (data: any) => void>;

  constructor() {
    this.worker = new Worker(new URL("./search.worker.js", import.meta.url), { type: "module" });
    this.callbacks = new Map();
    this.worker.onmessage = (e) => {
      let cb = this.callbacks.get(e.data.identifier);
      if (cb != null) {
        this.callbacks.delete(e.data.identifier);
        cb(e.data);
      }
    };
  }

  rpc(message: any): Promise<any> {
    return new Promise((resolve, _) => {
      let identifier = new Date().getTime() + "-" + Math.random();
      this.callbacks.set(identifier, resolve);
      this.worker.postMessage({ ...message, identifier: identifier });
    });
  }

  async clear() {
    await this.rpc({ type: "clear" });
  }

  async addPoints(points: { id: string | number; text: string }[]) {
    await this.rpc({ type: "points", points: points });
  }

  async query(query: string, limit: number): Promise<string[]> {
    let data = await this.rpc({ type: "query", query: query, limit: limit });
    return data.result;
  }
}

export class FullTextSearcher implements Searcher {
  coordinator: Coordinator;
  table: string;
  columns: { text: string; id: string };

  backend: SearchWorkerAPI;
  currentIndex: { predicate: string | null } | null = null;

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
    this.backend = new SearchWorkerAPI();
  }

  predicateString(predicate: any | null): string | null {
    if (predicate != null && predicate.toString() != "") {
      return predicate.toString();
    } else {
      return null;
    }
  }

  async buildIndexIfNeeded(predicate: any | null) {
    let predicateString = this.predicateString(predicate);
    if (this.currentIndex != null && this.currentIndex.predicate == predicateString) {
      return;
    }
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
    await this.backend.clear();
    await this.backend.addPoints(Array.from(result));
    this.currentIndex = { predicate: predicateString };
  }

  async fullTextSearch(query: string, options: { limit?: number; predicate?: any } = {}): Promise<{ id: any }[]> {
    let limit = options.limit ?? 100;
    let predicate = options.predicate;
    await this.buildIndexIfNeeded(predicate);
    let resultIDs = await this.backend.query(query, limit);
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
  columns: { id: string; x?: string | null; y?: string | null; text?: string | null },
  additionalFields: Record<string, any> | null,
  predicate: string | null,
  items: { id: any; distance?: number }[],
): Promise<SearchResultItem[]> {
  let fieldExpressions: string[] = [`${SQL.column(columns.id, table)} AS id`];
  if (columns.x) {
    fieldExpressions.push(`${SQL.column(columns.x, table)} AS x`);
  }
  if (columns.y) {
    fieldExpressions.push(`${SQL.column(columns.y, table)} AS y`);
  }
  if (columns.text) {
    fieldExpressions.push(`${SQL.column(columns.text, table)} AS text`);
  }
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
      SELECT ${SQL.column(columns.id, table)} AS __search_result_id__
      FROM ${table}
      WHERE
        ${SQL.column(columns.id, table)} IN [${ids.map((x) => SQL.literal(x)).join(", ")}]
        ${predicate ? `AND (${predicate})` : ``}
    )
    LEFT JOIN ${table} ON ${SQL.column(columns.id, table)} = __search_result_id__
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

  if (searcher?.nearestNeighbors != null) {
    result.nearestNeighbors = searcher.nearestNeighbors.bind(searcher);
  } else if (neighborsColumn != null) {
    // Search with pre-computed nearest neighbors.
    result.nearestNeighbors = async (id: any): Promise<{ id: any; distance: number }[]> => {
      let q = SQL.Query.from(table)
        .select({ knn: SQL.column(neighborsColumn) })
        .where(SQL.eq(SQL.column(idColumn), SQL.literal(id)));
      let result = await coordinator.query(q);
      let items: any[] = Array.from(result);
      if (items.length != 1) {
        return [];
      }
      let { distances, ids } = items[0].knn;
      let r = Array.from(ids)
        .map((nid, i) => {
          return { id: nid, distance: distances[i] };
        })
        .filter((x) => x.id != id);
      return r;
    };
  }

  return result;
}

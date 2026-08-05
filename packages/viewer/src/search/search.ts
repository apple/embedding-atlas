// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { connectWorker, type WorkerProxy } from "@embedding-atlas/utils";
import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import type { Searcher } from "../api.js";
import { escapeLikePattern, parseQuery } from "./query_parser.js";
import type { SearchIndex } from "./search.worker.js";

/** A bound large enough to stand in for "every match" in a flexsearch query. */
const UNLIMITED_SEARCH_RESULTS = 1e9;

async function createSearchIndex(): Promise<WorkerProxy<SearchIndex>> {
  let worker = new Worker(new URL("./search.worker.js", import.meta.url), { type: "module" });
  let conn = await connectWorker(worker);
  return conn.create<SearchIndex>("SearchIndex");
}

export class FullTextSearcher implements Searcher {
  coordinator: Coordinator;
  table: string;
  columns: { text: string; id: string };

  // Created on first use. A query made entirely of exact phrases is answered by
  // the database, so it never needs the fuzzy index or the worker hosting it.
  private backendPromise: Promise<WorkerProxy<SearchIndex>> | null = null;
  currentIndex: { predicate: string | null; promise: Promise<void> } | null = null;

  get backend(): Promise<WorkerProxy<SearchIndex>> {
    this.backendPromise ??= createSearchIndex();
    return this.backendPromise;
  }

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

  /**
   * Run the exact-phrase part of a query against the database.
   *
   * Every phrase must appear as a case-insensitive substring of the text
   * column. This runs as SQL rather than in the worker so the phrases are
   * matched against the text already in the database, instead of keeping a
   * second copy of every text in memory alongside the fuzzy index.
   *
   * When `candidateIDs` is given, the match is restricted to those rows, which
   * is how a mixed query like `"aldi" store` narrows the fuzzy hits for `store`
   * down to the ones that also contain the exact phrase.
   */
  private async queryPhrases(
    phrases: string[],
    predicate: string | null,
    candidateIDs: (string | number)[] | null,
    limit: number,
  ): Promise<any[]> {
    let idColumn = SQL.column(this.columns.id);
    let textColumn = SQL.column(this.columns.text);

    let conditions = phrases.map(
      (phrase) =>
        `lower(${textColumn}) LIKE ${SQL.literal(`%${escapeLikePattern(phrase.toLowerCase())}%`)} ESCAPE '\\'`,
    );
    if (predicate != null) {
      conditions.push(`(${predicate})`);
    }
    if (candidateIDs != null) {
      if (candidateIDs.length == 0) {
        return [];
      }
      conditions.push(`${idColumn} IN [${candidateIDs.map((x) => SQL.literal(x)).join(", ")}]`);
    }

    let result = await this.coordinator.query(`
      SELECT ${idColumn} AS id
      FROM ${this.table}
      WHERE ${conditions.join(" AND ")}
      LIMIT ${limit}
    `);
    return Array.from(result).map((row: any) => row.id);
  }

  async fullTextSearch(
    query: string,
    options: { limit?: number; predicate?: any; onStatus?: (status: string) => void } = {},
  ): Promise<{ id: any }[]> {
    let limit = options.limit ?? 100;
    let predicate = options.predicate;
    let { phrases, freeText } = parseQuery(query);

    // Phrases only: the database can answer this on its own, so skip building
    // the fuzzy index entirely.
    if (phrases.length > 0 && freeText.length == 0) {
      options?.onStatus?.("Searching...");
      let resultIDs = await this.queryPhrases(phrases, this.predicateString(predicate), null, limit);
      return resultIDs.map((id) => ({ id: id }));
    }

    options?.onStatus?.("Indexing...");
    await this.buildIndexIfNeeded(predicate);
    options?.onStatus?.("Searching...");
    let backend = await this.backend;

    // No phrases: the original fuzzy path, unchanged.
    if (phrases.length == 0) {
      let resultIDs = await backend.query(freeText, limit);
      return resultIDs.map((id) => ({ id: id }));
    }

    // Mixed query: rank by the fuzzy hits for the free text, then keep only the
    // ones that also contain every phrase. The candidate search is uncapped so
    // the phrase filter does not run against an already-truncated list. Note
    // that flexsearch treats a limit of 0 as "use the default of 100" rather
    // than "no limit", so this passes an explicit large bound instead.
    let candidateIDs = await backend.query(freeText, UNLIMITED_SEARCH_RESULTS);
    let matched = new Set(
      await this.queryPhrases(phrases, this.predicateString(predicate), candidateIDs, candidateIDs.length),
    );
    let resultIDs = candidateIDs.filter((id) => matched.has(id)).slice(0, limit);
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

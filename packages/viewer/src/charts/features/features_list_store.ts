// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import erfinv from "@stdlib/math/base/special/erfinv";
import { type Coordinator, makeClient, type MosaicClient, type SelectionClause } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import { get, type Readable, type Writable } from "svelte/store";

import { jsTypeFromDBType } from "../../utils/database.js";
import { deepStableDerived, stableDerived, stableWritable } from "../../utils/store.js";
import type { ChartContext } from "../chart.js";
import type { SQLField } from "../spec/spec.js";
import { inferBinaryPolarity } from "./binary_polarity.js";
import { buildPredictBinary, buildPredictMulticlass, type Predict } from "./prediction.js";
import type { FeaturesListSpec, FeaturesListState } from "./types.js";
import { fieldExpr, fromExpr } from "./utils.js";

/**
 * Kind of the `features` column: a plain list of feature strings ("list") or a
 * list of structs carrying per-feature fields like source/index ("struct").
 * Detected at query time from the column's SQL type.
 */
type FeaturesKind = "list" | "struct";

export interface SourceSegment {
  source: string;
  count: number;
  countSelected: number;
}

export interface ListItem {
  group: string;
  feature: string;
  count: number;
  countSelected: number;
  predict?: Predict;
  sourceSegments?: SourceSegment[];
  /**
   * Source-skew scores, computed over the cross-filtered (selected) occurrences
   * and both frequency-weighted (so rare features don't dominate). They split
   * the feature's occurrences by how they spread across sources: `sourceConc` is
   * the evidence the feature is concentrated in few sources (ranks "↓ Source
   * skew"); `sourceBal` is the evidence it is balanced across sources (ranks "↑
   * Source skew"). `sourceShape` is the scale-free concentration in `[0, 1]`
   * (0 = equal per source, 1 = single source), for display. All undefined when
   * there are fewer than two sources or nothing is selected.
   */
  sourceConc?: number;
  sourceBal?: number;
  sourceShape?: number;
}

export interface Data {
  /**
   * Every feature in the aggregation, fully computed (counts, predict,
   * source-skew) but NOT searched, sorted, limited, or topic-grouped. The
   * hosting component applies search/sort/limit/grouping and pinning so it can
   * keep pinned features visible outside that pipeline.
   */
  allItems: ListItem[];
  /**
   * All class names from the predict column. Empty in count-only mode.
   * When `hasClassPolarity` is true, ordered as [negative, positive];
   * otherwise sorted alphabetically.
   */
  classNames: string[];
  /**
   * True iff `classNames` has exactly two entries with a confident
   * negative/positive ordering inferred from the predict column name and
   * the class label strings. The UI can use this flag to pick a divergent
   * (e.g. red/blue) palette instead of a categorical one.
   */
  hasClassPolarity: boolean;
  /** Total feature count in the aggregation (= `allItems.length`). */
  totalCount: number;
  /** True if the cross-filter selection is narrower than the full data. */
  hasSelection: boolean;
  /** Number of distinct sources; used by the sort comparator (skew sorts need ≥2). */
  sourceCount: number;
}

interface AggFeature {
  feature: string;
  // Per-source counts, summed over klass when available. Always populated.
  count: Map<string, number>; // source → count
  countSelected: Map<string, number>; // source → count under cross-filter
  // Per-(klass, source) counts. Populated only in predict mode.
  classCount?: Map<string, Map<string, number>>; // klass → source → count
  classCountSelected?: Map<string, Map<string, number>>; // klass → source → count under cross-filter
}

interface AggGlobals {
  classTotals: Map<string, number>;
  classTotalsSelected: Map<string, number>;
  total: number;
  totalSelected: number;
}

interface AggregationResult {
  features: AggFeature[];
  globals?: AggGlobals;
}

interface ClientSignature {
  features: SQLField;
  featuresKind: FeaturesKind;
  predict: SQLField | undefined;
  sources: string[] | undefined;
}

export class FeaturesListStore {
  readonly context: ChartContext;
  readonly spec: Writable<FeaturesListSpec>;
  readonly state: Writable<FeaturesListState>;

  readonly topicsByFeature: Writable<Map<string, string[]> | null>;
  readonly allSources: Writable<string[] | null>;
  readonly aggregation: Writable<AggregationResult | null>;

  readonly data: Readable<Data>;

  private clients: MosaicClient[] = [];
  private unsubs: Array<() => void> = [];
  private sourcesLoadId = 0;
  private mainLoadId = 0;

  /** Last-resolved kind of the `features` column; read synchronously by syncSelection. */
  private featuresKind: FeaturesKind = "list";
  /** True once the kind has been resolved at least once; gates syncSelection (see below). */
  private featuresKindResolved = false;

  /** Clears `state.selected` when the global "reset filters" action fires. */
  private readonly onClearSelection: () => void;
  /** Stable identity for our cross-filter selection clause (so updates replace, not stack). */
  private readonly selectionSource = { reset: () => this.onClearSelection() };
  /** Bumped whenever the aggregation clients are recreated, to refresh the clause's `clients` set. */
  private clientsToken = 0;
  /** Signature of the last published selection clause, to skip redundant `filter.update` calls. */
  private lastSelectionKey: string | null = null;

  constructor(
    context: ChartContext,
    spec: FeaturesListSpec,
    state: FeaturesListState,
    onClearSelection: () => void = () => {},
  ) {
    this.context = context;
    this.onClearSelection = onClearSelection;
    this.spec = stableWritable(spec);
    this.state = stableWritable(state);
    this.topicsByFeature = stableWritable<Map<string, string[]> | null>(null);
    this.allSources = stableWritable<string[] | null>(null);
    this.aggregation = stableWritable<AggregationResult | null>(null);

    // `data` (allItems + scalars) depends only on the aggregation and spec.
    // Search/sort/limit/grouping live in the component, so neither `state`
    // (search) nor `topicsByFeature` is an input here.
    this.data = stableDerived([this.aggregation, this.spec], (args) => buildData(...args));

    this.bindLifecycle();
  }

  update(spec: FeaturesListSpec, state: FeaturesListState) {
    this.spec.set(spec);
    this.state.set(state);
    // Republish the selection clause: `state.selected` and/or the features
    // field may have changed. (Aggregation-client recreation triggered by the
    // spec change above also republishes, so the clause's `clients` stays fresh.)
    this.syncSelection();
  }

  destroy() {
    // Bump tokens so any in-flight loads that resolve after this point
    // skip writing into the (orphaned) writables.
    this.sourcesLoadId++;
    this.mainLoadId++;
    // Remove our selection clause from the global cross-filter.
    this.context.filter.update({
      source: this.selectionSource,
      clients: new Set<MosaicClient>(this.clients),
      value: null,
      predicate: null,
    });
    this.lastSelectionKey = null;
    for (const c of this.clients) {
      c.destroy();
    }
    this.clients = [];
    for (const u of this.unsubs) {
      u();
    }
    this.unsubs = [];
  }

  /**
   * Publish (or update) the feature-selection clause into the global
   * cross-filter. Selected features combine with AND: a data row matches iff
   * its features column contains every selected feature. The clause's `clients`
   * set is our own aggregation client(s), so the features list excludes its own
   * selection — its counts/bars stay stable while other views are filtered
   * (mirrors CountPlot's self-exclusion).
   */
  private syncSelection() {
    const spec = get(this.spec);
    const selected = get(this.state).selected ?? [];

    // The selection predicate depends on the column kind (string-list vs
    // struct-list). Until the kind is resolved we can't build a correct
    // `list_contains` (a struct list would raise a type error), so defer any
    // non-empty selection. We intentionally don't update `lastSelectionKey`
    // here, so the authoritative call from createAggregationClients (which runs
    // after the kind resolves) isn't deduped away.
    if (selected.length > 0 && !this.featuresKindResolved) {
      return;
    }

    // Skip when nothing relevant changed (e.g. typing in the search box still
    // calls update()), so we don't trigger redundant cross-filter requeries.
    const key = `${this.clientsToken}|${JSON.stringify(spec.data.features)}|${this.featuresKind}|${JSON.stringify(selected)}`;
    if (key === this.lastSelectionKey) {
      return;
    }
    this.lastSelectionKey = key;

    let predicate: SQL.ExprNode | null = null;
    let value: string[] | null = null;
    if (selected.length > 0) {
      // Plain list membership on the feature names; intentionally NOT
      // source-filtered (respecting the source filter would require aligning
      // each feature with its parallel source index). For struct-list columns
      // the names are projected out of the struct first.
      const namesExpr = featureValuesExpr(spec.data.features, this.featuresKind, { table: this.context.table });
      predicate = SQL.and(...selected.map((f) => SQL.listContains(namesExpr, SQL.literal(f))));
      value = selected;
    }

    const clause: SelectionClause = {
      source: this.selectionSource,
      clients: new Set<MosaicClient>(this.clients),
      value,
      predicate,
    };
    this.context.filter.update(clause);
  }

  /**
   * Subscribe to the two signature derivations that drive all data loading.
   *
   * - reloadSources only depends on `{features}`. It does NOT include
   *   `spec.sources`, so changing the source filter (e.g. from the Source
   *   dropdown) leaves `allSources` populated and the dropdown intact.
   * - reloadMain covers the rest of the data-shape inputs (data, metadata,
   *   sources). When it changes we tear down the aggregation clients,
   *   reload topics, and recreate the clients.
   *
   * Both resolve the column kind via reloadSources/reloadMain → resolveFeaturesKind;
   * each fires its own DESCRIBE (cheap, so no dedup).
   */
  private bindLifecycle() {
    this.unsubs.push(
      deepStableDerived([this.spec], ([sp]) => ({
        features: sp.data.features,
      })).subscribe((sig) => this.reloadSources(sig)),
    );
    this.unsubs.push(
      deepStableDerived([this.spec], ([sp]) => ({
        data: sp.data,
        metadata: sp.metadata,
        sources: sp.sources,
      })).subscribe((sig) => this.reloadMain(sig)),
    );
  }

  private async reloadSources(sig: { features: SQLField }) {
    const id = ++this.sourcesLoadId;
    const kind = await this.resolveFeaturesKind(sig.features);
    if (id !== this.sourcesLoadId) {
      return;
    }
    // Commit the resolved kind only after the staleness guard (see resolveFeaturesKind).
    this.featuresKind = kind;
    this.featuresKindResolved = true;
    if (kind === "list") {
      // String-list columns carry no per-feature source field; everything is "default".
      this.allSources.set(["default"]);
      return;
    }
    let sources: string[] = [];
    try {
      const sourceExpr = sourceValuesExpr(sig.features, kind, { table: this.context.table });
      const query = SQL.Query.from(this.context.table)
        .select({
          source: SQL.sql`UNNEST(${sourceExpr})::TEXT`,
        })
        .distinct();
      const result: any = await this.context.coordinator.query(query);
      sources = (Array.from(result) as any[])
        .map((r) => (r.source != null ? String(r.source) : null))
        .filter((x): x is string => x != null && x.length > 0)
        .sort();
    } catch (err) {
      console.error("[FeaturesList] failed to load sources:", err);
    } finally {
      if (id === this.sourcesLoadId) {
        this.allSources.set(sources);
      }
    }
  }

  private async reloadMain(sig: {
    data: FeaturesListSpec["data"];
    metadata: FeaturesListSpec["metadata"];
    sources: string[] | undefined;
  }) {
    const id = ++this.mainLoadId;

    // Tear down old aggregation clients and reset the writables they feed.
    for (const c of this.clients) {
      c.destroy();
    }
    this.clients = [];
    this.aggregation.set(null);
    this.topicsByFeature.set(null);

    // Topics: async, runs in parallel with aggregation client setup.
    if (sig.metadata?.topics != null) {
      this.loadTopics(sig.metadata, id);
    }

    // Resolve the column kind before building the (synchronous) aggregation
    // query. Re-check the load id after the await in case the spec changed
    // while detecting.
    const featuresKind = await this.resolveFeaturesKind(sig.data.features);
    if (id !== this.mainLoadId) {
      return;
    }
    // Commit the resolved kind only after the staleness guard (see resolveFeaturesKind).
    this.featuresKind = featuresKind;
    this.featuresKindResolved = true;

    // Aggregation clients (synchronous setup; queries fire on Mosaic's schedule).
    this.createAggregationClients({
      features: sig.data.features,
      featuresKind,
      predict: sig.data.predict,
      sources: sig.sources,
    });
  }

  /**
   * Detect the `features` column kind (string-list vs struct-list) via a cheap
   * DESCRIBE. Pure detection — callers commit the result to `this.featuresKind`
   * only after their own load-id guard, so a stale in-flight resolve from an
   * earlier `features` column can't overwrite the kind of a newer load.
   */
  private async resolveFeaturesKind(features: SQLField): Promise<FeaturesKind> {
    return detectFeaturesKind(this.context.coordinator, this.context.table, features).catch(() => "list" as const);
  }

  private async loadTopics(metadata: NonNullable<FeaturesListSpec["metadata"]>, id: number) {
    const topicsField = metadata.topics;
    if (topicsField == null) {
      return;
    }
    try {
      const ctx = { table: this.context.table };
      const result: any = await this.context.coordinator.query(
        SQL.Query.from(fromExpr(metadata.table, ctx)).select({
          feature: fieldExpr(metadata.feature, ctx),
          topics: fieldExpr(topicsField, ctx),
        }),
      );
      if (id !== this.mainLoadId) {
        return;
      }
      const map = new Map<string, string[]>();
      for (const row of Array.from(result) as any[]) {
        if (row.feature != null) {
          const ts = row.topics ?? [];
          // De-dupe (preserving order): a feature listed under the same topic
          // twice would otherwise put the same item twice in one group, which
          // the keyed {#each} in FeaturesList rejects as a duplicate key.
          const arr: string[] = Array.isArray(ts) ? Array.from(new Set(Array.from(ts).map((t) => String(t)))) : [];
          map.set(String(row.feature), arr);
        }
      }
      this.topicsByFeature.set(map);
    } catch (err) {
      if (id !== this.mainLoadId) {
        return;
      }
      console.error("[FeaturesList] failed to load topics:", err);
    }
  }

  private createAggregationClients(sig: ClientSignature) {
    const ctx = this.context;
    const client = makeClient({
      coordinator: ctx.coordinator,
      selection: ctx.filter,
      query: (predicate) => buildAggregationQuery(ctx.table, ctx.id, sig, predicate),
      queryResult: (rs: any) => {
        const features: AggFeature[] = [];
        const classTotals = new Map<string, number>();
        const classTotalsSelected = new Map<string, number>();
        let total = 0;
        let totalSelected = 0;

        for (const r of Array.from(rs) as any[]) {
          switch (r.kind) {
            case "count": {
              const count = new Map<string, number>();
              const countSelected = new Map<string, number>();
              for (const entry of (r.by_source ?? []) as any[]) {
                if (entry == null) {
                  continue;
                }
                const src = String(entry.source ?? "");
                count.set(src, Number(entry.c ?? 0));
                countSelected.set(src, Number(entry.cs ?? 0));
              }
              features.push({ feature: String(r.feature), count, countSelected });
              break;
            }
            case "class": {
              const classCount = new Map<string, Map<string, number>>();
              const classCountSelected = new Map<string, Map<string, number>>();
              // While walking by_class, also accumulate per-source totals so
              // the per-source view (count / countSelected) stays populated
              // without a separate count branch on the wire.
              const count = new Map<string, number>();
              const countSelected = new Map<string, number>();
              for (const cls of (r.by_class ?? []) as any[]) {
                if (cls == null) {
                  continue;
                }
                const klassKey = String(cls.klass ?? "");
                const cMap = new Map<string, number>();
                const csMap = new Map<string, number>();
                for (const entry of (cls.by_source ?? []) as any[]) {
                  if (entry == null) {
                    continue;
                  }
                  const src = String(entry.source ?? "");
                  const c = Number(entry.c ?? 0);
                  const cs = Number(entry.cs ?? 0);
                  cMap.set(src, c);
                  csMap.set(src, cs);
                  count.set(src, (count.get(src) ?? 0) + c);
                  countSelected.set(src, (countSelected.get(src) ?? 0) + cs);
                }
                classCount.set(klassKey, cMap);
                classCountSelected.set(klassKey, csMap);
              }
              features.push({
                feature: String(r.feature),
                count,
                countSelected,
                classCount,
                classCountSelected,
              });
              break;
            }
            case "global": {
              const cls = String(r.klass ?? "");
              const c = Number(r.count ?? 0);
              const cs = Number(r.count_selected ?? 0);
              classTotals.set(cls, c);
              classTotalsSelected.set(cls, cs);
              total += c;
              totalSelected += cs;
              break;
            }
          }
        }

        this.aggregation.set({
          features,
          globals: sig.predict != null ? { classTotals, classTotalsSelected, total, totalSelected } : undefined,
        });
      },
    });
    this.clients.push(client);
    // The clients backing the self-exclusion set changed; refresh the clause.
    this.clientsToken++;
    this.syncSelection();
  }
}

/**
 * SQL expression for the list of feature names (VARCHAR[]), regardless of column
 * kind. For struct-list columns the names are projected out of each struct.
 */
function featureValuesExpr(features: SQLField, kind: FeaturesKind, ctx: { table: string }): SQL.ExprNode {
  const e = fieldExpr(features, ctx);
  if (kind === "struct") {
    return SQL.sql`list_transform(${e}, x -> x.feature)`;
  }
  return e;
}

/**
 * SQL expression for the parallel list of sources (VARCHAR[]). Derived from the
 * same base list as featureValuesExpr, so the two stay element-aligned and the
 * positional UNNEST zip in buildAggregationQuery never drifts.
 * - string-list: every feature gets the "default" placeholder source.
 * - struct-list: read x.source, coalescing missing/NULL to "default" so it
 *   doesn't drop out of IN-filters or land under the empty-string source.
 */
function sourceValuesExpr(features: SQLField, kind: FeaturesKind, ctx: { table: string }): SQL.ExprNode {
  const e = fieldExpr(features, ctx);
  if (kind === "struct") {
    return SQL.sql`list_transform(${e}, x -> COALESCE(x.source, 'default'))`;
  }
  return SQL.sql`list_transform(${e}, x -> 'default')`;
}

/**
 * Detect whether the `features` column is a plain string list or a struct list.
 * Uses DESCRIBE on the projected expression and the shared jsTypeFromDBType
 * classifier: a VARCHAR/TEXT list reports "string[]" → "list"; anything else is
 * assumed to be the struct (dict) form, and the query is allowed to fail if the
 * type is genuinely unexpected.
 */
async function detectFeaturesKind(coordinator: Coordinator, table: string, features: SQLField): Promise<FeaturesKind> {
  const [desc] = Array.from(
    await coordinator.query(
      SQL.Query.describe(SQL.Query.from(table).select({ value: fieldExpr(features, { table }) })),
    ),
  ) as any[];
  return jsTypeFromDBType(desc?.column_type ?? "") === "string[]" ? "list" : "struct";
}

function filterExprToExpr(filter: SQL.FilterExpr | undefined | null): SQL.ExprNode {
  if (filter == null) {
    return SQL.literal(true);
  }
  if (filter instanceof Array) {
    if (filter.length === 0) {
      return SQL.literal(true);
    }
    return SQL.and(...filter.map(filterExprToExpr));
  }
  if (typeof filter === "string") {
    return SQL.sql`${filter}`;
  }
  if (typeof filter === "boolean") {
    return SQL.literal(filter);
  }
  return filter as SQL.ExprNode;
}

function applySourceFilter<Q extends { where(...args: any[]): Q }>(q: Q, sig: ClientSignature): Q {
  // `sources === undefined` means "no filter" → keep all sources. An explicit
  // empty array means "no sources selected" → match nothing.
  if (sig.sources == null) {
    return q;
  }
  if (sig.sources.length === 0) {
    return q.where(SQL.literal(false));
  }
  return q.where(
    SQL.isIn(
      SQL.sql`source`,
      sig.sources.map((s) => SQL.literal(s)),
    ),
  );
}

/**
 * Combined aggregation query.
 *
 * The query is built around an exploded CTE — one row per
 * (data row, feature occurrence) — that the count and class branches share:
 *
 *   row_id  — the user-supplied row id (unique per data row)
 *   feature — the unnested feature string
 *   source  — the source string for that feature: from the struct's `source`
 *             field (struct-list columns) or the 'default' placeholder
 *             (string-list columns); always element-aligned with `feature`.
 *   sel     — 1 if the row passes the cross-filter, else 0
 *   klass   — only present in predict mode: predict column cast to TEXT
 *
 * The query returns rows tagged by a `kind` column, unioned via
 * `UNION ALL BY NAME` (DuckDB-specific: missing columns are auto-padded
 * with NULL). To minimize wire size we fold per-source (and per-class) counts
 * into nested LIST<STRUCT> values so each feature appears once on the wire:
 *
 *   kind = 'count'  → one row per feature with `by_source` =
 *                     LIST({source, c, cs})  (count-only mode)
 *   kind = 'class'  → one row per feature with `by_class` =
 *                     LIST({klass, by_source: LIST({source, c, cs})})
 *                     (predict mode only)
 *   kind = 'global' → per-class totals over the selected sources
 *                     (predict mode only; flat shape)
 *
 * In predict mode the count branch is omitted: per-source totals are
 * recovered client-side by summing the class branch's nested data over klass.
 *
 * The 'global' branch is derived from the same (source-filtered) exploded CTE
 * as the class branch, not the base table. Per (klass, source) it counts
 * distinct rows that have any feature in that source, then sums over the
 * selected sources. This makes `M_c` / `N` share the exact scope and counting
 * convention as `a` (per-feature, per-class, summed over selected sources), so
 * the contingency is self-consistent: `a ≤ M_c ≤ N` always hold. The
 * interpretation: "how predictive is having feature f of class c, within the
 * selected sources". A data row that has no feature in any selected source is
 * outside the universe (it can never be feature-present), so it is excluded
 * from `N` rather than padding every feature's absent cell.
 */
function buildAggregationQuery(
  table: string,
  id: string,
  sig: ClientSignature,
  predicate: SQL.FilterExpr | null | undefined,
) {
  // Exploded CTE — one row per (data row, feature occurrence).
  const EXPLODED_CTE_NAME = "__features_list_exploded__";
  const featuresE = featureValuesExpr(sig.features, sig.featuresKind, { table });
  const sourceE = sourceValuesExpr(sig.features, sig.featuresKind, { table });
  const explodedSelect: Record<string, SQL.ExprNode> = {
    row_id: SQL.column(id),
    feature: SQL.sql`UNNEST(${featuresE})::TEXT`,
    source: SQL.sql`UNNEST(${sourceE})::TEXT`,
    sel: SQL.cast(filterExprToExpr(predicate), "INT"),
  };
  if (sig.predict != null) {
    explodedSelect.klass = SQL.sql`(${fieldExpr(sig.predict, { table })})::TEXT`;
  }
  const exploded = SQL.Query.from(table).select(explodedSelect);

  // Count branch — only emitted when there's no predict column. In predict mode
  // the per-source counts are derivable from the class branch's nested data
  // (sum over klass), so we skip it on the wire.
  if (sig.predict == null) {
    let perFeatureSource = SQL.Query.from(EXPLODED_CTE_NAME)
      .select({
        feature: SQL.sql`feature`,
        source: SQL.sql`source`,
        c: SQL.sql`COUNT(DISTINCT row_id)`,
        cs: SQL.sql`COUNT(DISTINCT row_id) FILTER (WHERE sel = 1)`,
      })
      .where(SQL.sql`feature IS NOT NULL`)
      .groupby(SQL.sql`feature`, SQL.sql`source`);
    perFeatureSource = applySourceFilter(perFeatureSource, sig);

    const countsBranch = SQL.Query.from(perFeatureSource)
      .select({
        kind: SQL.literal("count"),
        feature: SQL.sql`feature`,
        by_source: SQL.sql`list({source: source, c: c, cs: cs})`,
      })
      .groupby(SQL.sql`feature`);

    return countsBranch.with({ [EXPLODED_CTE_NAME]: exploded });
  }

  // Class branch — per-feature class breakdown. Three-stage aggregation:
  //   per (feature, klass, source) → fold sources into list per (feature, klass)
  //   → fold klasses into list per feature.
  let perFeatureClassSource = SQL.Query.from(EXPLODED_CTE_NAME)
    .select({
      feature: SQL.sql`feature`,
      klass: SQL.sql`klass`,
      source: SQL.sql`source`,
      c: SQL.sql`COUNT(DISTINCT row_id)`,
      cs: SQL.sql`COUNT(DISTINCT row_id) FILTER (WHERE sel = 1)`,
    })
    .where(SQL.sql`feature IS NOT NULL AND klass IS NOT NULL`)
    .groupby(SQL.sql`feature`, SQL.sql`klass`, SQL.sql`source`);
  perFeatureClassSource = applySourceFilter(perFeatureClassSource, sig);

  const perFeatureClass = SQL.Query.from(perFeatureClassSource)
    .select({
      feature: SQL.sql`feature`,
      klass: SQL.sql`klass`,
      by_source: SQL.sql`list({source: source, c: c, cs: cs})`,
    })
    .groupby(SQL.sql`feature`, SQL.sql`klass`);

  const classBranch = SQL.Query.from(perFeatureClass)
    .select({
      kind: SQL.literal("class"),
      feature: SQL.sql`feature`,
      by_class: SQL.sql`list({klass: klass, by_source: by_source})`,
    })
    .groupby(SQL.sql`feature`);

  // Globals branch — per-class totals over the selected sources, derived from
  // the same exploded CTE as the class branch (not the base table) so they
  // share `a`'s scope and counting convention. Per (klass, source) we count
  // distinct rows holding a feature in that source, then sum over the selected
  // sources, mirroring how `a` is folded client-side.
  let perClassSource = SQL.Query.from(EXPLODED_CTE_NAME)
    .select({
      klass: SQL.sql`klass`,
      source: SQL.sql`source`,
      c: SQL.sql`COUNT(DISTINCT row_id)`,
      cs: SQL.sql`COUNT(DISTINCT row_id) FILTER (WHERE sel = 1)`,
    })
    .where(SQL.sql`feature IS NOT NULL AND klass IS NOT NULL`)
    .groupby(SQL.sql`klass`, SQL.sql`source`);
  perClassSource = applySourceFilter(perClassSource, sig);

  const globalsBranch = SQL.Query.from(perClassSource)
    .select({
      kind: SQL.literal("global"),
      klass: SQL.sql`klass`,
      count: SQL.sql`SUM(c)`,
      count_selected: SQL.sql`SUM(cs)`,
    })
    .groupby(SQL.sql`klass`);

  return SQL.Query.unionAllByName(classBranch, globalsBranch).with({
    [EXPLODED_CTE_NAME]: exploded,
  });
}

function buildData(agg: AggregationResult | null, spec: FeaturesListSpec): Data {
  if (!agg) {
    return {
      allItems: [],
      classNames: [],
      hasClassPolarity: false,
      totalCount: 0,
      hasSelection: false,
      sourceCount: 0,
    };
  }
  const segmentBy = spec.segmentBy === "source";
  const predictMode = spec.data.predict != null;
  const totalCount = agg.features.length;

  let classNames: string[] = [];
  let hasClassPolarity = false;
  if (agg.globals) {
    classNames = Array.from(agg.globals.classTotals.keys()).sort();
    if (classNames.length === 2 && typeof spec.data.predict === "string") {
      const ordered = inferBinaryPolarity(spec.data.predict, [classNames[0], classNames[1]]);
      if (ordered) {
        classNames = [ordered[0], ordered[1]];
        hasClassPolarity = true;
      }
    }
  }

  // sourceTotals: total count per source across all features. Only its size
  // (the number of distinct sources, K) is used below; skew is measured against
  // a uniform "equal per source" reference, not these baseline totals.
  const sourceTotals = new Map<string, number>();
  for (const row of agg.features) {
    for (const [src, c] of row.count) {
      sourceTotals.set(src, (sourceTotals.get(src) ?? 0) + c);
    }
  }
  const sourceCount = sourceTotals.size;

  let hasSelection = false;
  for (const row of agg.features) {
    for (const [src, c] of row.count) {
      const cs = row.countSelected.get(src) ?? 0;
      if (c !== cs) {
        hasSelection = true;
        break;
      }
    }
    if (hasSelection) {
      break;
    }
  }

  // Precompute the Bonferroni-adjusted z for the binary LOR pipeline. Same
  // value for every feature (and for every per-source segment), so we don't
  // recompute Φ⁻¹ inside the loop.
  // Identity: Φ⁻¹(1 − α/2) = √2 · erfinv(1 − α). With family-wise α = 0.05
  // across F features (per-test α' = 0.05/F), the two-sided z is therefore
  // √2 · erfinv(1 − 0.05/F).
  const isBinary = classNames.length === 2;
  const zBinary =
    predictMode && isBinary && agg.globals ? Math.SQRT2 * erfinv(1 - 0.05 / Math.max(agg.features.length, 1)) : 0;

  // Per-class confidence z for the multi-class one-vs-rest LOR lower bounds.
  // Family-wise (feature × class) Bonferroni, mirroring the binary path's
  // per-feature Bonferroni: z = Φ⁻¹(1 − 0.025 / (F·K)) = √2·erfinv(1 − 0.05/(F·K)),
  // with F = feature count and K = number of non-empty classes. This is the
  // tunable confidence parameter from the spec (default = feature×class
  // Bonferroni at family-wise 95%).
  const nonEmptyClasses = agg.globals
    ? classNames.filter((c) => (agg.globals!.classTotalsSelected.get(c) ?? 0) > 0)
    : [];
  const zMulti =
    predictMode && !isBinary && agg.globals
      ? Math.SQRT2 * erfinv(1 - 0.05 / Math.max(agg.features.length * Math.max(nonEmptyClasses.length, 1), 1))
      : 0;

  const rawItems: ListItem[] = [];
  for (const row of agg.features) {
    let totalCount = 0;
    let totalSelected = 0;
    const segs: SourceSegment[] = [];
    for (const [src, c] of row.count) {
      const cs = row.countSelected.get(src) ?? 0;
      totalCount += c;
      totalSelected += cs;
      if (segmentBy) {
        segs.push({ source: src, count: c, countSelected: cs });
      }
    }
    if (segmentBy) {
      segs.sort((a, b) => a.source.localeCompare(b.source));
    }

    let predict: Predict | undefined;
    if (predictMode && agg.globals) {
      const M = agg.globals.classTotalsSelected;
      const N = agg.globals.totalSelected;
      const featureCounts = featureCountsByClass(row);
      if (featureCounts != null) {
        predict = isBinary
          ? buildPredictBinary(featureCounts, classNames, M, zBinary)
          : buildPredictMulticlass(featureCounts, classNames, M, N, zMulti);
      }
    }

    // Source skew over the selected occurrences, split into two
    // frequency-weighted scores that partition the feature's mass by how it
    // spreads across the K sources (p_s = count_{f,s} / n, n = totalSelected):
    //   sourceConc = n·KL(p ‖ uniform) = Σ O_s·ln(K·O_s / n)  — single-source evidence
    //   sourceBal  = n·H(p)            = Σ O_s·ln(n / O_s)     — balanced-across evidence
    // They sum to n·ln(K). The n factor is what keeps rare features off both
    // extremes: a feature seen a handful of times scores low either way.
    // sourceShape is the scale-free concentration in [0, 1] for display.
    let sourceConc: number | undefined;
    let sourceBal: number | undefined;
    let sourceShape: number | undefined;
    if (sourceCount >= 2 && totalSelected > 0) {
      const K = sourceCount;
      const n = totalSelected;
      let conc = 0;
      let bal = 0;
      // Absent sources (O_s = 0) contribute 0 to both sums, so iterating the
      // present sources' selected counts suffices.
      for (const o of row.countSelected.values()) {
        if (o > 0) {
          conc += o * Math.log((K * o) / n);
          bal += o * Math.log(n / o);
        }
      }
      sourceConc = conc;
      sourceBal = bal;
      const lnK = Math.log(K);
      sourceShape = lnK > 0 ? conc / (n * lnK) : 0;
    }

    rawItems.push({
      group: "Features",
      feature: row.feature,
      count: totalCount,
      countSelected: totalSelected,
      predict,
      sourceSegments: segmentBy ? segs : undefined,
      sourceConc,
      sourceBal,
      sourceShape,
    });
  }

  // Search, sort, limit, and topic grouping are applied by the hosting
  // component (so it can keep pinned features visible outside that pipeline).
  return {
    allItems: rawItems,
    classNames,
    hasClassPolarity,
    totalCount,
    hasSelection,
    sourceCount,
  };
}

function featureCountsByClass(feature: AggFeature): Map<string, number> | null {
  const which = feature.classCountSelected;
  if (!which) {
    return null;
  }
  const nByClass = new Map<string, number>();
  for (const [cls, sourceMap] of which) {
    let total = 0;
    for (const n of sourceMap.values()) {
      total += n;
    }
    nByClass.set(cls, total);
  }
  return nByClass;
}

/**
 * Build the list comparator. Sorting is multi-key: the selected sort option is
 * the primary key, total frequency (descending) breaks primary ties, and the
 * feature label (alphabetical) breaks the rest, so the order is fully
 * deterministic even when the primary key is flat across many features.
 */
export function makeSortFunc(
  sort: FeaturesListSpec["sort"],
  hasPredict: boolean,
  sourceCount: number,
): (a: ListItem, b: ListItem) => number {
  const primary = primarySortFunc(sort, hasPredict, sourceCount);
  const byTotal = (a: ListItem, b: ListItem) => b.count - a.count;
  const byLabel = (a: ListItem, b: ListItem) => a.feature.localeCompare(b.feature);
  return (a, b) => primary(a, b) || byTotal(a, b) || byLabel(a, b);
}

function primarySortFunc(
  sort: FeaturesListSpec["sort"],
  hasPredict: boolean,
  sourceCount: number,
): (a: ListItem, b: ListItem) => number {
  switch (sort) {
    case "frequency-ascending":
      return (a, b) => a.countSelected - b.countSelected;
    case "frequency-descending":
      return (a, b) => b.countSelected - a.countSelected;
    case "predictiveness-ascending":
      if (!hasPredict) {
        return () => 0;
      }
      return (a, b) => (a.predict?.strength ?? 0) - (b.predict?.strength ?? 0);
    case "predictiveness-descending":
      if (!hasPredict) {
        return () => 0;
      }
      return (a, b) => (b.predict?.strength ?? 0) - (a.predict?.strength ?? 0);
    case "source-skew-ascending":
      // "Low skew" = balanced across sources: rank by the balance score (desc).
      if (sourceCount < 2) {
        return () => 0;
      }
      return (a, b) => (b.sourceBal ?? 0) - (a.sourceBal ?? 0);
    case "source-skew-descending":
      // "High skew" = concentrated in few sources: rank by the concentration score (desc).
      if (sourceCount < 2) {
        return () => 0;
      }
      return (a, b) => (b.sourceConc ?? 0) - (a.sourceConc ?? 0);
    case "alphabetical":
      return (a, b) => a.feature.localeCompare(b.feature);
    default:
      return () => 0;
  }
}

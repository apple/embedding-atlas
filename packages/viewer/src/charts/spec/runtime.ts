// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { deepEquals } from "@embedding-atlas/utils";
import { makeClient, type MosaicClient, type SelectionClause } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import * as d3 from "d3";
import { derived, writable, type Writable } from "svelte/store";

import { predicateToString } from "../../utils/database.js";
import type { ChartContext } from "../chart.js";
import { computeFieldStats, inferAggregate, type FieldStats } from "../common/aggregate.js";
import type { AxisConfig, ScaleConfig } from "../common/types.js";
import type {
  AggregateFn,
  Channel,
  ChartSpec,
  Dimension,
  Encoding,
  Interpolate,
  Layer,
  MarkStyle,
  Scale,
  SQLField,
  SQLTable,
} from "./spec.js";

export type DataValue = string | number | [number, number] | null | undefined;
export type DataTable = { length: number; columns: Record<string, DataValue[]> };

export interface LayerOutputs {
  key: string;
  primitive: "rect" | "point" | "line" | "rule" | "area";
  style: MarkStyle;
  data: DataTable;
  zIndex: number;
  interpolate: Interpolate;
  orientation?: "vertical" | "horizontal";

  xDimension?: Dimension;
  yDimension?: Dimension;
}

export interface SelectionOutputs {
  /** The key for storing the selection's value in the chart state */
  key: string;
  type: "x" | "y" | "xy";

  clause: (value: any) => Omit<SelectionClause, "source" | "clients"> | undefined;
}

export type AxisOutputs = AxisConfig & {
  title?: string;
};

export interface ChartOutputs {
  scale: Record<string, ScaleConfig>;
  axis: Record<"x" | "y", AxisOutputs>;
  layers: LayerOutputs[];
}

export interface ChartState {
  /** Selections */
  [key: string]: any;
}

interface ScaleHints {
  kind: "quantitative" | "nominal";
  type?: ScaleConfig["type"];
  domain?: DataValue[];
  specialValues?: string[];
  includeZero?: boolean;

  title?: string;

  predicate?: (v: DataValue) => SQL.ExprNode | undefined;
}

export class ChartRuntime {
  private context: ChartContext;
  private spec: ChartSpec | undefined;
  private source: { reset: () => void };
  private cleanupFn: (() => void) | undefined;

  readonly outputs: Writable<ChartOutputs | undefined>;
  readonly selections: Writable<SelectionOutputs[] | undefined>;
  readonly state: Writable<ChartState | undefined>;

  constructor(context: ChartContext, onReset?: () => void) {
    this.context = context;

    this.spec = undefined;
    this.outputs = writable(undefined);
    this.state = writable(undefined);
    this.selections = writable(undefined);

    this.source = {
      reset: () => {
        onReset?.();
      },
    };
  }

  async setSpec(spec: ChartSpec) {
    if (deepEquals(spec, this.spec)) {
      return;
    }

    this.cleanup();
    this.spec = spec;

    let ctx = new BuildContext(this.context, this.spec);

    // Compute stats
    ctx.stats = await computeAllFieldStats(ctx);
    if (this.spec !== spec) {
      // The result is outdated (setSpec is called again while waiting), skip.
      return;
    }

    let clients = buildChart(ctx, this.outputs, this.selections);

    // Sync the filter's selection
    let currentClause: { selection: SelectionOutputs; value: any } | undefined = undefined;

    let unsub = derived([this.selections, this.state], (x) => x).subscribe(([selections, state]) => {
      for (let selection of selections ?? []) {
        let value = state?.[selection.key];
        let clause = selection.clause(value);

        if (currentClause && currentClause.selection === selection && deepEquals(currentClause.value, value)) {
          continue;
        }

        currentClause = {
          selection: selection,
          value: value,
        };

        if (clause) {
          this.context.filter.update({
            source: this.source,
            clients: new Set(clients),
            ...clause,
          });
        } else {
          this.context.filter.update({
            source: this.source,
            clients: new Set(clients),
            value: null,
            predicate: null,
          });
        }
      }
    });

    this.cleanupFn = () => {
      // Set current state to empty to clear existing selections.
      this.source.reset();
      this.state.set(undefined);
      unsub();
      for (let client of clients) {
        try {
          client.destroy();
        } catch (_) {}
      }
    };
  }

  private cleanup() {
    this.cleanupFn?.();
    this.cleanupFn = undefined;
    this.spec = undefined;
  }

  destroy() {
    this.cleanup();
    this.outputs.set(undefined);
    this.state.set(undefined);
    this.selections.set(undefined);
  }
}

class BuildContext {
  constructor(
    public readonly context: ChartContext,
    public readonly spec: ChartSpec,
    public stats: Record<string, FieldStats | undefined> = {},
  ) {}

  fieldExpr(field: SQLField): SQL.ExprNode {
    let vars = { table: this.context.table, filter: "(true)" };
    if (typeof field == "string") {
      return SQL.column(field);
    } else {
      return SQL.sql`${replaceVars(field.sql, vars)}`;
    }
  }

  fromExpr(table: SQLTable, predicate?: string | null): SQL.FromExpr {
    let vars = { table: this.context.table, filter: predicate ?? "(true)" };
    if (typeof table == "string") {
      return new SQL.TableRefNode(table);
    } else {
      return SQL.sql`(${replaceVars(table.sql, vars)})`;
    }
  }
}

function fieldStatsKey(from: SQL.FromExpr, field: SQLField): string {
  return JSON.stringify([from.toString(), field]);
}

/** Compute distribution stats for all fields used in a chart. */
async function computeAllFieldStats(ctx: BuildContext) {
  let result: Record<string, Promise<FieldStats | undefined>> = {};
  (ctx.spec.layers ?? []).forEach((layer) => {
    let from = ctx.fromExpr(layer.from ?? ctx.context.table);
    Object.entries(layer.encoding ?? {}).forEach(([_, encoding]) => {
      if ("field" in encoding && !("aggregate" in encoding)) {
        let key = fieldStatsKey(from, encoding.field);
        result[key] = ctx.context.cache.value("field-stats/" + key, () =>
          computeFieldStats(ctx.context.coordinator, from, ctx.fieldExpr(encoding.field)),
        );
      }
    });
  });
  let entries = Object.entries(result);
  let values = await Promise.all(entries.map((x) => x[1]));
  return Object.fromEntries(entries.map(([k, _], i) => [k, values[i]]));
}

function buildChart(
  ctx: BuildContext,
  outputs: Writable<ChartOutputs | undefined>,
  selections: Writable<SelectionOutputs[] | undefined>,
): MosaicClient[] {
  let clients: MosaicClient[] = [];

  let scaleHints: Record<string, ScaleHints> = {};
  let allOutputs: Writable<LayerOutputs | undefined>[] = [];

  let layers = ctx.spec.layers ?? [];

  // Create clients for each layer.
  layers.forEach((layer, layerIndex) => {
    let { client, outputs, scaleHints: hints } = buildLayer(ctx, layer, layerIndex);

    for (let channel in hints) {
      if (scaleHints[channel]) {
        scaleHints[channel] = mergeScaleHints(scaleHints[channel], hints[channel]);
      } else {
        scaleHints[channel] = hints[channel];
      }
    }
    if (client != undefined) {
      clients.push(client);
    }
    allOutputs.push(outputs);
  });

  derived(allOutputs, (v) => v).subscribe((values) => {
    outputs.set(chartOutputs(ctx.spec, scaleHints, values));
  });

  selections.set(selectionOutputs(ctx.spec, scaleHints));

  return clients;
}

function buildLayer(ctx: BuildContext, layer: Layer, layerIndex: number) {
  // Resolve the encodings for the layer.
  let scaleHints: Record<string, ScaleHints> = {};
  let fieldMappers: Record<string, (v: any) => any> = {};
  let selectEntries: Record<string, SQL.ExprNode> = {};
  let normalizeEntries: Record<string, "x" | "y"> = {};
  let groupBys: SQL.ExprNode[] = [];
  let orders: Record<string, (a: any, b: any) => number> = {};

  let from = ctx.fromExpr(layer.from ?? ctx.context.table);

  // If the layer's encoding has aggregate or field with bin, we must be using a query with groupby.
  // This means all fields need to be in groupby (binned or not), and only aggregates can exist outside groupby.
  let hasAggregateOrBin = false;
  Object.entries(layer.encoding ?? {}).forEach(([attribute, encoding]) => {
    if ("aggregate" in encoding) {
      hasAggregateOrBin = true;
    } else if ("field" in encoding) {
      if (encoding.bin != undefined) {
        hasAggregateOrBin = true;
      }
    }
  });

  Object.entries(layer.encoding ?? {}).forEach(([attribute, encoding]) => {
    let channel = channelForAttribute(attribute);
    let r = buildEncoding(ctx, from, channel, encoding, hasAggregateOrBin);
    if (r == undefined) {
      return;
    }
    if (r.select != undefined) {
      selectEntries[attribute] = r.select;
    }
    if (r.normalize != undefined) {
      normalizeEntries[attribute] = r.normalize;
    }
    if (r.grouping != undefined) {
      groupBys = groupBys.concat(r.grouping);
    }
    if (r.mapper != undefined) {
      fieldMappers[attribute] = r.mapper;
    }
    if (r.order != undefined) {
      orders[attribute] = r.order;
    }
    if (channel != undefined && r.scale != undefined) {
      if (scaleHints[channel] == undefined) {
        scaleHints[channel] = r.scale;
      } else {
        scaleHints[channel] = mergeScaleHints(scaleHints[channel], r.scale);
      }
    }
  });

  let outputs = writable<LayerOutputs | undefined>(undefined);

  if (Object.keys(selectEntries).length == 0) {
    // No select entries, we create a single row table.
    let table: DataTable = { length: 1, columns: {} };
    for (let key in fieldMappers) {
      table.columns[key] = [fieldMappers[key](undefined)];
    }
    let outputs = writable<LayerOutputs | undefined>(layerOutputs(layer, layerIndex, table, orders));
    return { outputs, scaleHints };
  } else {
    let hasFilter = layer.filter == "$filter";
    if (typeof layer.from == "object" && layer.from.sql.indexOf("$filter") >= 0) {
      hasFilter = true;
    }
    let client = makeClient({
      coordinator: ctx.context.coordinator,
      selection: hasFilter ? ctx.context.filter : undefined,
      query: (predicate) => {
        let baseQuery = SQL.Query.from(ctx.fromExpr(layer.from ?? ctx.context.table, predicateToString(predicate)))
          .select(selectEntries)
          .where(layer.filter == "$filter" ? predicate : [])
          .groupby(...groupBys);
        if (Object.keys(normalizeEntries).length == 0) {
          return baseQuery;
        }
        return SQL.Query.from(baseQuery).select(
          Object.fromEntries(
            Object.entries(selectEntries).map(([key, _]) => [
              key,
              normalizeEntries[key] != undefined
                ? SQL.sql`${SQL.column(key)} / (SUM(${SQL.column(key)}) OVER (PARTITION BY ${SQL.column(normalizeEntries[key])}))`
                : key,
            ]),
          ),
        );
      },
      queryResult: (data: any) => {
        let table: DataTable = {
          length: data.numRows,
          columns: {},
        };
        for (let name of data.names) {
          table.columns[name] = Array.from(data.getChild(name));
        }
        for (let key in fieldMappers) {
          if (table.columns[key]) {
            table.columns[key] = table.columns[key].map(fieldMappers[key]);
          } else {
            table.columns[key] = Array.from({ length: table.length }, () => fieldMappers[key](undefined));
          }
        }
        outputs.set(layerOutputs(layer, layerIndex, table, orders));
      },
    });
    return { outputs, client, scaleHints };
  }
}

const numericalAggregates: Partial<
  Record<AggregateFn, (e: SQL.ExprNode, props: { quantile?: number }) => SQL.AggregateNode>
> = {
  min: (e) => SQL.min(e),
  max: (e) => SQL.max(e),
  mean: (e) => SQL.avg(e),
  average: (e) => SQL.avg(e),
  median: (e) => SQL.median(e),
  stdev: (e) => SQL.stddev(e),
  stdevp: (e) => SQL.stddevPop(e),
  variance: (e) => SQL.variance(e),
  variancep: (e) => SQL.varPop(e),
  sum: (e) => SQL.sum(e),
  product: (e) => SQL.product(e),
  quantile: (e, { quantile }) => SQL.quantile(e, quantile ?? 0.5),
};

function buildEncoding(
  ctx: BuildContext,
  from: SQL.FromExpr,
  channel: Channel | undefined,
  encoding: Encoding,
  fieldsInGroupBy: boolean,
):
  | {
      select?: SQL.ExprNode;
      scale?: ScaleHints;
      normalize?: "x" | "y";
      grouping?: SQL.ExprNode[];
      mapper?: (v: any) => any;
      order?: (a: any, b: any) => number;
    }
  | undefined {
  if ("aggregate" in encoding) {
    // Aggregate, generate SQL expression node for the aggregation operation.
    switch (encoding.aggregate) {
      case "count": {
        return {
          select: SQL.count(),
          scale: { title: "Count", kind: "quantitative", includeZero: true },
          normalize: encoding.normalize,
        };
      }
      case "ecdf-value": {
        if (encoding.field != undefined) {
          let qs = quantilesExpr();
          let expr = SQL.cast(ctx.fieldExpr(encoding.field), "DOUBLE");
          return {
            select: SQL.sql`unnest(quantile_disc(${expr}, ${qs}))`,
            scale: {
              kind: "quantitative",
              predicate: (v) => quantitativePredicate(expr, v),
              title: fieldTitle(encoding.field),
            },
            normalize: encoding.normalize,
          };
        } else {
          console.warn(`Invalid spec: aggregate '${encoding.aggregate}' require a field`);
          return;
        }
      }
      case "ecdf-rank": {
        let qs = quantilesExpr();
        return {
          select: SQL.sql`unnest(${qs})`,
          scale: { kind: "quantitative", domain: [0, 1], title: "Quantile" },
          normalize: encoding.normalize,
        };
      }
      default: {
        if (typeof encoding.aggregate == "string" && numericalAggregates[encoding.aggregate]) {
          let fn = numericalAggregates[encoding.aggregate]!;
          if (encoding.field != undefined) {
            let expr = ctx.fieldExpr(encoding.field);
            return {
              select: fn(expr, { quantile: encoding.quantile }).where(SQL.isFinite(expr)),
              scale: {
                kind: "quantitative",
                predicate: (v) => quantitativePredicate(expr, v),
                title: `${encoding.aggregate.toUpperCase()}(${fieldTitle(encoding.field)})`,
              },
              normalize: encoding.normalize,
            };
          } else {
            console.warn(`Invalid spec: aggregate '${encoding.aggregate}' require a field`);
            return;
          }
        }
        if (typeof encoding.aggregate == "object" && "sql" in encoding.aggregate) {
          return {
            select: SQL.sql`${encoding.aggregate.sql}`,
            scale: { kind: "quantitative" },
            normalize: encoding.normalize,
          };
        } else {
          console.warn(`Invalid spec: unknown aggregate '${encoding.aggregate}'`);
          return;
        }
      }
    }
  } else if ("field" in encoding) {
    let bin = encoding.bin;
    if (bin == undefined && fieldsInGroupBy && channel != undefined) {
      bin = {};
    }
    let stats = ctx.stats[fieldStatsKey(from, encoding.field)];
    if (!stats) {
      return undefined;
    }
    if (bin) {
      let aggregate = inferAggregate({
        stats: stats,
        scaleType: channel ? ctx.spec.scale?.[channel]?.type : undefined,
        binCount: bin.desiredCount ?? (channel == "x" || channel == "y" ? 20 : 5),
      });
      if (aggregate == undefined) {
        console.warn(`Invalid spec: unsupported data type`);
        return;
      }
      return {
        select: aggregate.select,
        grouping: [aggregate.select],
        scale: channel
          ? {
              ...aggregate.scale,
              kind: aggregate.scale.type == "band" ? "nominal" : "quantitative",
              predicate: aggregate.predicate as any,
              title: fieldTitle(encoding.field),
            }
          : undefined,
        mapper: aggregate.field,
        order: aggregate.order,
      };
    } else {
      let select = ctx.fieldExpr(encoding.field);
      return {
        select: select,
        grouping: fieldsInGroupBy ? [select] : [],
        scale: {
          kind: stats.quantitative ? "quantitative" : "nominal",
        },
      };
    }
  } else if ("value" in encoding) {
    // Plain values are passed through.
    return {
      mapper: () => encoding.value,
      scale:
        typeof encoding.value == "string"
          ? { kind: "nominal", domain: [encoding.value] }
          : typeof encoding.value == "number"
            ? { kind: "quantitative", domain: [encoding.value] }
            : undefined,
    };
  }
}

function replaceVars(text: string, vars: Record<string, string>): string {
  return text.replace(/\$([a-zA-Z][a-zA-Z0-9\_]*)/g, (original, name) => {
    if (vars[name] != undefined) {
      return vars[name];
    } else {
      return original;
    }
  });
}

function fieldTitle(field: SQLField): string {
  if (typeof field == "string") {
    return field;
  } else {
    return field.sql;
  }
}

function quantilesExpr() {
  let result = [];
  let count = 200;
  for (let i = 0; i <= count; i++) {
    result.push(i / count);
  }
  return SQL.sql`[${result.map((x) => SQL.literal(x)).join(",")}]::DOUBLE[]`;
}

function quantitativePredicate(field: SQL.ExprNode, v: DataValue): SQL.ExprNode | undefined {
  if (v instanceof Array) {
    if (v.length == 2 && typeof v[0] == "number") {
      let [v1, v2] = v;
      if (typeof v1 == "number" && typeof v2 == "number") {
        return SQL.isBetween(field, [Math.min(v1, v2), Math.max(v1, v2)]);
      }
    }
  }
  return undefined;
}

function channelForAttribute(attribute: string): Channel | undefined {
  switch (attribute) {
    case "x":
    case "x1":
    case "x2":
      return "x";
    case "y":
    case "y1":
    case "y2":
      return "y";
    case "color":
      return "color";
    case "size":
      return "size";
  }
  return undefined;
}

function mergeScaleHints(current: ScaleHints, value: ScaleHints): ScaleHints {
  let titles = Array.from(new Set([current.title, value.title].filter((x) => x != undefined && x.trim() != "")));
  let newTitle = titles.length > 0 ? titles.join(", ") : undefined;

  return {
    kind: value.kind,
    type: current.type ?? value.type,
    domain: (current.domain ?? []).concat(value.domain ?? []),
    specialValues: (current.specialValues ?? []).concat(value.specialValues ?? []),
    title: newTitle,
    includeZero: current.includeZero === true || value.includeZero === true ? true : undefined,
    predicate: current.predicate ?? value.predicate,
  };
}

function finiteExtent(values: any[]): [number | undefined, number | undefined] {
  return d3.extent(values, (x) => (typeof x == "number" && isFinite(x) ? x : undefined));
}

function inferScale(spec: Scale, hints: ScaleHints, data: DataValue[][], channel: Channel): ScaleConfig {
  switch (hints.kind) {
    case "quantitative": {
      let parts = [hints.domain ?? [], ...data].map((vals) => finiteExtent(vals.flat())).flat();
      let [min, max] = finiteExtent(parts);
      // Placeholder for no data
      if (min == undefined || max == undefined || !isFinite(min) || !isFinite(max)) {
        min = 0;
        max = 1;
      }
      // Single point, we extend to zero
      if (min == max) {
        if (min > 0) {
          min = 0;
        } else if (min < 0) {
          max = 0;
        } else {
          max = 1;
        }
      }
      // Include zero logic
      let includeZero = channel == "size" || hints.includeZero === true;
      if (includeZero) {
        if (min > 0) {
          min = 0;
        } else if (max < 0) {
          max = 0;
        }
      }
      return {
        type: spec.type ?? hints.type ?? "linear",
        domain: spec.domain ?? [min, max],
        specialValues: hints.specialValues,
        constant: spec.constant,
        range: spec.range,
      };
    }
    case "nominal": {
      let parts = [hints.domain ?? [], ...data]
        .map((x) => new Set(x.flat().filter((v) => typeof v == "string")))
        .flatMap((v) => Array.from(v));
      let specialValuesSet = new Set(hints.specialValues ?? []);
      let stringValues: string[] = Array.from(new Set(parts)).filter((x) => !specialValuesSet.has(x));
      return {
        type: spec.type ?? hints.type ?? "band",
        domain: spec.domain ?? stringValues,
        specialValues: hints.specialValues,
        range: spec.range,
      };
    }
    default:
      return { type: "linear", domain: [0, 1] };
  }
}

function completeCross(data: DataTable, by: string, field: string, value: DataValue): DataTable {
  let key = (value: DataValue | DataValue[]) => JSON.stringify(value);
  let valueForKey = (key: string) => JSON.parse(key);

  let byColumn = data.columns[by].slice();
  let fieldColumn = data.columns[field].slice();
  let otherColumns = Object.entries(data.columns)
    .filter(([name, _]) => name != by && name != field)
    .map(([name, column]) => [name, column.slice()] as [string, DataValue[]]);

  let bySet = new Set<string>();
  let otherSet = new Set<string>();
  let combinedSet = new Set<string>();

  for (let i = 0; i < byColumn.length; i++) {
    let byKey = key(byColumn[i]);
    let otherKey = key(otherColumns.map(([_, c]) => c[i]));
    bySet.add(byKey);
    otherSet.add(otherKey);
    combinedSet.add(byKey + "," + otherKey);
  }

  for (let byKey of bySet) {
    for (let otherKey of otherSet) {
      if (!combinedSet.has(byKey + "," + otherKey)) {
        byColumn.push(valueForKey(byKey));
        fieldColumn.push(value);
        let others = valueForKey(otherKey);
        for (let i = 0; i < otherColumns.length; i++) {
          otherColumns[i][1].push(others[i]);
        }
      }
    }
  }

  return {
    length: byColumn.length,
    columns: {
      [by]: byColumn,
      [field]: fieldColumn,
      ...Object.fromEntries(otherColumns),
    },
  };
}

function stack(data: DataTable, by: "x" | "y", orders: Record<string, (a: any, b: any) => number>): DataTable {
  let field = by == "x" ? "y" : "x";

  let map = new Map<string, number[]>();

  for (let i = 0; i < data.length; i++) {
    let key = data.columns[by] != undefined ? JSON.stringify(data.columns[by][i]) : "";
    let entry = map.get(key);
    if (entry == undefined) {
      map.set(key, [i]);
    } else {
      entry.push(i);
    }
  }

  let newFieldColumn = data.columns[field].slice();

  for (let entry of map.values()) {
    // Sort the items within each stack column by the given orders
    entry.sort((i1, i2) => {
      for (let key in orders) {
        let colump = data.columns[key];
        if (colump == undefined) {
          continue;
        }
        let r = orders[key](colump[i1], colump[i2]);
        if (r != 0) {
          return r;
        }
      }
      return i1 - i2;
    });
    // Stack
    let csum = 0;
    for (let i of entry) {
      let value = newFieldColumn[i];
      if (typeof value != "number" || !isFinite(value)) {
        newFieldColumn[i] = null;
        continue;
      }
      let ncsum = csum + value;
      newFieldColumn[i] = [csum, ncsum];
      csum = ncsum;
    }
  }

  return {
    length: data.length,
    columns: { ...data.columns, [field]: newFieldColumn },
  };
}

function inferOrientation(layer: Layer): "vertical" | "horizontal" {
  if (layer.encoding?.x != undefined && "aggregate" in layer.encoding.x) {
    return "horizontal";
  } else {
    return "vertical";
  }
}

function layerOutputs(
  layer: Layer,
  layerIndex: number,
  data: DataTable,
  orders: Record<string, (a: any, b: any) => number>,
): LayerOutputs | undefined {
  let common: Omit<LayerOutputs, "primitive" | "style"> = {
    key: layerIndex.toString(),
    zIndex: layer.zIndex ?? 0,
    interpolate: layer.interpolate ?? "linear",
    orientation: layer.orientation,
    xDimension: layer.width,
    yDimension: layer.height,
    data: data,
  };
  switch (layer.mark) {
    case "bar":
      switch (layer.orientation ?? inferOrientation(layer)) {
        case "horizontal":
          common.data = stack(data, "y", orders);
          common.yDimension = layer.height ?? { gap: 1, clampToRatio: 0.1 };
          common.orientation = "horizontal";
          break;
        case "vertical":
          common.data = stack(data, "x", orders);
          common.xDimension = layer.width ?? { gap: 1, clampToRatio: 0.1 };
          common.orientation = "vertical";
          break;
        default:
          console.warn(`Invalid orientation: '${layer.orientation}'`);
          return;
      }
      return {
        ...common,
        primitive: "rect",
        style: { fillColor: "$encoding", ...layer.style },
      };
    case "point":
      return {
        ...common,
        primitive: layer.mark,
        style: { fillColor: "$encoding", ...layer.style },
      };
    case "rect":
      return {
        ...common,
        primitive: layer.mark,
        style: { fillColor: "$encoding", ...layer.style },
      };
    case "line":
      return {
        ...common,
        primitive: layer.mark,
        style: {
          strokeColor: "$encoding",
          strokeWidth: 2,
          strokeJoin: "round",
          strokeCap: "round",
          ...layer.style,
        },
      };
    case "area":
      switch (layer.orientation ?? inferOrientation(layer)) {
        case "horizontal":
          common.data = stack(completeCross(data, "y", "x", 0), "y", orders);
          common.orientation = "horizontal";
          break;
        case "vertical":
          common.data = stack(completeCross(data, "x", "y", 0), "x", orders);
          common.orientation = "vertical";
          break;
        default:
          console.warn(`Invalid orientation: '${layer.orientation}'`);
          return;
      }
      return {
        ...common,
        primitive: "area",
        style: { fillColor: "$encoding", ...layer.style },
      };
    case "rule":
      if (layer.orientation == "vertical") {
        common.xDimension = 0;
      } else if (layer.orientation == "horizontal") {
        common.yDimension = 0;
      }
      return {
        ...common,
        primitive: layer.mark,
        style: { strokeColor: "$encoding", strokeWidth: 2, ...layer.style },
      };
    default:
      console.warn(`Invalid mark: '${layer.mark}'`);
      return;
  }
}

function chartOutputs(
  spec: ChartSpec,
  scaleHints: Record<Channel, ScaleHints>,
  layerOutputs: (LayerOutputs | undefined)[],
): ChartOutputs | undefined {
  let scale: Record<string, ScaleConfig> = {};

  let scaleData: Record<string, DataValue[][]> = {};

  // Collect data for scale inference.
  (spec.layers ?? []).forEach((layer, layerIndex) => {
    Object.entries(layer.encoding ?? {}).forEach(([attribute, encoding]) => {
      if (layerOutputs[layerIndex] == undefined) {
        return;
      }
      let channel = channelForAttribute(attribute);
      if (channel != undefined) {
        let data = layerOutputs[layerIndex].data.columns[attribute];
        if (data != undefined) {
          if (scaleData[channel] == undefined) {
            scaleData[channel] = [data];
          } else {
            scaleData[channel].push(data);
          }
        }
      }
    });
  });

  // Infer scales
  for (let channel of Object.keys(scaleHints) as Channel[]) {
    scale[channel] = inferScale(spec.scale?.[channel] ?? {}, scaleHints[channel], scaleData[channel] ?? [], channel);
  }

  return {
    scale: scale,
    axis: {
      x: { title: scaleHints.x?.title, ...(spec.axis?.x ?? {}) },
      y: { title: scaleHints.y?.title, ...(spec.axis?.y ?? {}) },
    },
    layers: layerOutputs.filter((x) => x != undefined),
  };
}

function selectionOutputs(spec: ChartSpec, scaleHints: Record<string, ScaleHints>): SelectionOutputs[] {
  let selections: SelectionOutputs[] = [];

  // Selections
  Object.entries(spec.selection ?? {}).forEach(([key, selection]) => {
    if ("encoding" in selection) {
      let xPredicate: ((v: DataValue) => SQL.ExprNode | undefined) | undefined = undefined;
      let yPredicate: ((v: DataValue) => SQL.ExprNode | undefined) | undefined = undefined;
      // Find the predicate functions from scale hints.
      if (selection.encoding == "x" || selection.encoding == "xy") {
        if (scaleHints.x?.predicate != undefined) {
          xPredicate = scaleHints.x.predicate;
        }
      }
      if (selection.encoding == "y" || selection.encoding == "xy") {
        if (scaleHints.y?.predicate != undefined) {
          yPredicate = scaleHints.y.predicate;
        }
      }
      selections.push({
        key: key,
        type: selection.encoding,
        clause: (value) => {
          if (value == undefined) {
            return;
          }
          let predicate = SQL.and(...[xPredicate?.(value?.x), yPredicate?.(value?.y)].filter((x) => x != undefined));
          return {
            value: value,
            predicate: predicate,
          };
        },
      });
    }
  });

  return selections;
}

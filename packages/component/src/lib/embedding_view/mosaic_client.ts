// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { boundingRect, type Point, type Rectangle } from "../utils.js";
import { fieldExpr, type SQLField } from "./data/fields.js";
import type { DataField, DataPoint, DataPointID } from "./types.js";

/** A predicate for data points: by identifier, or by position (and category index, a SQL expression). */
export function predicateForDataPoints(
  source: { x: SQLField; y: SQLField; identifier?: string | null; category?: SQL.ExprNode | null },
  points: DataPoint[],
) {
  if (points.length == 0) {
    return SQL.literal(false);
  }
  if (source.identifier != null) {
    let identifier = source.identifier;
    return SQL.or(...points.map((p) => SQL.eq(SQL.column(identifier), SQL.literal(p.identifier))));
  } else {
    let x = fieldExpr(source.x);
    let y = fieldExpr(source.y);
    let category = source.category;
    if (category != null) {
      return SQL.or(
        ...points.map((p) =>
          SQL.and(
            SQL.eq(SQL.cast(x, "DOUBLE"), SQL.literal(p.x)),
            SQL.eq(SQL.cast(y, "DOUBLE"), SQL.literal(p.y)),
            SQL.eq(SQL.cast(category, "INTEGER"), SQL.literal(p.category)),
          ),
        ),
      );
    } else {
      return SQL.or(
        ...points.map((p) =>
          SQL.and(SQL.eq(SQL.cast(x, "DOUBLE"), SQL.literal(p.x)), SQL.eq(SQL.cast(y, "DOUBLE"), SQL.literal(p.y))),
        ),
      );
    }
  }
}

function pointInPolygonPredicate(x: SQL.ExprNode, y: SQL.ExprNode, polygon: Point[]) {
  // Equavilant algorithm:
  // let counter = 0;
  // for each edge (x1, y1) to (x2, y2)) {
  //   if ((y1 <= y) && (y < y2) || (y2 <= y) && (y < y1)) { // -> pred1
  //     if ((x2 - x1) * (y - y1) / (y2 - y1) + x1 < x) {    // -> pred2
  //       counter += 1;
  //     }
  //   }
  // }
  // return counter % 2 == 1;

  let parts: SQL.ExprNode[] = [];
  for (let i = 0; i < polygon.length; i++) {
    let j = (i + 1) % polygon.length;
    let { x: x1, y: y1 } = polygon[i];
    let { x: x2, y: y2 } = polygon[j];
    let pred1 =
      y1 < y2
        ? SQL.and(SQL.lte(SQL.literal(y1), y), SQL.lt(y, SQL.literal(y2)))
        : SQL.and(SQL.lte(SQL.literal(y2), y), SQL.lt(y, SQL.literal(y1)));
    let pred2 = (y1 < y2 ? SQL.lt : SQL.gt)(
      SQL.sub(SQL.mul(SQL.literal(x2 - x1), y), SQL.mul(SQL.literal(y2 - y1), x)),
      SQL.literal((x2 - x1) * y1 - (y2 - y1) * x1),
    );
    parts.push(SQL.cast(SQL.and(pred1, pred2), "INT"));
  }
  let sum = parts.reduce((a, b) => SQL.add(a, b));
  return SQL.eq(SQL.mod(sum, SQL.literal(2)), SQL.literal(1));
}

export function predicateForRangeSelection(source: { x: SQLField; y: SQLField }, range: Rectangle | Point[]) {
  let x = fieldExpr(source.x);
  let y = fieldExpr(source.y);
  if (range instanceof Array) {
    if (range.length < 3) {
      // Degenerate case.
      return SQL.literal(false);
    }
    let bounds = boundingRect(range);
    return SQL.and(
      SQL.isBetween(x, [bounds.xMin, bounds.xMax]),
      SQL.isBetween(y, [bounds.yMin, bounds.yMax]),
      pointInPolygonPredicate(x, y, range),
    );
  } else {
    return SQL.and(SQL.isBetween(x, [range.xMin, range.xMax]), SQL.isBetween(y, [range.yMin, range.yMax]));
  }
}

export interface DataPointQuerySource {
  table: string;
  x: SQLField;
  y: SQLField;
  /** The category index (a SQL expression, see `categoryIndexExpr`). */
  category?: SQL.ExprNode | null;
  text?: SQLField | null;
  identifier?: string | null;
  additionalFields?: Record<string, DataField> | null;
}

export class DataPointQuery {
  coordinator: Coordinator;
  source: DataPointQuerySource;
  lastDistance: number;
  selectParams: Record<string, SQL.ExprNode>;

  constructor(coordinator: Coordinator, source: DataPointQuerySource) {
    this.coordinator = coordinator;
    this.source = source;
    this.lastDistance = 0;

    let { x, y, category, text, identifier } = this.source;
    let fieldExpressions: Record<string, SQL.ExprNode> = {};
    let fields = source.additionalFields ?? {};
    for (let key in fields) {
      let spec = fields[key];
      if (typeof spec == "string") {
        fieldExpressions["field_" + key] = SQL.column(spec);
      } else {
        fieldExpressions["field_" + key] = SQL.sql`${spec.sql}`;
      }
    }
    this.selectParams = {
      x: SQL.cast(fieldExpr(x), "DOUBLE"),
      y: SQL.cast(fieldExpr(y), "DOUBLE"),
      ...(category != null ? { category: SQL.cast(category, "INT") } : {}),
      ...(text != null ? { text: fieldExpr(text) } : {}),
      ...(identifier != null ? { identifier: SQL.sql`${SQL.column(identifier)}` } : {}),
      ...fieldExpressions,
    };
  }

  _convertToDataPoint(row: any): DataPoint {
    let fields: Record<string, any> = {};
    for (let key in row) {
      if (key.startsWith("field_")) {
        fields[key.slice("field_".length)] = row[key];
      }
    }
    return {
      x: row.x,
      y: row.y,
      category: row.category,
      text: row.text,
      identifier: row.identifier,
      fields: fields,
    };
  }

  async queryClosestPoint(
    predicate: any | null,
    px: number,
    py: number,
    unitDistance: number,
  ): Promise<DataPoint | null> {
    let rMax = unitDistance * 12;
    let { x, y } = this.source;

    for (let r of [this.lastDistance, rMax]) {
      if (r == 0 || r > rMax) {
        continue;
      }
      let q = SQL.Query.from(this.source.table).select(this.selectParams);
      q = q.where(SQL.isBetween(fieldExpr(x), [px - r, px + r]));
      q = q.where(SQL.isBetween(fieldExpr(y), [py - r, py + r]));
      if (predicate) {
        q = q.where(predicate);
      }
      q = q.orderby(SQL.sql`(x - (${px}))**2 + (y - (${py}))**2`).limit(1);
      let result = await this.coordinator.query(q);
      let point = result.get(0);
      if (point) {
        this.lastDistance = Math.max(Math.abs(point.x - px), Math.abs(point.y - py)) * 4;
        return this._convertToDataPoint(point);
      }
    }
    return null;
  }

  async queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    let { table, identifier } = this.source;
    if (identifier == null) {
      return [];
    }
    let q = SQL.Query.from(table).select(this.selectParams);
    q = q.where(
      SQL.isIn(
        SQL.column(identifier),
        identifiers.map((x) => SQL.literal(x)),
      ),
    );
    let result = Array.from(await this.coordinator.query(q));
    return result.map((row) => this._convertToDataPoint(row));
  }
}

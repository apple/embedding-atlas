// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Coordinator, Selection } from "@uwdata/mosaic-core";
import { column, eq, literal, Query } from "@uwdata/mosaic-sql";
import * as _ from "lodash";

import { ConfigContext, type Config } from "../context/config.svelte.js";
import { CoordinatorContext } from "../context/coordinator.svelte.js";
import type { Schema } from "../model/Schema.svelte.js";
import { TableModel, type Cell, type ColWidths, type TableData } from "../model/TableModel.svelte.js";
import { NumRowsClient } from "../mosaic-clients/NumRowsClient.js";
import { OID, RowsClient } from "../mosaic-clients/RowsClient.js";
import type { Sort } from "../types/index.js";

interface Props {
  tableName: string;
  rowKey: string;
  columns: string[];
  filterBy: Selection | null;
}

export class TableController {
  model: TableModel;
  schema: Schema;
  config: Config;
  coordinator: Coordinator = $derived(CoordinatorContext.coordinator);

  filterBy: Selection | null = null;

  rowsClient: RowsClient | null = null;
  numRowsClient: NumRowsClient | null = null;
  rowKeyColumn: string | null = null;

  public element: HTMLElement | null = $state(null);
  public viewHeight: number = $state(0);
  public viewWidth: number = $state(0);
  public yScroll: number = $state(0);
  public xScroll: number = $state(0);
  public isFetching: boolean = $state(false);
  public isJumping: boolean = $state(false);
  public sort: Sort | null = $state(null);
  public isReady: boolean = $state(false);
  public updateKey: number = $state(0); // increment when the table should refresh rendering
  public isStale: boolean = $state(false);
  public flashedRowId: string | null = $state(null);
  public hoveredRowId: string | null = $state(null);

  public rowsOnScreen = $derived(Math.ceil(this.viewHeight / ConfigContext.config.rowHeight));
  public renderWindowOffset = $derived(this.isJumping ? 0 : ConfigContext.config.renderWindowOffset);
  public firstVisibleRowOID: number | null = $derived.by(() => {
    if (this.model.renderableRows.length === 0) {
      return null;
    }
    const visibleRows = this.model.renderableRows.filter((row) => {
      const screenY = this.model.rowPositions[row] + this.yScroll;
      return screenY + this.model.rowHeights[row] > 0 && screenY < this.viewHeight;
    });

    if (visibleRows.length === 0) {
      return null;
    }
    const firstVisibleRow = visibleRows[0];
    return this.model.data[firstVisibleRow][OID];
  });
  public offset = $derived.by(() => {
    return Math.max(0, Math.floor(-this.yScroll / ConfigContext.config.rowHeight));
  });

  onFetchResolveBegin: Function | null = null;
  onFetchResolveEnd: Function | null = null;

  constructor(model: TableModel, schema: Schema) {
    this.model = model;
    this.schema = schema;
    this.config = ConfigContext.config;
  }

  handleFilterBy = () => {
    if (!this.rowsClient) {
      return;
    }
    this.rowsClient.offset = 0;
    this.rowsClient.limit = this.rowsOnScreen;
    this.isJumping = true;
    this.markStale();
  };

  updateData = (newData: any) => {
    if (!this.model || !this.rowKeyColumn) {
      return;
    }
    if (this.onFetchResolveBegin) {
      this.onFetchResolveBegin();
      this.onFetchResolveBegin = null;
    }

    const arr = newData.toArray();
    const toAdd: TableData = {};
    for (const d of arr) {
      const rowKey = d[this.rowKeyColumn];
      toAdd[rowKey] = d;
    }

    this.model.data = {
      ...this.model.data,
      ...toAdd,
    };

    if (this.onFetchResolveEnd) {
      this.onFetchResolveEnd();
      this.onFetchResolveEnd = null;
    }
    this.isFetching = false;
  };

  public initialize({ tableName, rowKey, columns, filterBy }: Props) {
    this.model.columns = columns;
    this.model.rowKeyColumn = rowKey;
    this.rowKeyColumn = rowKey;

    if (filterBy) {
      this.filterBy = filterBy;
      this.filterBy.addEventListener("value", this.handleFilterBy);
    }

    if (!this.rowKeyColumn) {
      throw new Error("rowkey cannot be null");
    }

    let requiredColumns = columns.includes(this.rowKeyColumn) ? columns : [...columns, this.rowKeyColumn];
    this.rowsClient = new RowsClient(
      tableName,
      requiredColumns,
      filterBy,
      (data) => {
        this.updateData(data);
      },
      (columnInfo) => {
        this.schema.columnInfo = columnInfo;
        this.computeColWidths(tableName, columns);
        this.isReady = true;
      },
    );
    this.coordinator.connect(this.rowsClient);
    this.numRowsClient = new NumRowsClient(tableName, filterBy, (numRows) => {
      if (!this.model) {
        return;
      }
      this.model.numRows = numRows;
    });
    this.coordinator.connect(this.numRowsClient);

    $effect(() => {
      if (!this.rowsClient || this.isFetching || !this.isReady) {
        return;
      }

      const renderWindowTop = -this.renderWindowOffset;
      const renderWindowBottom = this.viewHeight + this.renderWindowOffset;

      const maxRowScreenY = this.model.maxRowPosition + this.yScroll + this.config.rowHeight;
      const minRowScreenY = this.model.minRowPosition + this.yScroll;

      // TODO: do we need to throttle this?
      // if we've scrolled past all rows, then do a clean fetch
      if (
        (minRowScreenY < 0 && maxRowScreenY < 0) ||
        (minRowScreenY > this.viewHeight && maxRowScreenY > this.viewHeight)
      ) {
        const rowsToFetch = this.rowsOnScreen;
        this.isFetching = true;
        this.rowsClient.fetchRows(this.offset, rowsToFetch);
      } else {
        // otherwise, fetch from existing rows
        if (maxRowScreenY < renderWindowBottom) {
          const rowsToFetch = _.clamp(
            Math.ceil((renderWindowBottom - maxRowScreenY) / this.config.rowHeight),
            this.config.minFetchSize,
            this.rowsOnScreen,
          );
          if (rowsToFetch > 0 && this.model.maxRowOID !== this.model.numRows) {
            this.isFetching = true;
            this.rowsClient.fetchRows(this.model.maxRowOID, rowsToFetch);
          }
        }

        const minRowScreenY = this.model.minRowPosition + this.yScroll;
        if (minRowScreenY > renderWindowTop && this.model.minRowOID !== 1) {
          const rowsToFetch = _.clamp(
            Math.ceil((minRowScreenY - renderWindowTop) / this.config.rowHeight),
            this.config.minFetchSize,
            this.rowsOnScreen,
          );
          if (rowsToFetch > 0) {
            this.isFetching = true;
            this.rowsClient.fetchRows(Math.max(0, this.model.minRowOID - 1 - rowsToFetch), rowsToFetch);
          }
        }
      }

      // snapshot renderable rows, as we are going to mutate data.
      const renderableRows = $state.snapshot(this.model.renderableRows);

      // collapse rows that are above the viewport
      let k = 0;
      while (this.model.rowPositions[renderableRows[k]] + this.yScroll + this.model.rowHeights[renderableRows[k]] < 0) {
        // the yScroll needs to adjust to compensate for rows before the current scroll being collapsed
        this.yScroll += this.model.collapseRow(renderableRows[k]);
        k += 1;
      }

      // collapse rows that are below the viewport
      let l = renderableRows.length - 1;
      while (this.model.rowPositions[renderableRows[l]] + this.yScroll > this.viewHeight) {
        this.model.collapseRow(renderableRows[l]);
        l -= 1;
      }

      // remove rows that are above the viewport
      let i = 0;
      while (
        this.model.rowPositions[renderableRows[i]] + this.yScroll + this.model.rowHeights[renderableRows[i]] <
        renderWindowTop
      ) {
        this.model.deleteRow(renderableRows[i]);
        i += 1;
      }

      // remove rows that are below the viewport
      let j = renderableRows.length - 1;
      while (this.model.rowPositions[renderableRows[j]] + this.yScroll > renderWindowBottom) {
        this.model.deleteRow(renderableRows[j]);
        j -= 1;
      }
    });
  }

  public teardown() {
    if (this.filterBy) {
      this.filterBy.removeEventListener("value", this.handleFilterBy);
    }
  }

  public cellIsVisible(cell: Cell): boolean {
    const { x, y } = this.model.getPosition(cell);
    const { width, height } = this.model.getDimensions(cell);

    const screenX = x + this.xScroll;
    const screenY = y + this.yScroll;
    return screenX + width >= 0 && screenX <= this.viewWidth && screenY + height >= 0 && screenY <= this.viewHeight;
  }

  public rowIsVisible(row: string): boolean {
    const y = this.model.rowPositions[row];
    const height = this.model.rowHeights[row];
    const screenY = y + this.yScroll;
    return screenY + height >= 0 && screenY <= this.viewHeight;
  }

  public rowStillExists(row: string): boolean {
    return this.model.data[row] != null;
  }

  public colIsVisible(col: string): boolean {
    const x = this.model.colPositions[col];
    const width = this.model.colWidths[col];
    const screenX = x + this.xScroll;
    return screenX + width >= 0 && screenX <= this.viewWidth;
  }

  public scroll({ deltaX, deltaY }: { deltaX: number; deltaY: number }) {
    if (Math.abs(deltaY) > Math.abs(deltaX)) {
      const newYScroll = this.yScroll - deltaY;
      if (this.model.zeroRowPosition + newYScroll > 0) {
        this.yScroll = -this.model.zeroRowPosition;
      } else if (this.model.finalRowPosition + newYScroll < 0) {
        this.yScroll = -this.model.finalRowPosition;
      } else {
        this.yScroll = newYScroll;
      }
    } else {
      const newXScroll = this.xScroll - deltaX;
      if (-newXScroll < 0) {
        this.xScroll = 0;
      } else if (-newXScroll > Math.max(this.model.colsRightmostPosition, this.viewWidth) - this.viewWidth) {
        this.xScroll = -Math.max(this.model.colsRightmostPosition, this.viewWidth) + this.viewWidth;
      } else {
        this.xScroll = newXScroll;
      }
    }
  }

  public handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.isJumping = false; // scroll is not a jumping operation.
    this.scroll({ deltaX: e.deltaX, deltaY: e.deltaY });
  };

  public jumpToOffset(offset: number) {
    if (!this.rowsClient) {
      return;
    }
    this.isFetching = true;
    const rowsToFetch = this.rowsOnScreen;
    const oldOnFetchResolveEnd = this.onFetchResolveEnd;
    this.onFetchResolveEnd = () => {
      if (oldOnFetchResolveEnd) {
        oldOnFetchResolveEnd();
      }
      this.yScroll = -(offset * this.config.rowHeight);
    };
    this.markStale();
    this.rowsClient.fetchRows(offset, rowsToFetch);
  }

  public handleSort = (sort: Sort | null) => {
    if (!this.rowsClient) {
      return;
    }

    this.sort = sort;
    this.rowsClient.sort = sort;
    this.resetRows();
  };

  private resetRows() {
    this.model.reset();
    this.yScroll = 0;
  }

  private flashRow(rowId: string) {
    this.flashedRowId = rowId;
    setTimeout(() => {
      this.flashedRowId = null;
    }, 400);
  }

  async scrollToRow(rowKey: string, animate: boolean = true) {
    if (!this.rowsClient) {
      return;
    }
    this.isFetching = true;
    const query = Query.with({
      original: this.rowsClient
        .query(this.rowsClient.filterBy?.predicate(this.rowsClient))
        .offset(0)
        .limit(this.model.numRows),
    })
      .select([OID])
      .from("original")
      .where(eq(column(this.rowKeyColumn!), literal(rowKey)));

    const result = await this.coordinator.query(query);
    const arr = result.toArray();
    if (arr.length > 0) {
      const offset = arr[0][OID] - 1;
      this.onFetchResolveEnd = () => {
        if (animate) {
          this.flashRow(rowKey);
        }
      };
      this.jumpToOffset(offset);
    } else {
      this.isFetching = false;
      console.error("no row", rowKey, "found");
    }
  }

  public addHeightToRow(row: string, heightToAdd: number) {
    this.model.rowHeightAddition[row] = (this.model.rowHeightAddition[row] ?? 0) + heightToAdd;
  }

  public hideColumn(col: string) {
    if (col === OID) {
      if (this.config.onShowRowNumberChange) {
        this.config.onShowRowNumberChange(false);
      } else {
        this.config.showRowNumber = false;
      }
    } else {
      if (!this.config.columnConfigs[col]) {
        this.config.columnConfigs[col] = {};
      }

      this.config.columnConfigs[col].hidden = true;
    }
  }

  public showColumn(col: string) {
    if (col === OID) {
      if (this.config.onShowRowNumberChange) {
        this.config.onShowRowNumberChange(true);
      } else {
        this.config.showRowNumber = true;
      }
    } else {
      if (!this.config.columnConfigs[col]) {
        this.config.columnConfigs[col] = {};
      }

      this.config.columnConfigs[col].hidden = false;
    }
  }

  // Marks the current state stale, telling the view to destroy any existing cells on next render.
  private markStale() {
    this.isStale = true;
    const oldBegin = this.onFetchResolveBegin;
    this.onFetchResolveBegin = () => {
      if (oldBegin) {
        oldBegin();
      }
      this.resetRows();
    };
    const oldEnd = this.onFetchResolveEnd;
    this.onFetchResolveEnd = () => {
      if (oldEnd) {
        oldEnd();
      }
      this.updateKey += 1;
      this.isStale = false;
    };
  }

  private async computeColWidths(tableName: string, columns: string[]) {
    const cols = columns.filter((c) => c !== OID);
    const select = this.rowsClient?.getSelect({ includeRowNumber: false });
    const colWidths: ColWidths = cols.reduce((dict: ColWidths, c) => {
      dict[c] = 0;
      return dict;
    }, {});

    const query = Query.from(tableName).select(select).offset(0).limit(10);
    const sample = await this.coordinator.query(query);
    const arr = sample.toArray();
    for (const row of arr) {
      for (const col of cols) {
        colWidths[col] = Math.max(colWidths[col], widthForContent(row[col]));
      }
    }

    if (columns.includes(OID)) {
      colWidths[OID] = this.config.DEFAULT_ROW_NUMBER_COL_WIDTH;
    }
    this.model.defaultColWidths = colWidths;
  }
}

function widthForContent(content: any): number {
  const characterLength = String(content).length;
  if (characterLength > 200) {
    return 600;
  } else if (characterLength > 100) {
    return 300;
  } else if (characterLength > 20) {
    return 200;
  } else if (characterLength > 10) {
    return 150;
  } else {
    return 120;
  }
}

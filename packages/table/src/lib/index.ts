// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator, Selection } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./Table.svelte";
import type { ColumnConfigChangeCallback, ColumnConfigs, RowClickCallback } from "./context/config.svelte";

import type { CustomCell, CustomCellsConfig } from "./api/custom-cells.js";
import type { AdditionalHeaderContent, AdditionalHeaderContentsConfig } from "./api/custom-headers.js";
import type { Theme } from "./api/style.js";

export interface TableProps {
  /** The Mosaic coordinator. If not specified, the default coordinator from Mosaic's `coordinator()` method will be used. */
  coordinator?: Coordinator | null;

  /** The name of the DuckDB table to create a view for. */
  table: string;

  /** The columns of the table to render. These should match the names of the columns in the DuckDB table you wish to render. */
  columns: string[];

  /** The name of the column used to uniquely identify rows. */
  rowKey: string;

  /** Configure columns.
   *
   *  A `ColumnConfigs` is a `Record<string, ColumnConfig>`, where a `ColumnConfig` has the following optional options:
   *  - **width** `number` The width of the column, if not provided, a default width will be chosen by the table.
   *  - **title** `string` The string to render as the title of the column.
   *  - **hidden** `boolean` Whether the column should be hidden from the table.
   *
   *  The properties `width` and `hidden` may be modified by the table via UI interactions, with updates provided through the `onColumnConfigsChange` callback.
   */
  columnConfigs?: ColumnConfigs | null;

  /** A function that will be called whenever the table changes its `columnConfigs`.
   *  You can use this to save configurations through sessions.  */
  onColumnConfigsChange?: ColumnConfigChangeCallback | null;

  /** Whether to show the row number as column. */
  showRowNumber?: boolean | null;

  /** A function that will be called whenever the table changes its `showRowNumber` value.
   *  You can use this to save configurations through sessions. */
  onShowRowNumberChange?: (showRowNumber: boolean) => void;

  /** A Mosaic `Selection` used to filter the table. */
  filter?: Selection | null;

  /** A row's ID to scroll to. When this value is updated, the table will scroll to that row. */
  scrollTo?: any | null;

  /** Light or dark mode. */
  colorScheme?: "light" | "dark" | null;

  /** A theme object, which has the following options:
   *
   *  - **primaryTextColor** `string` The text color of elements such as cells.
   *  - **secondaryTextColor** `string` The text color of elements such as headers.
   *  - **tertiaryTextColor** `string` The text color of elements such as sort buttons.
   *  - **fontFamily** `string` The font family of text in the table.
   *  - **fontSize** `string` The font size of text in the table.
   *  - **primaryBackgroundColor** `string` The background of elements such as headers and even-numbered cells.
   *  - **secondaryBackgroundColor** `string` The background of elements such as odd-numbered cells.
   *  - **hoverBackgroundColor** `string` The background color of hovered elements such as buttons.
   *  - **headerFontFamily** `string` The font family of the header, will fall back to the table's font family.
   *  - **headerFontSize** `string` The font size of the header, will fall back to the table's font size.
   *  - **cellFontFamily** `string` The font family of the cells, will fall back to the table's font family.
   *  - **cellFontSize** `string` The font size of the cells, will fall back to the table's font size.
   *  - **scrollbarBackgroundColor** `string` The background color of the scrollbars.
   *  - **scrollbarPillColor** `string` The background color of the scrollbar pills.
   *  - **scrollbarLabelBackgroundColor** `string` The background color of the vertical scrollbar label.
   *  - **shadow** `string` The shadow of elements such as overlays.
   *  - **outlineColor** `string` The outline of elements such as overlays.
   *  - **dimmedRowColor** `string` The overlay color for dimmed rows when highlighted rows are present in the table.
   *  - **rowScrollToColor** `string` The color of rows will flash when they're scrolled to using the `scrollTo` property of the table.
   *  - **rowHoverColor** `string` The color of rows when they're hovered, enabled through the `showHoveredRow` property of the table.
   *
   *  These values can be css variables if you wish to use css defined custom properties. For example: `{ primaryTextColor: "var(--my-color-variable)" }`.
   *
   *  You can also provide these options as `light` and `dark` properties, which will control the appearance of the table depending on its `colorScheme`. For example:
   *
   *  ```ts
   *  {
   *    light: {
   *      primaryTextColor: "black";
   *    }
   *    dark: {
   *      primaryTextColor: "white";
   *    }
   *  }
   *  ```
   */
  theme?: Theme | null;

  /** The height of each line of text, in pixels. Defaults to `20`. */
  lineHeight?: number | null;

  /** The number of lines of text to show in each row. Defaults to `3`. */
  numLines?: number | null;

  /** You can use this to designate custom renderers for columns. A `CustomCellsConfig` is a `{ [column: string]: CustomCell }`. */
  customCells?: CustomCellsConfig | null;

  /** You can use this to designate additional content for column headers.
   *  Additional header content is rendered above the title of the header, and can
   *  be used to add helpful content such as summaries and visualizations to headers.
   *  A `AdditionalHeaderContentsConfig` is a `{ [column: string]: AdditionalHeaderContent }`. */
  additionalHeaderContents?: AdditionalHeaderContentsConfig | null;

  /** The height of the header, in pixels. Defaults to an auto height based on rendered title. */
  headerHeight?: number | null;

  /** A handler for rows being clicked, which can be used to coordinate with other views. */
  onRowClick?: RowClickCallback | null;

  /** When provided, these all other rows will be dimmed in the table. */
  highlightedRows?: any[] | null;

  /** Whether to highlight the hovered row. */
  highlightHoveredRow?: boolean | null;
}

export class Table {
  private component: any;
  private currentProps: TableProps;

  constructor(target: HTMLElement, props: TableProps) {
    this.currentProps = { ...props };
    this.component = createClassComponent({ component: Component, target: target, props: props });
  }

  update(props: Partial<TableProps>) {
    let updates: Partial<TableProps> = {};
    for (let key in props) {
      if ((props as any)[key] !== (this.currentProps as any)[key]) {
        (updates as any)[key] = (props as any)[key];
        (this.currentProps as any)[key] = (props as any)[key];
      }
    }
    this.component.$set(updates);
  }

  destroy() {
    this.component.$destroy();
  }
}

export type { CustomCell, AdditionalHeaderContent as CustomHeader };

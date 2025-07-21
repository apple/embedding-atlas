// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator, Selection } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./Table.svelte";

import type { CustomCell, CustomCellsConfig } from "./api/custom-cells";
import type { AdditionalHeaderContent, AdditionalHeaderContentsConfig } from "./api/custom-headers";
import type { Theme } from "./api/style";
import type { ColumnConfigChangeCallback, ColumnConfigs, RowClickCallback } from "./context/config.svelte";

export interface TableProps {
  table: string;
  columns: string[];
  rowKey: string;
  columnConfigs?: ColumnConfigs | null;
  onColumnConfigsChange?: ColumnConfigChangeCallback | null;
  showRowNumber?: boolean | null;
  onShowRowNumberChange?: (showRowNumber: boolean) => void;
  coordinator?: Coordinator | null;
  filter?: Selection | null;
  scrollTo?: any | null;
  colorScheme?: "light" | "dark" | null;
  theme?: Theme | null;
  lineHeight?: number | null;
  numLines?: number | null;
  customCells?: CustomCellsConfig | null;
  additionalHeaderContents?: AdditionalHeaderContentsConfig | null;
  headerHeight?: number | null;
  onRowClick?: RowClickCallback | null;
  highlightedRows?: any[] | null;
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

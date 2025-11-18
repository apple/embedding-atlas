// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// The component API for embedding viewer.

import type { EmbeddingViewConfig, Label } from "@embedding-atlas/component";
import type { CustomCell } from "@embedding-atlas/table";
import type { Coordinator } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingAtlas.svelte";

import type { ChartThemeConfig } from "./charts/common/theme.js";
import type { DefaultChartsConfig } from "./charts/default_charts.js";

import cssCode from "./app.css?inline";

export interface EmbeddingAtlasProps {
  /** The Mosaic coordinator. */
  coordinator: Coordinator;

  /** The data source. */
  data: {
    /** The name of the data table. */
    table: string;

    /** The column for unique row identifiers. */
    id: string;

    /** The X and Y columns for the embedding projection view. */
    projection?: { x: string; y: string } | null;

    /** The column for pre-computed nearest neighbors.
     *  Each value in the column should be a dictionary with the format: `{ "ids": [id1, id2, ...], "distances": [distance1, distance2, ...] }`.
     *  `"ids"` should be an array of row ids (as given by the `idColumn`) of the neighbors, sorted by distance.
     *  `"distances"` should contain the corresponding distances to each neighbor.
     *  Note that if `searcher.nearestNeighbors` is specified, the UI will use the searcher instead.
     */
    neighbors?: string | null;

    /** The column for text. The text will be used as content for the tooltip and search features. */
    text?: string | null;
  };

  /** The color scheme. */
  colorScheme?: "light" | "dark" | null;

  /** The initial viewer state. */
  initialState?: EmbeddingAtlasState | null;

  /**
   * Configure the default charts.
   * By default, we show a distribution chart for each column based on the data type in addition to the embedding and table.
   * You may configure these charts with this option.
   */
  defaultChartsConfig?: DefaultChartsConfig | null;

  /** Configuration for the embedding view. See docs for the EmbeddingView. */
  embeddingViewConfig?: EmbeddingViewConfig | null;

  /** Labels for the embedding view. */
  embeddingViewLabels?: Label[] | null;

  /** Theme config for charts. */
  chartTheme?: ChartThemeConfig | null;

  /** Custom CSS stylesheet to apply at the root of the component. */
  stylesheet?: string | null;

  /** An object that provides search functionalities, including full text search, vector search, and nearest neighbor queries.
   *  If not specified (undefined), a default full-text search with the text column will be used.
   *  If set to null, search will be disabled. */
  searcher?: Searcher | null;

  /** Custom cell renderers for the table view. */
  tableCellRenderers?: Record<string, CustomCell | "markdown">;

  /** A callback to export the currently selected points. */
  onExportSelection?:
    | ((predicate: string | null, format: "json" | "jsonl" | "csv" | "parquet") => Promise<void>)
    | null;

  /** A callback to download the application as archive. */
  onExportApplication?: (() => Promise<void>) | null;

  /** A callback when the state of the viewer changes. You may serialize the state to JSON and load it back. */
  onStateChange?: ((state: EmbeddingAtlasState) => void) | null;

  /** A cache to speed up initialization of the viewer. */
  cache?: Cache | null;
}

export interface EmbeddingAtlasState {
  /** The version of Embedding Atlas that created this state. */
  version: string;

  /** UNIX timestamp when this was created. */
  timestamp: number;

  /** The list of charts. */
  charts?: Record<string, any>;

  /** The state of all charts, stored as a map of id to chart state. */
  chartStates?: Record<string, any>;

  /** The current layout */
  layout?: string;

  /** The state of all layouts. */
  layoutStates?: Record<string, any>;

  /** The selection predicate (SQL expression).
   *  This property is derived from chart states, changing this directly has no effect. */
  predicate?: string | null;
}

export interface Cache {
  /** Gets an object from the cache with the given key. Returns `null` if the entry is not found. */
  get(key: string): Promise<any | null>;

  /** Sets an object to the cache with the given key */
  set(key: string, value: any): Promise<void>;
}

export interface Searcher {
  /** Perform a full text search with the given query */
  fullTextSearch?(
    query: string,
    options?: { limit: number; predicate: string | null; onStatus: (status: string) => void },
  ): Promise<{ id: any }[]>;

  /** Perform a vector search with the given query */
  vectorSearch?(
    query: string,
    options?: { limit: number; predicate: string | null; onStatus: (status: string) => void },
  ): Promise<{ id: any; distance?: number }[]>;

  /** Find nearest neighbors of the row of the given id */
  nearestNeighbors?(
    id: any,
    options?: { limit: number; predicate: string | null; onStatus: (status: string) => void },
  ): Promise<{ id: any; distance?: number }[]>;
}

export class EmbeddingAtlas {
  private component: any;
  private container: HTMLDivElement;
  private currentProps: EmbeddingAtlasProps;

  constructor(target: HTMLElement, props: EmbeddingAtlasProps) {
    this.currentProps = { ...props };

    // Container element
    this.container = document.createElement("div");
    this.container.style.display = "flex";
    this.container.style.width = "100%";
    this.container.style.height = "100%";
    target.appendChild(this.container);

    // Shadow root on container
    let shadowRoot = this.container.attachShadow({ mode: "open" });
    let sheet = new CSSStyleSheet();
    sheet.replaceSync(cssCode);
    shadowRoot.adoptedStyleSheets = [sheet];
    if (props.stylesheet != undefined) {
      let customSheet = new CSSStyleSheet();
      customSheet.replaceSync(props.stylesheet);
      shadowRoot.adoptedStyleSheets.push(customSheet);
    }

    // Inner container element
    let innerContainer = document.createElement("div");
    innerContainer.style.display = "flex";
    innerContainer.style.width = "100%";
    innerContainer.style.height = "100%";
    shadowRoot.appendChild(innerContainer);

    // The Svelte component
    this.component = createClassComponent({ component: Component, target: innerContainer, props: props });
  }

  update(props: Partial<EmbeddingAtlasProps>) {
    let updates: Partial<EmbeddingAtlasProps> = {};
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
    this.container.remove();
  }
}

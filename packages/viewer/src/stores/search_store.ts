// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { debounce } from "@embedding-atlas/utils";
import type { Coordinator } from "@uwdata/mosaic-core";
import { derived, get, writable, type Readable, type Writable } from "svelte/store";

import type { Searcher } from "../api.js";
import type { Overlay } from "../charts/chart.js";
import { performSearch, querySearchResultItems, resolveSearcher, type SearchResultItem } from "../search/search.js";
import type { ColumnDesc } from "../utils/database.js";
import { latestAsync } from "../utils/latest_async.js";

const SEARCH_LIMIT = 500;

export interface SearchStoreConfig {
  coordinator: Coordinator;
  table: string;
  id: string;
  getCurrentPredicate: () => string | null;
  getColumns: () => ColumnDesc[];

  text?: string;
  neighbors?: string;
  searcher?: Searcher;
}

interface SearchStoreResult {
  query: any;
  mode: string;
  limit: number;
  label: string;
  highlight: string;
  items: SearchResultItem[];
  overlay: Overlay | null;
}

export class SearchStore {
  /** Input configuration — set this instead of calling configure() */
  config: Writable<SearchStoreConfig | undefined>;

  /** Supported search modes */
  searchModes: Readable<string[]>;

  /** Current search mode in the search panel */
  mode: Writable<string>;

  /** Current text column in the search panel (for in-browser full-text search mode) */
  textColumn: Writable<string | undefined>;

  /** Current query in the search panel */
  query: Writable<any>;

  /** Search result, call `request` to start search and populate this */
  result: Writable<SearchStoreResult | undefined>;

  /** Searcher status */
  status: Writable<string | undefined>;

  private _searchDerived: Readable<{
    modes: string[];
    search: ((query: any, mode: string) => Promise<SearchStoreResult | undefined>) | undefined;
  }>;
  private _requestSearch: (query: any, mode: string) => void;

  constructor() {
    this.config = writable(undefined);
    this.mode = writable("full-text");
    this.textColumn = writable(undefined);
    this.query = writable(undefined);
    this.result = writable(undefined);
    this.status = writable(undefined);

    const effectiveConfig = derived([this.config, this.textColumn], ([config, textColumn]) => {
      if (!config) return undefined;
      if (textColumn !== undefined && textColumn !== config.text) {
        return { ...config, text: textColumn };
      }
      return config;
    });

    this._searchDerived = derived(effectiveConfig, (config) => {
      if (!config) return { modes: [], search: undefined };

      let searcher = resolveSearcher({
        coordinator: config.coordinator,
        table: config.table,
        idColumn: config.id,
        textColumn: config.text,
        neighborsColumn: config.neighbors,
        searcher: config.searcher,
      });

      let modes = [
        ...(searcher.fullTextSearch != null ? ["full-text"] : []),
        ...(searcher.vectorSearch != null ? ["vector"] : []),
      ];

      const search = async (query: any, mode: string): Promise<SearchStoreResult | undefined> => {
        let predicate = config.getCurrentPredicate?.() ?? null;
        let searcherResult = await performSearch({
          searcher: searcher,
          predicate: predicate,
          query: query,
          mode: mode,
          limit: SEARCH_LIMIT,
          onStatus: (value) => {
            this.status.set(value);
          },
        });

        let fields: Record<string, string> = Object.fromEntries(config.getColumns().map((c) => [c.name, c.name]));

        let result = await querySearchResultItems(
          config.coordinator,
          config.table,
          config.id,
          fields,
          predicate,
          searcherResult,
        );

        let label = query.toString().trim();
        let highlight = query.toString().trim();
        let overlay: Overlay = {
          nodes: result.map((x) => x.id),
        };

        if (mode == "raw") {
          label = query.label ?? "Raw Points";
          overlay = query.overlay ?? { nodes: result.map((x) => x.id) };
          highlight = "";
        }

        this.status.set(undefined);

        return {
          query: query,
          mode: mode,
          limit: SEARCH_LIMIT,
          label: label,
          highlight: highlight,
          overlay: overlay,
          items: result,
        };
      };

      return { modes, search };
    });

    this.searchModes = derived(this._searchDerived, (d) => d.modes);

    // Reset state when effective config changes
    effectiveConfig.subscribe(() => {
      this.mode.set("full-text");
      this.query.set(undefined);
      this.result.set(undefined);
      this.status.set(undefined);
    });

    // Construct the wrapped search function to use latestAsync
    this._requestSearch = latestAsync(
      async (query, mode) => {
        if (query === undefined || query === "") {
          return undefined;
        }
        return await get(this._searchDerived).search?.(query, mode);
      },
      (result) => {
        this.result.set(result);
      },
    );

    const debouncedSearch = debounce((query: any, mode: string) => {
      this.search(query, mode);
    }, 500);

    // Start search on query / mode change
    derived([this.query, this.mode], (x) => x).subscribe(([query, mode]) => {
      if (query == null || query === "") {
        // Clear is immediate.
        this.search(undefined, mode);
      } else {
        debouncedSearch(query, mode);
      }
    });
  }

  search(query: any, mode: string) {
    this._requestSearch(query, mode);
  }

  clear() {
    this._requestSearch(undefined, "none");
  }
}

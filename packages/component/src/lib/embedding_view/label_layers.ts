// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ViewportState } from "../utils.js";
import type { AutoLabels, ResolvedLabelsLayer } from "./layers.js";
import type { Label } from "./types.js";

/** Generated labels, with the viewport they were generated over (null for static labels). */
export interface LayerLabels {
  labels: Label[];
  viewport: ViewportState | null;
}

interface LayerState {
  /** The static labels shown, if any. */
  static: Label[] | null;
  /** The request (key) of the automatic labels, the last one started or waiting to start. */
  requestKey: string | null;
  /** The query of `requestKey` (see `AutoLabels.queryKey`). */
  queryKey: string | null;
  /** The labels shown. */
  result: LayerLabels | null;
  timer: ReturnType<typeof setTimeout> | null;
  /** Identifies the current generation; results of other generations are dropped. */
  token: object | null;
  /** The latest automatic labels request of the layer (for retrying a failed generation). */
  auto: AutoLabels | null;
  /** Whether the current request was retried after failing. */
  retried: boolean;
}

export interface LabelLayersOptions {
  /** Generate the automatic labels of a request. */
  generate: (auto: AutoLabels) => Promise<LayerLabels>;
  /** Called when the labels of any layer change (or the layers change). */
  onChange: () => void;
  /** Called when the number of running generations changes. */
  onGenerating?: (count: number) => void;
  onError?: (error: unknown) => void;
  /** Delay before regenerating labels when only their data revision changes (e.g., the filter). */
  debounce?: number;
  /** Delay before retrying a failed generation (once; after that, labels are generated again when the request
   *  changes). */
  retryDelay?: number;
}

/** The labels of the labels layers of an embedding view: decides when automatic labels are generated, which labels
 *  are shown while they're generated, and drops results that are no longer current.
 *
 *  - Static labels are shown as they are.
 *  - Automatic labels are generated when their `requestKey` changes. If only the data revision changed (the same
 *    `queryKey`), the current labels stay until the new ones arrive, and the generation is debounced (e.g., while
 *    brushing). Otherwise the labels are cleared and generated right away.
 *  - A layer whose inputs aren't ready waits (its labels are cleared if its query changed). */
export class LabelLayers {
  private states = new Map<string, LayerState>();
  private order: string[] = [];
  private generating = 0;
  private options: LabelLayersOptions;

  constructor(options: LabelLayersOptions) {
    this.options = options;
  }

  /** The labels of the layers, in layer order. */
  results(): { key: string; labels: LayerLabels | null }[] {
    return this.order.map((key) => ({ key, labels: this.states.get(key)?.result ?? null }));
  }

  sync(layers: ResolvedLabelsLayer[]) {
    let order = layers.map((l) => l.key);
    let changed = order.length != this.order.length || order.some((k, i) => k != this.order[i]);
    this.order = order;

    let keys = new Set(order);
    for (let [key, state] of this.states) {
      if (!keys.has(key)) {
        this.cancel(state);
        this.states.delete(key);
        changed = true;
      }
    }

    for (let layer of layers) {
      let state = this.states.get(layer.key);
      if (state == null) {
        state = {
          static: null,
          requestKey: null,
          queryKey: null,
          result: null,
          timer: null,
          token: null,
          auto: null,
          retried: false,
        };
        this.states.set(layer.key, state);
      }
      changed = this.syncLayer(layer.key, state, layer) || changed;
    }

    if (changed) {
      this.options.onChange();
    }
  }

  destroy() {
    for (let state of this.states.values()) {
      this.cancel(state);
    }
    this.states.clear();
    this.order = [];
  }

  /** Update a layer's state. Returns whether its labels changed. */
  private syncLayer(key: string, state: LayerState, layer: ResolvedLabelsLayer): boolean {
    if (layer.labels != null) {
      if (state.static === layer.labels) {
        return false;
      }
      this.cancel(state);
      state.static = layer.labels;
      state.requestKey = null;
      state.queryKey = null;
      state.result = { labels: layer.labels, viewport: null };
      return true;
    }

    let changed = false;
    if (state.static != null) {
      state.static = null;
      state.result = null;
      changed = true;
    }

    let auto = layer.auto;
    state.auto = auto;
    if (auto == null) {
      this.cancel(state);
      state.requestKey = null;
      state.queryKey = null;
      if (state.result != null) {
        state.result = null;
        changed = true;
      }
      return changed;
    }

    let sameQuery = state.queryKey == auto.queryKey;
    if (!auto.ready) {
      // Wait for the inputs. Labels of another query are no longer the layer's labels.
      if (!sameQuery) {
        this.cancel(state);
        state.requestKey = null;
        state.queryKey = auto.queryKey;
        if (state.result != null) {
          state.result = null;
          changed = true;
        }
      } else if (state.timer != null) {
        // A pending regeneration is for data that's no longer current: wait for the new data instead.
        this.cancel(state);
        state.requestKey = null;
      }
      return changed;
    }

    if (state.requestKey == auto.requestKey) {
      return changed;
    }

    // Only the data revision changed (for the same query, generated or being generated): keep the current labels,
    // and debounce. Otherwise, clear the labels and generate right away.
    let debounce = sameQuery && state.requestKey != null;
    if (!sameQuery && state.result != null) {
      state.result = null;
      changed = true;
    }
    this.cancel(state);
    state.requestKey = auto.requestKey;
    state.queryKey = auto.queryKey;
    state.retried = false;
    let run = () => {
      state.timer = null;
      this.start(key, state, auto);
    };
    if (debounce) {
      state.timer = setTimeout(run, this.options.debounce ?? 300);
    } else {
      run();
    }
    return changed;
  }

  private start(key: string, state: LayerState, auto: AutoLabels) {
    let token = {};
    state.token = token;
    this.setGenerating(this.generating + 1);
    this.options
      .generate(auto)
      .then((result) => {
        if (state.token !== token || this.states.get(key) !== state) {
          return;
        }
        state.token = null;
        state.result = result;
        this.options.onChange();
      })
      .catch((error) => {
        // Errors of generations that were cancelled (or of removed layers) don't matter anymore.
        if (state.token !== token || this.states.get(key) !== state) {
          return;
        }
        state.token = null;
        this.options.onError?.(error);
        // Retry once after a delay (e.g., after a transient error). If it fails again, the labels are generated again
        // when the request changes (retrying on every sync would repeat a failing query, e.g., while brushing).
        if (!state.retried) {
          state.retried = true;
          state.timer = setTimeout(() => {
            state.timer = null;
            if (state.auto != null && state.auto.requestKey == state.requestKey && state.auto.ready) {
              this.start(key, state, state.auto);
            }
          }, this.options.retryDelay ?? 2000);
        }
      })
      .finally(() => {
        this.setGenerating(this.generating - 1);
      });
  }

  /** Cancel the pending or running generation of a layer (a running generation's result is dropped). */
  private cancel(state: LayerState) {
    if (state.timer != null) {
      clearTimeout(state.timer);
      state.timer = null;
    }
    state.token = null;
  }

  private setGenerating(count: number) {
    this.generating = count;
    this.options.onGenerating?.(count);
  }
}

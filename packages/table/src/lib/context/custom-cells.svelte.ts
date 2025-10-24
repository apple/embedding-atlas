// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { getContext, setContext } from "svelte";
import type { CustomCellsConfig } from "../api/custom-cells.js";

export class CustomCellsState {
  config: CustomCellsConfig = $state({});
}

const CUSTOM_CELLS_KEY = Symbol("custom-cells");

export class CustomCellsContext {
  public static initialize() {
    setContext(CUSTOM_CELLS_KEY, new CustomCellsState());
  }

  public static set config(value: CustomCellsConfig) {
    const customCellsState: CustomCellsState = getContext(CUSTOM_CELLS_KEY);
    customCellsState.config = value;
  }

  public static get config(): CustomCellsConfig {
    const customCellsState: CustomCellsState = getContext(CUSTOM_CELLS_KEY);
    return customCellsState.config;
  }
}

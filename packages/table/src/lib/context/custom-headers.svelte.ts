// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { getContext, setContext } from "svelte";
import type { AdditionalHeaderContentsConfig } from "../api/custom-headers";

export class CustomHeadersState {
  config: AdditionalHeaderContentsConfig = $state({});
}

const CUSTOM_HEADERS_KEY = "custom-cells";

export class CustomHeadersContext {
  public static initialize() {
    setContext(CUSTOM_HEADERS_KEY, new CustomHeadersState());
  }

  public static set config(value: AdditionalHeaderContentsConfig) {
    const CustomHeadersState: CustomHeadersState = getContext(CUSTOM_HEADERS_KEY);
    CustomHeadersState.config = value;
  }

  public static get config(): AdditionalHeaderContentsConfig {
    const customHeadersState: CustomHeadersState = getContext(CUSTOM_HEADERS_KEY);
    return customHeadersState.config;
  }
}

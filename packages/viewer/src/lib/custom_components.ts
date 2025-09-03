// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { DataPoint, OverlayProxy } from "@embedding-atlas/component";
import type { Component } from "svelte";
import { createClassComponent } from "svelte/legacy";

import SearchResultOverlay from "./SearchResultOverlay.svelte";
import Tooltip from "./Tooltip.svelte";

import type { SearchResultItem } from "./search.js";

function createCustomComponentClass<Props extends {}>(Component: Component<Props>): any {
  return class {
    private component: any;

    constructor(target: HTMLDivElement, props: Props) {
      this.component = createClassComponent({ component: Component, target: target, props: props });
    }

    update(props: Partial<Props>) {
      this.component.$set(props);
    }

    destroy() {
      this.component.$destroy();
    }
  };
}

export const CustomTooltip = createCustomComponentClass<{
  tooltip: DataPoint;
  darkMode: boolean;
  onNearestNeighborSearch?: (id: any) => void;
}>(Tooltip);

export const CustomOverlay = createCustomComponentClass<{ proxy: OverlayProxy; items: SearchResultItem[] }>(
  SearchResultOverlay,
);

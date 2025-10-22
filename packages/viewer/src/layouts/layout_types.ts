// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";
import GridLayout from "./GridLayout.svelte";
import ListLayout from "./ListLayout.svelte";

import type { LayoutOptionsProps, LayoutProps } from "./layout.js";
import ListLayoutOptions from "./ListLayoutOptions.svelte";

export type LayoutComponentClass = Component<LayoutProps<any>, {}, "">;
export type LayoutOptionsComponentClass = Component<LayoutOptionsProps<any>, {}, "">;

export const layoutTypes: Record<string, [LayoutComponentClass, LayoutOptionsComponentClass]> = {
  list: [ListLayout, ListLayoutOptions],
  grid: [GridLayout, ListLayoutOptions],
};

export function findLayoutComponent(type: string): LayoutComponentClass {
  if (layoutTypes[type] == null) {
    return layoutTypes.list[0];
  }
  return layoutTypes[type][0];
}

export function findLayoutOptionsComponent(type: string): LayoutOptionsComponentClass {
  if (layoutTypes[type] == null) {
    return layoutTypes.list[1];
  }
  return layoutTypes[type][1];
}

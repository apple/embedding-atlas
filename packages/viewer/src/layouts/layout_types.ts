// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";

import DashboardLayout from "./dashboard/DashboardLayout.svelte";
import DashboardLayoutOptions from "./dashboard/DashboardLayoutOptions.svelte";
import ListLayout from "./list/ListLayout.svelte";
import ListLayoutOptions from "./list/ListLayoutOptions.svelte";

import type { DashboardLayoutSpec } from "./dashboard/types.js";
import type { LayoutOptionsProps, LayoutProps } from "./layout.js";
import type { ListLayoutSpec } from "./list/types.js";

export type BuiltinLayoutSpec = ListLayoutSpec | DashboardLayoutSpec;

export type LayoutComponentClass = Component<LayoutProps, {}, "">;
export type LayoutOptionsComponentClass = Component<LayoutOptionsProps, {}, "">;

export const layoutTypes: Record<string, [LayoutComponentClass, LayoutOptionsComponentClass | undefined]> = {
  list: [ListLayout, ListLayoutOptions],
  dashboard: [DashboardLayout, DashboardLayoutOptions],
};

export function findLayoutComponent(type: string): LayoutComponentClass {
  if (layoutTypes[type] == null) {
    return layoutTypes.list[0];
  }
  return layoutTypes[type][0];
}

export function findLayoutOptionsComponent(type: string): LayoutOptionsComponentClass | undefined {
  if (layoutTypes[type] == null) {
    return layoutTypes.list[1];
  }
  return layoutTypes[type][1];
}

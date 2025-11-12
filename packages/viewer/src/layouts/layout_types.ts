// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";

import DashboardLayout from "./dashboard/DashboardLayout.svelte";
import DashboardLayoutOptions from "./dashboard/DashboardLayoutOptions.svelte";
import ListLayout from "./list/ListLayout.svelte";
import ListLayoutOptions from "./list/ListLayoutOptions.svelte";

import type { LayoutOptionsProps, LayoutProps } from "./layout.js";

export type LayoutComponentClass = Component<LayoutProps<any>, {}, "">;
export type LayoutOptionsComponentClass = Component<LayoutOptionsProps<any>, {}, "">;

export const layoutTypes: Record<string, [LayoutComponentClass, LayoutOptionsComponentClass]> = {
  list: [ListLayout, ListLayoutOptions],
  dashboard: [DashboardLayout, DashboardLayoutOptions],
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

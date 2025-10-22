// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { derived, writable } from "svelte/store";

export type ColorScheme = "light" | "dark";

let matcher = typeof window !== "undefined" ? window.matchMedia?.("(prefers-color-scheme: dark)") : null;

let systemColorScheme = writable<ColorScheme>(matcher?.matches ? "dark" : "light");
matcher?.addEventListener("change", (event) => {
  systemColorScheme.set(event.matches ? "dark" : "light");
});

export { systemColorScheme };

export function makeColorSchemeStore() {
  let userColorScheme = writable<ColorScheme | undefined | null>(null);
  let colorScheme = derived([systemColorScheme, userColorScheme], ([system, user]) => user ?? system);
  return { colorScheme, userColorScheme };
}

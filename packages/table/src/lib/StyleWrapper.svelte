<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount, type Snippet } from "svelte";

  import { StyleContext } from "./context/style.svelte";

  import { resolveTheme, type Theme } from "./api/style.js";

  interface Props {
    children: Snippet;
  }

  let { children }: Props = $props();

  const style = StyleContext.style;

  let userColorScheme: "light" | "dark" | null = $derived(style.colorScheme);
  let theme: Theme = $derived(style.theme);

  let systemColorScheme: "light" | "dark" | null = $state(null);
  let colorScheme = $derived(userColorScheme ?? systemColorScheme ?? "light");

  const onSystemColorSchemeChange = (e: MediaQueryListEvent) => {
    if (e.matches) {
      systemColorScheme = "dark";
    } else {
      systemColorScheme = "light";
    }
  };
  const darkMatcher = "(prefers-color-scheme: dark";

  onMount(() => {
    systemColorScheme = window.matchMedia(darkMatcher).matches ? "dark" : "light";

    window.matchMedia(darkMatcher).addEventListener("change", onSystemColorSchemeChange);

    return () => {
      window.matchMedia(darkMatcher).removeEventListener("change", onSystemColorSchemeChange);
    };
  });

  let {
    primaryTextColor,
    secondaryTextColor,
    tertiaryTextColor,
    fontFamily,
    fontSize,
    primaryBackgroundColor,
    secondaryBackgroundColor,
    hoverBackgroundColor,
    headerFontFamily,
    headerFontSize,
    cellFontFamily,
    cellFontSize,
    scrollbarBackgroundColor,
    scrollbarPillColor,
    scrollbarLabelBackgroundColor,
    shadow,
    outlineColor,
    dimmedRowColor,
    rowScrollToColor,
    rowHoverColor,
  } = $derived(resolveTheme(theme, colorScheme));

  $effect(() => {});
</script>

<div
  class="style-wrapper table-defaults {colorScheme}"
  style:--user-primary-text-color={primaryTextColor}
  style:--user-secondary-text-color={secondaryTextColor}
  style:--user-tertiary-text-color={tertiaryTextColor}
  style:--user-font-family={fontFamily}
  style:--user-font-size={fontSize}
  style:--user-primary-bg={primaryBackgroundColor}
  style:--user-secondary-bg={secondaryBackgroundColor}
  style:--user-hover-bg={hoverBackgroundColor}
  style:--user-header-font-family={headerFontFamily}
  style:--user-header-font-size={headerFontSize}
  style:--user-cell-font-family={cellFontFamily}
  style:--user-cell-font-size={cellFontSize}
  style:--user-scrollbar-bg={scrollbarBackgroundColor}
  style:--user-scrollbar-pill-bg={scrollbarPillColor}
  style:--user-scrollbar-label-bg={scrollbarLabelBackgroundColor}
  style:--user-shadow={shadow}
  style:--user-outline-color={outlineColor}
  style:--user-dimmed-row-color={dimmedRowColor}
  style:--user-row-scroll-to-color={rowScrollToColor}
  style:--user-row-hover-color={rowHoverColor}
>
  {@render children()}
</div>

<style>
  .table-defaults.light {
    --default-primary-text-color: black;
    --default-secondary-text-color: gray;
    --default-tertiary-text-color: lightgray;
    --default-font-family: sans-serif;
    --default-font-size: 1rem;
    --default-primary-bg: white;
    --default-secondary-bg: rgb(246, 246, 247);
    --default-tertiary-bg: rgb(234, 234, 235);
    --default-hover-bg: rgba(0, 0, 0, 0.05);
    --default-scrollbar-bg: rgba(0, 0, 0, 0.05);
    --default-scrollbar-pill-bg: rgba(0, 0, 0, 0.5);
    --default-scrollbar-label-bg: rgba(255, 255, 255, 0.9);
    --default-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --default-outline-color: rgb(0 0 0 / 0.2);
    --default-dimmed-row-color: rgb(0 0 0 / 0.2);
    --default-row-scroll-to-color: rgb(202 225 255);
    --default-row-hover-color: rgb(220, 235, 255);
  }

  .table-defaults.dark {
    --default-primary-text-color: lightgray;
    --default-secondary-text-color: gray;
    --default-tertiary-text-color: dimgray;
    --default-font-family: sans-serif;
    --default-font-size: 1rem;
    --default-primary-bg: #060607;
    --default-secondary-bg: #161617;
    --default-hover-bg: rgba(255, 255, 255, 0.05);
    --default-scrollbar-bg: rgba(255, 255, 255, 0.05);
    --default-scrollbar-pill-bg: rgba(255, 255, 255, 0.5);
    --default-scrollbar-label-bg: rgba(0, 0, 0, 0.9);
    --default-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --default-outline-color: rgb(255 255 255 / 0.2);
    --default-dimmed-row-color: rgb(0 0 0 / 0.6);
    --default-row-scroll-to-color: rgb(1, 24, 106);
    --default-row-hover-color: rgb(0, 6, 35);
  }

  .style-wrapper {
    width: 100%;
    height: 100%;

    --primary-text-color: var(--user-primary-text-color, var(--default-primary-text-color));
    --secondary-text-color: var(--user-secondary-text-color, var(--default-secondary-text-color));
    --tertiary-text-color: var(--user-tertiary-text-color, var(--default-tertiary-text-color));
    --font-family: var(--user-font-family, var(--default-font-family));
    --font-size: var(--user-font-size, var(--default-font-size));
    --primary-bg: var(--user-primary-bg, var(--default-primary-bg));
    --secondary-bg: var(--user-secondary-bg, var(--default-secondary-bg));
    --tertiary-bg: var(--user-tertiarty-bg, var(--default-tertiary-bg));
    --hover-bg: var(--user-hover-bg, var(--default-hover-bg));

    --header-font-family: var(--user-header-font-family, var(--font-family));
    --header-font-size: var(--user-header-font-size, var(--font-size));

    --cell-font-family: var(--user-cell-font-family, var(--font-family));
    --cell-font-size: var(--user-cell-font-size, var(--font-size));

    --scrollbar-bg: var(--user-scrollbar-bg, var(--default-scrollbar-bg));
    --scrollbar-pill-bg: var(--user-scrollbar-pill-bg, var(--default-scrollbar-pill-bg));
    --scrollbar-label-bg: var(--user-scrollbar-label-bg, var(--default-scrollbar-label-bg));

    --shadow: var(--user-shadow, var(--default-shadow));
    --outline-color: var(--user-outline-color, var(--default-outline-color));

    --outline: 0.5px solid var(--outline-color);
    --dimmed-row-color: var(--user-dimmed-row-color, var(--default-dimmed-row-color));
    --row-scroll-to-color: var(--user-row-scroll-to-color, var(--default-row-scroll-to-color));
    --row-hover-color: var(--user-row-hover-color, var(--default-row-hover-color));
  }
</style>

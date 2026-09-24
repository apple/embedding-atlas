// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface ParallelCoordinatesViewTheme {
  /** Font family for axis tick labels and titles. */
  fontFamily: string;
  /** Font size for axis tick labels, in CSS px. */
  labelFontSize: number;
  /** Font size for axis titles, in CSS px. */
  titleFontSize: number;
  /** Color of the vertical axis lines. */
  axisLineColor: string;
  /** Color of the tick marks. */
  tickColor: string;
  /** Color of the tick labels. */
  labelColor: string;
  /** Color of the outline drawn behind the tick labels (defaults to the background color). */
  labelOutlineColor: string;
}

export type ParallelCoordinatesViewThemeConfig = Partial<ParallelCoordinatesViewTheme> & {
  /** Overrides for light mode. */
  light?: Partial<ParallelCoordinatesViewTheme>;
  /** Overrides for dark mode. */
  dark?: Partial<ParallelCoordinatesViewTheme>;
};

const defaultThemeConfig: { light: ParallelCoordinatesViewTheme; dark: ParallelCoordinatesViewTheme } = {
  light: {
    fontFamily: "system-ui,sans-serif",
    labelFontSize: 11,
    titleFontSize: 11,
    axisLineColor: "#000000",
    tickColor: "#000000",
    labelColor: "#333333",
    labelOutlineColor: "rgba(255,255,255,0.8)",
  },
  dark: {
    fontFamily: "system-ui,sans-serif",
    labelFontSize: 11,
    titleFontSize: 11,
    axisLineColor: "#ffffff",
    tickColor: "#ffffff",
    labelColor: "#cccccc",
    labelOutlineColor: "rgba(0,0,0,0.8)",
  },
};

export function resolveTheme(
  theme: ParallelCoordinatesViewThemeConfig | null | undefined,
  colorScheme: "light" | "dark",
): ParallelCoordinatesViewTheme {
  if (theme == null) {
    return defaultThemeConfig[colorScheme];
  }
  return { ...defaultThemeConfig[colorScheme], ...theme, ...(theme[colorScheme] != null ? theme[colorScheme] : {}) };
}

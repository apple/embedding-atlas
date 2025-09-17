// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface EmbeddingViewTheme {
  /** The font family for texts. */
  fontFamily: string;
  /** Color for cluster labels. */
  clusterLabelColor: string;
  /** Color for cluster labels' outline. */
  clusterLabelOutlineColor: string;
  /** Opacity for cluster labels. */
  clusterLabelOpacity: number;
  /** Whether to show the status bar at the bottom. */
  statusBar: boolean;
  /** Color for status bar text. */
  statusBarTextColor: string;
  /** Color for status bar background. */
  statusBarBackgroundColor: string;
  /** Branding link. */
  brandingLink: { text: string; href: string } | null;
}

export type ThemeConfig = Partial<EmbeddingViewTheme> & {
  /** Overrides for light mode. */
  dark?: Partial<EmbeddingViewTheme>;
  /** Overrides for dark mode. */
  light?: Partial<EmbeddingViewTheme>;
};

const defaultThemeConfig: { light: EmbeddingViewTheme; dark: EmbeddingViewTheme } = {
  light: {
    fontFamily: "system-ui,sans-serif",
    clusterLabelColor: "#000",
    clusterLabelOutlineColor: "rgba(255,255,255,0.8)",
    clusterLabelOpacity: 0.8,
    statusBar: true,
    statusBarTextColor: "#525252",
    statusBarBackgroundColor: "rgba(255,255,255,0.9)",
    brandingLink: { text: "Embedding Atlas", href: "https://apple.github.io/embedding-atlas" },
  },
  dark: {
    fontFamily: "system-ui,sans-serif",
    clusterLabelColor: "#ccc",
    clusterLabelOutlineColor: "rgba(0,0,0,0.8)",
    clusterLabelOpacity: 0.8,
    statusBar: true,
    statusBarTextColor: "#d9d9d9",
    statusBarBackgroundColor: "rgba(0,0,0,0.9)",
    brandingLink: { text: "Embedding Atlas", href: "https://apple.github.io/embedding-atlas" },
  },
};

export function resolveTheme(theme: ThemeConfig | null, colorScheme: "light" | "dark"): EmbeddingViewTheme {
  if (theme == null) {
    return defaultThemeConfig[colorScheme];
  } else {
    return { ...defaultThemeConfig[colorScheme], ...theme, ...(theme[colorScheme] != null ? theme[colorScheme] : {}) };
  }
}

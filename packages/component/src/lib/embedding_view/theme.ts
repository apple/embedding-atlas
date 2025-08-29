// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface ThemeConfig {
  fontFamily: string;
  clusterLabelColor: string;
  clusterLabelOutlineColor: string;
  clusterLabelOpacity: number;
  statusBar: boolean;
  statusBarTextColor: string;
  statusBarBackgroundColor: string;
  brandingLink: { text: string; href: string } | null;
}

export type Theme = Partial<ThemeConfig> & {
  /** Overrides for light mode. */
  dark?: Partial<ThemeConfig>;
  /** Overrides for dark mode. */
  light?: Partial<ThemeConfig>;
};

const defaultThemeConfig: { light: ThemeConfig; dark: ThemeConfig } = {
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

export function resolveTheme(theme: Theme | null, colorScheme: "light" | "dark"): ThemeConfig {
  if (theme == null) {
    return defaultThemeConfig[colorScheme];
  } else {
    return { ...defaultThemeConfig[colorScheme], ...theme, ...(theme[colorScheme] != null ? theme[colorScheme] : {}) };
  }
}

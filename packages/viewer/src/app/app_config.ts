// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/** Type for window.EMBEDDING_ATLAS_CONFIG. Set this in index.html to configure the application. */
export interface AppConfig {
  home: "file-viewer" | "backend-viewer";
}

export function resolveAppConfig(): AppConfig {
  let config: Partial<AppConfig> = {};
  if (typeof window != undefined && (window as any).EMBEDDING_ATLAS_CONFIG != undefined) {
    config = (window as any).EMBEDDING_ATLAS_CONFIG;
  }
  return {
    home: "backend-viewer",
    ...config,
  };
}

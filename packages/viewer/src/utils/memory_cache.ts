// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Cache } from "../api.js";

/** In-memory cache for JSON-serializable intermediate results such as label computation. */
export class MemoryCache implements Cache {
  private entries = new Map<string, any>();

  async get(key: string): Promise<any | null> {
    return this.entries.get(key) ?? null;
  }

  async set(key: string, value: any): Promise<void> {
    this.entries.set(key, value);
  }
}

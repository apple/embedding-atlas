// Copyright (c) 2026 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { extensionRepositoryUrl } from "../src/utils/duckdb.js";

describe("extensionRepositoryUrl", () => {
  test("resolves relative to directory page URL", () => {
    expect(extensionRepositoryUrl("https://example.com/viewer/")).toBe("https://example.com/viewer/duckdb-wasm");
  });

  test("resolves relative to HTML entry file", () => {
    expect(extensionRepositoryUrl("https://example.com/viewer/index.html")).toBe(
      "https://example.com/viewer/duckdb-wasm",
    );
  });

  test("ignores slashes in query parameters and hash fragments", () => {
    expect(extensionRepositoryUrl("https://example.com/viewer/?dataset=apple/atlas&path=a/b#state=x/y")).toBe(
      "https://example.com/viewer/duckdb-wasm",
    );
    expect(extensionRepositoryUrl("https://example.com/viewer/index.html?dataset=apple/atlas#s/1")).toBe(
      "https://example.com/viewer/duckdb-wasm",
    );
  });

  test("handles bare origin without trailing slash", () => {
    expect(extensionRepositoryUrl("https://example.com")).toBe("https://example.com/duckdb-wasm");
  });
});

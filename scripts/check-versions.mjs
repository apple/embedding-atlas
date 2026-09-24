#!/usr/bin/env node
// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// Check that all version numbers across the project are consistent.
// Usage: node scripts/check-versions.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function read(path) {
  return readFileSync(join(root, path), "utf-8");
}

function match(path, regex) {
  const m = read(path).match(regex);
  return m ? m[1] : null;
}

const sources = [
  {
    file: "packages/embedding-atlas/package.json",
    get: (path) => JSON.parse(read(path)).version,
  },
  {
    file: "package-lock.json",
    get: (path) => JSON.parse(read(path)).packages?.["packages/embedding-atlas"]?.version,
  },
  {
    file: "packages/backend/pyproject.toml",
    get: (path) => match(path, /^\[project\][^[]*?^version\s*=\s*"([^"]+)"/m),
  },
  {
    file: "packages/backend/embedding_atlas/version.py",
    get: (path) => match(path, /^__version__\s*=\s*"([^"]+)"/m),
  },
  {
    file: "uv.lock",
    get: (path) => match(path, /^\[\[package\]\]\nname = "embedding-atlas"\nversion = "([^"]+)"/m),
  },
  {
    file: "packages/viewer/src/constants.ts",
    get: (path) => match(path, /EMBEDDING_ATLAS_VERSION\s*=\s*"([^"]+)"/),
  },
  {
    file: "packages/docs/examples/examples.data.ts",
    get: (path) => match(path, /const VERSION\s*=\s*"([^"]+)"/),
  },
];

const results = sources.map(({ file, get }) => {
  let version = null;
  try {
    version = get(file) ?? null;
  } catch (e) {
    console.error(`Error reading ${file}: ${e.message}`);
  }
  return { file, version };
});

const width = Math.max(...results.map((r) => r.file.length));
for (const { file, version } of results) {
  console.log(`${file.padEnd(width)}  ${version ?? "(not found)"}`);
}

const versions = new Set(results.map((r) => r.version));
if (versions.size !== 1 || versions.has(null)) {
  console.error("\nError: version numbers are inconsistent.");
  process.exit(1);
}

console.log(`\nAll versions match: ${[...versions][0]}`);

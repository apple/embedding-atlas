// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

const READONLY_FIRST_WORD = new Set([
  "SELECT",
  "WITH",
  "VALUES",
  "TABLE",
  "DESCRIBE",
  "DESC",
  "SHOW",
  "EXPLAIN",
  "SUMMARIZE",
  "PIVOT",
]);

const WRITE_KEYWORDS =
  /\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|DETACH|INSTALL|LOAD|CALL|PRAGMA|SET|VACUUM|CHECKPOINT|TRUNCATE|GRANT|REVOKE|USE)\b/i;

function stripStringsAndComments(sql: string): string {
  // Remove '...', "...", `...`, /* ... */, -- ... to avoid false positives.
  return sql
    .replace(/'(?:[^']|'')*'/g, " ")
    .replace(/"(?:[^"\\]|\\.)*"/g, " ")
    .replace(/`(?:[^`\\]|\\.)*`/g, " ")
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/--[^\n]*/g, " ");
}

export function isReadonlyQuery(query: string): boolean {
  let cleaned = stripStringsAndComments(query).trim();
  // Allow a single trailing semicolon; reject stacked statements.
  cleaned = cleaned.replace(/;+\s*$/, "").trim();
  if (cleaned === "" || cleaned.includes(";")) {
    return false;
  }
  let firstWord = cleaned
    .split(/\s+/, 1)[0]
    ?.replace(/[^A-Za-z]/g, "")
    .toUpperCase();
  if (firstWord == null || !READONLY_FIRST_WORD.has(firstWord)) {
    return false;
  }
  if (WRITE_KEYWORDS.test(cleaned)) {
    return false;
  }
  return true;
}

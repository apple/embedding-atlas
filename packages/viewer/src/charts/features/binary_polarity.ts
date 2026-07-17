// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Heuristic to order a binary classification's two labels as [negative, positive],
 * where "positive" means the semantically affirmative / desirable outcome.
 *
 * Three-case decision:
 *   1. Both labels are boolean-like (true/false, 1/0, yes/no, …) and opposite
 *      polarity → defer to the column-name valence.
 *   2. Both labels are semantic (safe/unsafe, pass/fail, …) and opposite
 *      polarity → trust the labels themselves.
 *   3. Otherwise (same polarity, mixed kinds, unknown words, …) → return
 *      `undefined` to signal we can't decide.
 *
 * In case 1 the column-name valence flips the default (boolean-negative first):
 *   - positive-valenced column (e.g., `is_safe`)   → [false, true]
 *   - negative-valenced column (e.g., `is_unsafe`) → [true,  false]
 *   - no valence word found                        → undefined
 *
 * Comparisons are done on a normalized form (lowercase + trim); the original
 * label strings are returned unchanged.
 */

const BOOLEAN_POSITIVE = new Set(["true", "t", "1", "yes", "y", "on"]);
const BOOLEAN_NEGATIVE = new Set(["false", "f", "0", "no", "n", "off"]);

const SEMANTIC_POSITIVE = new Set([
  "pass",
  "passed",
  "passing",
  "ok",
  "okay",
  "success",
  "successful",
  "succeeded",
  "valid",
  "safe",
  "good",
  "correct",
  "accept",
  "accepted",
  "approved",
  "positive",
  "pos",
  "healthy",
  "active",
  "right",
  "clean",
  "complete",
  "completed",
  "allowed",
]);

const SEMANTIC_NEGATIVE = new Set([
  "fail",
  "failed",
  "failing",
  "failure",
  "error",
  "errored",
  "invalid",
  "unsafe",
  "bad",
  "incorrect",
  "reject",
  "rejected",
  "denied",
  "negative",
  "neg",
  "harmful",
  "harm",
  "wrong",
  "broken",
  "toxic",
  "malicious",
  "dangerous",
  "hazardous",
  "risky",
  "buggy",
  "bug",
  "faulty",
  "defective",
  "blocked",
  "disallowed",
]);

// Boilerplate auxiliaries we strip when reading column-name valence.
const COLUMN_PREFIX_TOKENS = new Set([
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "has",
  "have",
  "had",
  "will",
  "would",
  "should",
  "could",
  "do",
  "does",
  "did",
]);

type LabelKind = "boolean" | "semantic" | "unknown";
type Polarity = "positive" | "negative";

interface LabelInfo {
  kind: LabelKind;
  polarity: Polarity | "unknown";
}

function classifyLabel(label: string): LabelInfo {
  const norm = label.trim().toLowerCase();
  if (BOOLEAN_POSITIVE.has(norm)) {
    return { kind: "boolean", polarity: "positive" };
  }
  if (BOOLEAN_NEGATIVE.has(norm)) {
    return { kind: "boolean", polarity: "negative" };
  }
  if (SEMANTIC_POSITIVE.has(norm)) {
    return { kind: "semantic", polarity: "positive" };
  }
  if (SEMANTIC_NEGATIVE.has(norm)) {
    return { kind: "semantic", polarity: "negative" };
  }
  return { kind: "unknown", polarity: "unknown" };
}

function tokenize(name: string): string[] {
  return name
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .split(/[_\s\-./]+/)
    .filter((t) => t.length > 0);
}

function columnPolarity(columnName: string): Polarity | "unknown" {
  const tokens = tokenize(columnName).filter((t) => !COLUMN_PREFIX_TOKENS.has(t));
  for (const tok of tokens) {
    if (SEMANTIC_NEGATIVE.has(tok)) {
      return "negative";
    }
    if (SEMANTIC_POSITIVE.has(tok)) {
      return "positive";
    }
  }
  return "unknown";
}

/**
 * Order two labels as [negative, positive] using the column name and label
 * strings. Returns `undefined` when the signal is too weak to commit.
 */
export function inferBinaryPolarity(columnName: string, labels: [string, string]): [string, string] | undefined {
  const [labelA, labelB] = labels;
  const a = classifyLabel(labelA);
  const b = classifyLabel(labelB);

  // Case 2: both semantic, opposite polarity → trust the labels.
  if (a.kind === "semantic" && b.kind === "semantic" && a.polarity !== b.polarity) {
    return a.polarity === "negative" ? [labelA, labelB] : [labelB, labelA];
  }

  // Case 1: both boolean-like, opposite polarity → defer to the column name.
  if (a.kind === "boolean" && b.kind === "boolean" && a.polarity !== b.polarity) {
    const valence = columnPolarity(columnName);
    if (valence === "unknown") {
      return undefined;
    }
    // negativeFirst orders the labels as [boolean-negative, boolean-positive].
    const negativeFirst: [string, string] = a.polarity === "negative" ? [labelA, labelB] : [labelB, labelA];
    return valence === "negative" ? [negativeFirst[1], negativeFirst[0]] : negativeFirst;
  }

  // Anything else (same polarity, mixed kinds, both unknown, …): not confident.
  return undefined;
}

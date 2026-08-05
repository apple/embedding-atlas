// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * A query parsed into exact phrases and the remaining free text.
 *
 * Double-quoted runs become exact phrases, everything outside the quotes is
 * collected as free text. For example `"aldi" store` parses to one phrase
 * (`aldi`) plus the free text `store`.
 */
export interface ParsedQuery {
  phrases: string[];
  freeText: string;
}

/**
 * Parse a query into exact phrases and free text.
 *
 * A double-quoted run (e.g. `"aldi"`) means the user wants an exact,
 * case-insensitive substring match instead of the default fuzzy token search.
 * The default encoder maps similar-looking words to the same token (for example
 * "aldi" and "aldea"), which is great for fuzzy recall but surfaces unwanted
 * matches when the user knows exactly what they are looking for. Quoting opts
 * out of that behavior for the quoted run while leaving any unquoted words on
 * the fuzzy path, so `"aldi" store` requires the exact substring "aldi" and
 * fuzzy-matches "store".
 *
 * Empty quotes (`""`) contribute no phrase. An unterminated trailing quote is
 * treated as a literal character of the free text so a half-typed query still
 * searches.
 */
export function parseQuery(query: string): ParsedQuery {
  let phrases: string[] = [];
  let freeText: string[] = [];
  let rest = query;

  while (true) {
    let open = rest.indexOf('"');
    if (open < 0) {
      freeText.push(rest);
      break;
    }
    let close = rest.indexOf('"', open + 1);
    if (close < 0) {
      // No closing quote, keep the remainder as free text verbatim.
      freeText.push(rest);
      break;
    }
    freeText.push(rest.slice(0, open));
    let inner = rest.slice(open + 1, close);
    if (inner.length > 0) {
      phrases.push(inner);
    }
    rest = rest.slice(close + 1);
  }

  return { phrases, freeText: freeText.join(" ").trim() };
}

/**
 * Escape a phrase for use as the pattern of a `LIKE ... ESCAPE '\'` comparison.
 *
 * The phrase is user input, so the `LIKE` wildcards `%` and `_` must be treated
 * as literal characters, as must the escape character itself.
 */
export function escapeLikePattern(phrase: string): string {
  return phrase.replace(/[\\%_]/g, (c) => `\\${c}`);
}

import type { SQLField, SQLTable } from "../spec/spec.js";

export interface FeaturesListSpec {
  type: "features-list";
  title?: string;

  data: {
    /**
     * The features column. Either:
     * - string[]: a list of feature strings. Every feature is assigned source "default".
     * - struct[]: a list of structs, one per feature, each `{ feature: string, source?: string, index?: number }`.
     *   - feature: The feature string (required).
     *   - source: The source of the feature, could be a column name, or "prompt", "response", "tool_result", etc.
     *     If missing/NULL, defaults to "default".
     *   - index: For multi-turn conversations, the turn index. Currently unused.
     *
     * The column type (string[] vs struct[]) is detected at runtime.
     */
    features: SQLField;

    /** If specified, show a prediction view where features are scored by how well they predict an outcome. */
    predict?: SQLField;
  };

  /** Metadata for features. */
  metadata?: {
    /** The metadata table */
    table: SQLTable;

    /** The feature name column, should be string. */
    feature: SQLField;

    /** The topics column if available, should be string[]. A feature can be assigned any number of topics. Topics may overlap. */
    topics?: SQLField;
  };

  /** Sort order of the feature list */
  sort?:
    | "frequency-ascending"
    | "frequency-descending"
    | "predictiveness-ascending"
    | "predictiveness-descending"
    | "source-skew-ascending"
    | "source-skew-descending"
    | "alphabetical";

  /**
   * Limit the number of features shown, default 100.
   * Limit is applied after search and sort.
   */
  limit?: number;

  /**
   * Show features only from the specified sources if specified. Otherwise show all features.
   * All counts and predictiveness should be computed from the specified sources.
   */
  sources?: string[];

  /** Group the features list by topics if specified. Note that a feature may appear in more than one topic. */
  groupBy?: "topics";

  /** Segment bars by source if specified. */
  segmentBy?: "source";
}

export interface FeaturesListState {
  /** Free-text search filter applied to feature names */
  search?: string;

  /** Features participating in the AND cross-filter selection (a data row matches iff its features column contains all of them). */
  selected?: string[];

  /** Pinned features (presentation only); shown in a sticky section above the list, always visible regardless of search/sort/limit. */
  pinned?: string[];
}

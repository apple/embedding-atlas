// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { RowID } from "../charts/chart.js";

/** Represents a qualitative code that can be applied to data points */
export interface Code {
  /** Unique identifier for the code */
  id: string;
  /** Display name of the code */
  name: string;
  /** Optional description of what this code means */
  description?: string;
  /** Color for visual identification (hex or CSS color) */
  color: string;
  /** Parent code ID for hierarchical organization */
  parentId?: string | null;
  /** Coding level: 1=open coding, 2=axial coding, 3=selective coding */
  level: 1 | 2 | 3;
  /** When this code was created */
  createdAt: number;
  /** Who created this code (for team coding) */
  createdBy?: string;
  /** Number of data points with this code applied */
  frequency: number;
  /** For ANT analysis: actor type classification */
  antActorType?: "human" | "non-human" | "hybrid";
}

/** Represents a memo attached to codes or data points */
export interface Memo {
  /** Unique identifier for the memo */
  id: string;
  /** Rich text content of the memo */
  content: string;
  /** Codes this memo discusses */
  linkedCodes: string[];
  /** Data points this memo references */
  linkedDataPoints: RowID[];
  /** Type of memo */
  memoType: "theoretical" | "methodological" | "observational";
  /** When this memo was created */
  createdAt: number;
  /** Who created this memo */
  createdBy?: string;
  /** Tags for organization */
  tags: string[];
}

/** Represents a coding event in the audit trail */
export interface CodingEvent {
  /** When this event occurred */
  timestamp: number;
  /** Type of action */
  action: "apply" | "remove" | "create" | "merge" | "split" | "update";
  /** The code involved */
  codeId: string;
  /** Data points affected */
  dataPointIds: RowID[];
  /** Who performed this action */
  coder?: string;
  /** Additional notes about the action */
  notes?: string;
}

/** Represents the application of a code to a data point */
export interface CodeApplication {
  /** The code ID */
  codeId: string;
  /** The data point ID */
  dataPointId: RowID;
  /** When this code was applied */
  appliedAt: number;
  /** Who applied this code */
  appliedBy?: string;
  /** Optional notes about why this code was applied */
  notes?: string;
}

/** State for the entire coding system */
export interface CodingState {
  /** All codes in the system */
  codes: Record<string, Code>;
  /** All memos in the system */
  memos: Record<string, Memo>;
  /** Mapping of data point IDs to their applied code IDs */
  codeApplications: Record<string, string[]>;
  /** Audit trail of coding events */
  codingEvents: CodingEvent[];
  /** Settings for the coding system */
  settings: CodingSettings;
}

/** Settings for the coding system */
export interface CodingSettings {
  /** Whether to show code descriptions in tooltips */
  showDescriptions: boolean;
  /** Whether to show code frequencies */
  showFrequencies: boolean;
  /** Whether to show code hierarchy */
  showHierarchy: boolean;
  /** Columns to display in the details modal */
  visibleMetadataColumns: string[];
  /** Whether to enable multi-coder mode */
  multiCoderMode: boolean;
  /** Current coder name/ID */
  currentCoder?: string;
}

/** Default settings for the coding system */
export const defaultCodingSettings: CodingSettings = {
  showDescriptions: true,
  showFrequencies: true,
  showHierarchy: true,
  visibleMetadataColumns: [],
  multiCoderMode: false,
};

/** Default colors for codes */
export const codeColors = [
  "#ef4444", // red
  "#f97316", // orange
  "#eab308", // yellow
  "#22c55e", // green
  "#14b8a6", // teal
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#6366f1", // indigo
  "#06b6d4", // cyan
];

/** Generate a unique ID */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/** Create a new empty coding state */
export function createEmptyCodingState(): CodingState {
  return {
    codes: {},
    memos: {},
    codeApplications: {},
    codingEvents: [],
    settings: { ...defaultCodingSettings },
  };
}

/** Get all child codes of a parent code */
export function getChildCodes(codes: Record<string, Code>, parentId: string | null): Code[] {
  return Object.values(codes).filter((code) => code.parentId === parentId);
}

/** Get the full path of a code (including ancestors) */
export function getCodePath(codes: Record<string, Code>, codeId: string): Code[] {
  const path: Code[] = [];
  let current = codes[codeId];
  while (current) {
    path.unshift(current);
    current = current.parentId ? codes[current.parentId] : undefined!;
  }
  return path;
}

/** Get codes applied to a data point */
export function getCodesForDataPoint(
  state: CodingState,
  dataPointId: RowID
): Code[] {
  const codeIds = state.codeApplications[String(dataPointId)] ?? [];
  return codeIds.map((id) => state.codes[id]).filter(Boolean);
}

/** Check if a code is applied to a data point */
export function isCodeApplied(
  state: CodingState,
  codeId: string,
  dataPointId: RowID
): boolean {
  const codeIds = state.codeApplications[String(dataPointId)] ?? [];
  return codeIds.includes(codeId);
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Code relationship types and utilities for qualitative coding.
 * Supports various relationship types for grounded theory analysis.
 */

import type { Code } from "./types.js";

/** Types of relationships between codes */
export type RelationshipType =
  | "is_a"           // Hierarchical: child IS A type of parent
  | "part_of"        // Composition: code is PART OF another
  | "causes"         // Causal: one code CAUSES another
  | "influences"     // Weaker causal: one code INFLUENCES another
  | "contradicts"    // Dialectic: codes CONTRADICT each other
  | "co_occurs"      // Co-occurrence: codes often appear together
  | "precedes"       // Temporal: one code PRECEDES another
  | "associated"     // General association

/** A relationship between two codes */
export interface CodeRelationship {
  id: string;
  sourceCodeId: string;
  targetCodeId: string;
  type: RelationshipType;
  description?: string;
  strength?: number; // 1-5 for relationship strength
  createdAt: number;
  createdBy?: string;
}

/** Relationship type metadata */
export const relationshipTypes: Record<RelationshipType, {
  label: string;
  description: string;
  directed: boolean;
  color: string;
}> = {
  is_a: {
    label: "Is A",
    description: "Hierarchical relationship (child is a type of parent)",
    directed: true,
    color: "#3b82f6", // blue
  },
  part_of: {
    label: "Part Of",
    description: "Composition relationship (part belongs to whole)",
    directed: true,
    color: "#8b5cf6", // purple
  },
  causes: {
    label: "Causes",
    description: "Strong causal relationship",
    directed: true,
    color: "#ef4444", // red
  },
  influences: {
    label: "Influences",
    description: "Weak causal or influential relationship",
    directed: true,
    color: "#f97316", // orange
  },
  contradicts: {
    label: "Contradicts",
    description: "Opposing or contradicting concepts",
    directed: false,
    color: "#ec4899", // pink
  },
  co_occurs: {
    label: "Co-occurs",
    description: "Codes that frequently appear together",
    directed: false,
    color: "#22c55e", // green
  },
  precedes: {
    label: "Precedes",
    description: "Temporal sequence relationship",
    directed: true,
    color: "#06b6d4", // cyan
  },
  associated: {
    label: "Associated",
    description: "General association without specific type",
    directed: false,
    color: "#64748b", // gray
  },
};

/**
 * Get all relationships for a specific code
 */
export function getCodeRelationships(
  relationships: CodeRelationship[],
  codeId: string
): {
  outgoing: CodeRelationship[];
  incoming: CodeRelationship[];
  bidirectional: CodeRelationship[];
} {
  const outgoing: CodeRelationship[] = [];
  const incoming: CodeRelationship[] = [];
  const bidirectional: CodeRelationship[] = [];

  for (const rel of relationships) {
    const typeInfo = relationshipTypes[rel.type];

    if (rel.sourceCodeId === codeId) {
      if (typeInfo.directed) {
        outgoing.push(rel);
      } else {
        bidirectional.push(rel);
      }
    } else if (rel.targetCodeId === codeId) {
      if (typeInfo.directed) {
        incoming.push(rel);
      } else if (!bidirectional.some(r => r.id === rel.id)) {
        bidirectional.push(rel);
      }
    }
  }

  return { outgoing, incoming, bidirectional };
}

/**
 * Find codes that frequently co-occur in the data
 */
export function findCoOccurrences(
  codeApplications: Record<string, string[]>,
  minOccurrences: number = 3
): Array<{ codeA: string; codeB: string; count: number }> {
  const coOccurrences: Map<string, number> = new Map();

  // Count co-occurrences for each data point
  for (const codeIds of Object.values(codeApplications)) {
    if (codeIds.length < 2) continue;

    // Count pairs
    for (let i = 0; i < codeIds.length; i++) {
      for (let j = i + 1; j < codeIds.length; j++) {
        const pair = [codeIds[i], codeIds[j]].sort().join("|");
        coOccurrences.set(pair, (coOccurrences.get(pair) || 0) + 1);
      }
    }
  }

  // Convert to array and filter
  const result: Array<{ codeA: string; codeB: string; count: number }> = [];

  for (const [pair, count] of coOccurrences) {
    if (count >= minOccurrences) {
      const [codeA, codeB] = pair.split("|");
      result.push({ codeA, codeB, count });
    }
  }

  return result.sort((a, b) => b.count - a.count);
}

/**
 * Build a hierarchy tree from codes with parent relationships
 */
export function buildCodeHierarchy(
  codes: Record<string, Code>
): Map<string | null, Code[]> {
  const hierarchy = new Map<string | null, Code[]>();

  for (const code of Object.values(codes)) {
    const parentId = code.parentId ?? null;
    if (!hierarchy.has(parentId)) {
      hierarchy.set(parentId, []);
    }
    hierarchy.get(parentId)!.push(code);
  }

  // Sort children by name
  for (const children of hierarchy.values()) {
    children.sort((a, b) => a.name.localeCompare(b.name));
  }

  return hierarchy;
}

/**
 * Get the depth of a code in the hierarchy
 */
export function getCodeDepth(
  codes: Record<string, Code>,
  codeId: string
): number {
  let depth = 0;
  let current = codes[codeId];

  while (current?.parentId) {
    depth++;
    current = codes[current.parentId];
    if (depth > 10) break; // Prevent infinite loops
  }

  return depth;
}

/**
 * Get all ancestor codes for a given code
 */
export function getAncestors(
  codes: Record<string, Code>,
  codeId: string
): Code[] {
  const ancestors: Code[] = [];
  let current = codes[codeId];

  while (current?.parentId) {
    const parent = codes[current.parentId];
    if (parent) {
      ancestors.push(parent);
      current = parent;
    } else {
      break;
    }
    if (ancestors.length > 10) break; // Prevent infinite loops
  }

  return ancestors;
}

/**
 * Get all descendant codes for a given code
 */
export function getDescendants(
  codes: Record<string, Code>,
  codeId: string
): Code[] {
  const descendants: Code[] = [];
  const queue = [codeId];

  while (queue.length > 0) {
    const currentId = queue.shift()!;
    const children = Object.values(codes).filter(c => c.parentId === currentId);

    for (const child of children) {
      descendants.push(child);
      queue.push(child.id);
    }

    if (descendants.length > 100) break; // Prevent runaway
  }

  return descendants;
}

/**
 * Generate a visual representation of code relationships (for export)
 */
export function generateRelationshipDiagram(
  codes: Record<string, Code>,
  relationships: CodeRelationship[]
): string {
  const lines: string[] = [];
  lines.push("# Code Relationship Diagram");
  lines.push("");
  lines.push("```mermaid");
  lines.push("graph TD");

  // Add codes as nodes
  for (const code of Object.values(codes)) {
    const label = code.name.replace(/"/g, "'");
    lines.push(`  ${code.id}["${label}"]`);
  }

  lines.push("");

  // Add relationships as edges
  for (const rel of relationships) {
    const typeInfo = relationshipTypes[rel.type];
    const arrow = typeInfo.directed ? "-->" : "---";
    const label = typeInfo.label;
    lines.push(`  ${rel.sourceCodeId} ${arrow}|${label}| ${rel.targetCodeId}`);
  }

  lines.push("```");
  lines.push("");

  return lines.join("\n");
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { RowID } from "../charts/chart.js";
import {
  type Code,
  type CodingEvent,
  type CodingState,
  type Memo,
  codeColors,
  createEmptyCodingState,
  generateId,
} from "./types.js";

/** Create a reactive coding store */
export function createCodingStore(initialState?: Partial<CodingState>) {
  let state = $state<CodingState>({
    ...createEmptyCodingState(),
    ...initialState,
  });

  function addCodingEvent(
    action: CodingEvent["action"],
    codeId: string,
    dataPointIds: RowID[],
    notes?: string
  ) {
    const event: CodingEvent = {
      timestamp: Date.now(),
      action,
      codeId,
      dataPointIds,
      coder: state.settings.currentCoder,
      notes,
    };
    state.codingEvents = [...state.codingEvents, event];
  }

  function updateCodeFrequencies() {
    const frequencies: Record<string, number> = {};
    for (const codeIds of Object.values(state.codeApplications)) {
      for (const codeId of codeIds) {
        frequencies[codeId] = (frequencies[codeId] ?? 0) + 1;
      }
    }
    for (const code of Object.values(state.codes)) {
      code.frequency = frequencies[code.id] ?? 0;
    }
  }

  return {
    get state() {
      return state;
    },

    get codes() {
      return state.codes;
    },

    get memos() {
      return state.memos;
    },

    get codeApplications() {
      return state.codeApplications;
    },

    get settings() {
      return state.settings;
    },

    /** Create a new code */
    createCode(
      name: string,
      options?: {
        description?: string;
        color?: string;
        parentId?: string | null;
        level?: 1 | 2 | 3;
        antActorType?: "human" | "non-human" | "hybrid";
      }
    ): Code {
      const id = generateId();
      const existingColors = Object.values(state.codes).map((c) => c.color);
      const availableColor =
        codeColors.find((c) => !existingColors.includes(c)) ?? codeColors[0];

      const code: Code = {
        id,
        name,
        description: options?.description,
        color: options?.color ?? availableColor,
        parentId: options?.parentId ?? null,
        level: options?.level ?? 1,
        createdAt: Date.now(),
        createdBy: state.settings.currentCoder,
        frequency: 0,
        antActorType: options?.antActorType,
      };

      state.codes = { ...state.codes, [id]: code };
      addCodingEvent("create", id, []);
      return code;
    },

    /** Update an existing code */
    updateCode(id: string, updates: Partial<Omit<Code, "id" | "createdAt">>) {
      const existing = state.codes[id];
      if (!existing) return;

      state.codes = {
        ...state.codes,
        [id]: { ...existing, ...updates },
      };
      addCodingEvent("update", id, [], `Updated: ${Object.keys(updates).join(", ")}`);
    },

    /** Delete a code */
    deleteCode(id: string) {
      const { [id]: removed, ...rest } = state.codes;
      state.codes = rest;

      // Remove applications of this code
      const newApplications = { ...state.codeApplications };
      for (const [dpId, codeIds] of Object.entries(newApplications)) {
        newApplications[dpId] = codeIds.filter((cid) => cid !== id);
        if (newApplications[dpId].length === 0) {
          delete newApplications[dpId];
        }
      }
      state.codeApplications = newApplications;

      // Update children to have no parent
      for (const code of Object.values(state.codes)) {
        if (code.parentId === id) {
          code.parentId = null;
        }
      }
    },

    /** Apply a code to one or more data points */
    applyCode(codeId: string, dataPointIds: RowID[]) {
      const newApplications = { ...state.codeApplications };

      for (const dpId of dataPointIds) {
        const key = String(dpId);
        const existing = newApplications[key] ?? [];
        if (!existing.includes(codeId)) {
          newApplications[key] = [...existing, codeId];
        }
      }

      state.codeApplications = newApplications;
      updateCodeFrequencies();
      addCodingEvent("apply", codeId, dataPointIds);
    },

    /** Remove a code from one or more data points */
    removeCode(codeId: string, dataPointIds: RowID[]) {
      const newApplications = { ...state.codeApplications };

      for (const dpId of dataPointIds) {
        const key = String(dpId);
        const existing = newApplications[key] ?? [];
        newApplications[key] = existing.filter((id) => id !== codeId);
        if (newApplications[key].length === 0) {
          delete newApplications[key];
        }
      }

      state.codeApplications = newApplications;
      updateCodeFrequencies();
      addCodingEvent("remove", codeId, dataPointIds);
    },

    /** Toggle a code on a data point */
    toggleCode(codeId: string, dataPointId: RowID) {
      const key = String(dataPointId);
      const existing = state.codeApplications[key] ?? [];
      if (existing.includes(codeId)) {
        this.removeCode(codeId, [dataPointId]);
      } else {
        this.applyCode(codeId, [dataPointId]);
      }
    },

    /** Get codes applied to a data point */
    getCodesForDataPoint(dataPointId: RowID): Code[] {
      const codeIds = state.codeApplications[String(dataPointId)] ?? [];
      return codeIds.map((id) => state.codes[id]).filter(Boolean);
    },

    /** Check if a code is applied to a data point */
    isCodeApplied(codeId: string, dataPointId: RowID): boolean {
      const codeIds = state.codeApplications[String(dataPointId)] ?? [];
      return codeIds.includes(codeId);
    },

    /** Get root-level codes (no parent) */
    getRootCodes(): Code[] {
      return Object.values(state.codes).filter((c) => !c.parentId);
    },

    /** Get child codes of a parent */
    getChildCodes(parentId: string): Code[] {
      return Object.values(state.codes).filter((c) => c.parentId === parentId);
    },

    /** Create a new memo */
    createMemo(
      content: string,
      options?: {
        linkedCodes?: string[];
        linkedDataPoints?: RowID[];
        memoType?: "theoretical" | "methodological" | "observational";
        tags?: string[];
      }
    ): Memo {
      const id = generateId();
      const memo: Memo = {
        id,
        content,
        linkedCodes: options?.linkedCodes ?? [],
        linkedDataPoints: options?.linkedDataPoints ?? [],
        memoType: options?.memoType ?? "observational",
        createdAt: Date.now(),
        createdBy: state.settings.currentCoder,
        tags: options?.tags ?? [],
      };

      state.memos = { ...state.memos, [id]: memo };
      return memo;
    },

    /** Update a memo */
    updateMemo(id: string, updates: Partial<Omit<Memo, "id" | "createdAt">>) {
      const existing = state.memos[id];
      if (!existing) return;

      state.memos = {
        ...state.memos,
        [id]: { ...existing, ...updates },
      };
    },

    /** Delete a memo */
    deleteMemo(id: string) {
      const { [id]: removed, ...rest } = state.memos;
      state.memos = rest;
    },

    /** Get memos for a data point */
    getMemosForDataPoint(dataPointId: RowID): Memo[] {
      return Object.values(state.memos).filter((m) =>
        m.linkedDataPoints.includes(dataPointId)
      );
    },

    /** Update settings */
    updateSettings(updates: Partial<typeof state.settings>) {
      state.settings = { ...state.settings, ...updates };
    },

    /** Export state as JSON */
    exportState(): CodingState {
      return JSON.parse(JSON.stringify(state));
    },

    /** Import state from JSON */
    importState(newState: CodingState) {
      state = { ...createEmptyCodingState(), ...newState };
    },

    /** Reset to empty state */
    reset() {
      state = createEmptyCodingState();
    },
  };
}

export type CodingStore = ReturnType<typeof createCodingStore>;

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export class HistoryManager<State> {
  private _before: State[];
  private _after: State[];
  private _current: State | undefined;
  private _pending: State | undefined;
  private _timeout: ReturnType<typeof setTimeout> | undefined;
  private _debounceMs: number;
  private _maxSize: number;

  constructor(options: { debounce?: number; maxSize?: number } = {}) {
    this._before = [];
    this._after = [];
    this._current = undefined;
    this._pending = undefined;
    this._timeout = undefined;
    this._debounceMs = options.debounce ?? 0;
    this._maxSize = options.maxSize ?? 100;
  }

  private _push(state: State) {
    this._after = [];
    if (this._current != undefined) {
      this._before.push(this._current);
      if (this._before.length > this._maxSize) {
        this._before.shift();
      }
    }
    this._current = state;
  }

  private _flush() {
    if (this._timeout != undefined) {
      clearTimeout(this._timeout);
      this._timeout = undefined;
    }
    if (this._pending != undefined) {
      this._push(this._pending);
      this._pending = undefined;
    }
  }

  get canUndo(): boolean {
    // Should track if undo() returns a non-undefined value.
    return this._before.length > 0 || (this._pending != undefined && this._current != undefined);
  }

  get canRedo(): boolean {
    // Should track if redo() returns a non-undefined value.
    return this._after.length > 0 && this._pending == undefined;
  }

  update(state: State) {
    if (this._current === state) {
      return;
    }
    if (this._debounceMs > 0) {
      this._pending = state;
      if (this._timeout != undefined) {
        clearTimeout(this._timeout);
      }
      this._timeout = setTimeout(() => {
        this._timeout = undefined;
        if (this._pending != undefined) {
          this._push(this._pending);
          this._pending = undefined;
        }
      }, this._debounceMs);
    } else {
      this._push(state);
    }
  }

  undo(): State | undefined {
    this._flush();
    if (this._before.length > 0) {
      const item = this._before.pop()!;
      if (this._current != undefined) this._after.push(this._current);
      this._current = item;
      return item;
    }
    return undefined;
  }

  redo(): State | undefined {
    this._flush();
    if (this._after.length > 0) {
      const item = this._after.pop()!;
      if (this._current != undefined) this._before.push(this._current);
      this._current = item;
      return item;
    }
    return undefined;
  }

  clear() {
    if (this._timeout != undefined) {
      clearTimeout(this._timeout);
      this._timeout = undefined;
    }
    this._pending = undefined;
    this._after = [];
    this._before = [];
    this._current = undefined;
  }
}

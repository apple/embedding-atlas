// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { interactionHandler } from "@embedding-atlas/utils";

import { afterEach, describe, expect, it, vi } from "vitest";

class FakeEventTarget {
  listeners = new Map();

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  removeEventListener(type, listener) {
    if (this.listeners.get(type) === listener) {
      this.listeners.delete(type);
    }
  }

  dispatch(type, event) {
    this.listeners.get(type)?.(event);
  }
}

function mouseEvent(overrides = {}) {
  return {
    clientX: 0,
    clientY: 0,
    pageX: 0,
    pageY: 0,
    shiftKey: false,
    ctrlKey: false,
    altKey: false,
    metaKey: false,
    preventDefault: vi.fn(),
    stopPropagation: vi.fn(),
    ...overrides,
  };
}

describe("interactionHandler", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("passes a CursorValue to the drag up handler", () => {
    const element = new FakeEventTarget();
    const windowTarget = new FakeEventTarget();
    vi.stubGlobal("window", windowTarget);
    vi.stubGlobal("document", { activeElement: null, body: {} });

    const up = vi.fn();
    interactionHandler(element, {
      drag: () => ({ up }),
    });

    element.dispatch("mousedown", mouseEvent({ clientX: 10, clientY: 20, pageX: 30, pageY: 40 }));
    windowTarget.dispatch("mousemove", mouseEvent({ clientX: 20, clientY: 30, pageX: 40, pageY: 50 }));
    windowTarget.dispatch(
      "mouseup",
      mouseEvent({
        clientX: 25,
        clientY: 35,
        pageX: 45,
        pageY: 55,
        shiftKey: true,
        metaKey: true,
      }),
    );

    expect(up).toHaveBeenCalledExactlyOnceWith({
      clientX: 25,
      clientY: 35,
      pageX: 45,
      pageY: 55,
      modifiers: {
        shift: true,
        ctrl: false,
        alt: false,
        meta: true,
      },
    });
  });
});

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { interactionHandler } from "@embedding-atlas/utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function mouse(target, type, x = 0) {
  const event = new Event(type);
  Object.assign(event, {
    clientX: x,
    clientY: 0,
    pageX: x,
    pageY: 0,
    shiftKey: false,
    ctrlKey: false,
    altKey: false,
    metaKey: false,
  });
  target.dispatchEvent(event);
}

describe("interactionHandler gesture cleanup", () => {
  let element;
  let windowTarget;

  beforeEach(() => {
    element = new EventTarget();
    windowTarget = new EventTarget();
    vi.stubGlobal("window", windowTarget);
    vi.stubGlobal("document", { activeElement: null });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not start a drag or click after destruction during a pending gesture", () => {
    const drag = vi.fn();
    const click = vi.fn();
    const action = interactionHandler(element, { drag, click });

    mouse(element, "mousedown");
    action.destroy();
    mouse(windowTarget, "mousemove", 10);
    mouse(windowTarget, "mouseup", 10);

    expect(drag).not.toHaveBeenCalled();
    expect(click).not.toHaveBeenCalled();
  });

  it("cancels an active drag once and stops callbacks after destruction", () => {
    const move = vi.fn();
    const up = vi.fn();
    const cancel = vi.fn();
    const click = vi.fn();
    const action = interactionHandler(element, { drag: () => ({ move, up, cancel }), click });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    expect(move).toHaveBeenCalledTimes(1);
    action.destroy();
    action.destroy();
    mouse(windowTarget, "mousemove", 20);
    mouse(windowTarget, "mouseup", 20);

    expect(cancel).toHaveBeenCalledTimes(1);
    expect(move).toHaveBeenCalledTimes(1);
    expect(up).not.toHaveBeenCalled();
    expect(click).not.toHaveBeenCalled();
  });

  it("cancels a replaced drag and only tracks the new gesture", () => {
    const previous = { move: vi.fn(), up: vi.fn(), cancel: vi.fn() };
    const next = { move: vi.fn(), up: vi.fn(), cancel: vi.fn() };
    const drag = vi.fn().mockReturnValueOnce(previous).mockReturnValueOnce(next);
    const action = interactionHandler(element, { drag });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(element, "mousedown", 20);
    mouse(windowTarget, "mousemove", 30);
    mouse(windowTarget, "mouseup", 40);
    action.destroy();

    expect(previous.cancel).toHaveBeenCalledTimes(1);
    expect(previous.up).not.toHaveBeenCalled();
    expect(previous.move).toHaveBeenCalledTimes(1);
    expect(drag).toHaveBeenCalledTimes(2);
    expect(next.move).toHaveBeenCalledTimes(1);
    expect(next.up).toHaveBeenCalledExactlyOnceWith({
      clientX: 40,
      clientY: 0,
      pageX: 40,
      pageY: 0,
      modifiers: { shift: false, ctrl: false, alt: false, meta: false },
    });
    expect(next.cancel).not.toHaveBeenCalled();
  });

  it("finishes a normal drag without cancelling it on destruction", () => {
    const move = vi.fn();
    const up = vi.fn();
    const cancel = vi.fn();
    const action = interactionHandler(element, { drag: () => ({ move, up, cancel }) });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(windowTarget, "mouseup", 20);
    action.destroy();
    mouse(windowTarget, "mousemove", 30);
    mouse(windowTarget, "mouseup", 30);

    expect(move).toHaveBeenCalledTimes(1);
    expect(up).toHaveBeenCalledTimes(1);
    expect(cancel).not.toHaveBeenCalled();
  });

  it("does not restart tracking when cancellation destroys the action", () => {
    const move = vi.fn();
    const up = vi.fn();
    const click = vi.fn();
    const cancel = vi.fn(() => action.destroy());
    const drag = vi.fn(() => ({ move, up, cancel }));
    const action = interactionHandler(element, { drag, click });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(element, "mousedown", 20);
    mouse(windowTarget, "mousemove", 30);
    mouse(windowTarget, "mouseup", 40);

    expect(cancel).toHaveBeenCalledTimes(1);
    expect(drag).toHaveBeenCalledTimes(1);
    expect(move).toHaveBeenCalledTimes(1);
    expect(up).not.toHaveBeenCalled();
    expect(click).not.toHaveBeenCalled();
  });

  it("preserves clicks and removes tracking on mouseup", () => {
    const drag = vi.fn();
    const click = vi.fn();
    const action = interactionHandler(element, { drag, click });

    mouse(element, "mousedown");
    mouse(windowTarget, "mouseup");
    mouse(windowTarget, "mousemove", 10);
    mouse(windowTarget, "mouseup", 10);
    action.destroy();

    expect(click).toHaveBeenCalledTimes(1);
    expect(drag).not.toHaveBeenCalled();
  });

  it("does not cancel a completed drag when up destroys the action", () => {
    const cancel = vi.fn();
    const action = interactionHandler(element, {
      drag: () => ({ up: () => action.destroy(), cancel }),
    });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(windowTarget, "mouseup", 20);

    expect(cancel).not.toHaveBeenCalled();
  });

  it("cancels a handler returned after its drag factory destroys the action", () => {
    const move = vi.fn();
    const up = vi.fn();
    const cancel = vi.fn();
    const action = interactionHandler(element, {
      drag: () => {
        action.destroy();
        return { move, up, cancel };
      },
    });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(windowTarget, "mouseup", 20);

    expect(cancel).toHaveBeenCalledTimes(1);
    expect(move).not.toHaveBeenCalled();
    expect(up).not.toHaveBeenCalled();
  });

  it("does not replace a gesture started by the cancellation callback", () => {
    const next = { move: vi.fn(), up: vi.fn() };
    const drag = vi
      .fn()
      .mockReturnValueOnce({ cancel: () => mouse(element, "mousedown", 100) })
      .mockReturnValueOnce(next);
    const action = interactionHandler(element, { drag });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(element, "mousedown", 20);
    mouse(windowTarget, "mousemove", 101);
    expect(drag).toHaveBeenCalledTimes(1);
    mouse(windowTarget, "mousemove", 110);
    mouse(windowTarget, "mouseup", 120);
    action.destroy();

    expect(drag).toHaveBeenCalledTimes(2);
    expect(next.move).toHaveBeenCalledTimes(1);
    expect(next.up).toHaveBeenCalledTimes(1);
  });

  it("does not install tracking if blurring the active element destroys the action", () => {
    const drag = vi.fn();
    const click = vi.fn();
    const action = interactionHandler(element, { drag, click });
    vi.stubGlobal("document", { activeElement: { blur: () => action.destroy() } });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    mouse(windowTarget, "mouseup", 20);

    expect(drag).not.toHaveBeenCalled();
    expect(click).not.toHaveBeenCalled();
  });

  it("cancels a stale factory result without overwriting a replacement gesture", () => {
    const previous = { move: vi.fn(), up: vi.fn(), cancel: vi.fn() };
    const next = { move: vi.fn(), up: vi.fn(), cancel: vi.fn() };
    const drag = vi
      .fn()
      .mockImplementationOnce(() => {
        mouse(element, "mousedown", 100);
        return previous;
      })
      .mockReturnValueOnce(next);
    const action = interactionHandler(element, { drag });

    mouse(element, "mousedown");
    mouse(windowTarget, "mousemove", 10);
    expect(previous.cancel).toHaveBeenCalledTimes(1);
    expect(previous.move).not.toHaveBeenCalled();
    mouse(windowTarget, "mousemove", 110);
    mouse(windowTarget, "mouseup", 120);
    action.destroy();

    expect(drag).toHaveBeenCalledTimes(2);
    expect(previous.up).not.toHaveBeenCalled();
    expect(next.move).toHaveBeenCalledTimes(1);
    expect(next.up).toHaveBeenCalledTimes(1);
    expect(next.cancel).not.toHaveBeenCalled();
  });
});

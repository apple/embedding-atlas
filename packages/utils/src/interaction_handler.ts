// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface CursorValue {
  clientX: number;
  clientY: number;
  pageX: number;
  pageY: number;
  modifiers: {
    shift: boolean;
    ctrl: boolean;
    alt: boolean;
    meta: boolean;
  };
}

export interface DragHandler {
  move?: (value: CursorValue) => void;
  up?: (value: CursorValue) => void;
  cancel?: () => void;
}

interface InteractionHandlerProps {
  /** If defined, handles click gestures. */
  click?: (value: CursorValue) => void;

  /**
   * If defined, handles drag gestures with a single pointer.
   * Reports the movement of the tracked pointer.
   */
  drag?: (value: CursorValue) => DragHandler | undefined;

  /** If defined, tracks hovering cursors. */
  hover?: (value: CursorValue | null) => void;
}

type UnwrapOptional<T> = T extends undefined ? never : T;

/**
 * Invoke the method on target if defined, and if all args are defined.
 */
function tryInvoke<T extends object, K extends keyof T>(
  target: T,
  method: K,
  ...args: any[]
): T[K] extends ((...args: any[]) => any) | undefined ? ReturnType<UnwrapOptional<T[K]>> | undefined : undefined {
  try {
    if (args.every((x) => x !== undefined)) {
      const fn = target[method];
      if (typeof fn === "function") {
        return fn(...args);
      }
    }
  } catch (e) {
    console.error(e);
  }
}

const DRAG_THRESHOLD = 2;

function valueForEvent(e: MouseEvent): CursorValue {
  return {
    clientX: e.clientX,
    clientY: e.clientY,
    pageX: e.pageX,
    pageY: e.pageY,
    modifiers: { shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey, meta: e.metaKey },
  };
}

export function interactionHandler(element: SVGElement | HTMLElement, props: InteractionHandlerProps) {
  let handlers = props;
  let dragHandler: DragHandler | undefined = undefined;
  let isDown = false;
  let isHovering = false;

  const onMouseDown = (e1: MouseEvent) => {
    let v1 = valueForEvent(e1);
    e1.preventDefault();
    e1.stopPropagation();
    blurActiveElement();

    isDown = true;

    if (dragHandler) {
      tryInvoke(dragHandler, "up");
      dragHandler = undefined;
    }

    let onMove = (e2: MouseEvent) => {
      let v2 = valueForEvent(e2);

      if (handlers.drag && !dragHandler) {
        if (Math.hypot(v1.clientX - v2.clientX, v1.clientY - v2.clientY) > DRAG_THRESHOLD) {
          dragHandler = tryInvoke(handlers, "drag", v1);
          if (dragHandler) {
            tryInvoke(dragHandler, "move", v2);
          }
        }
      } else if (dragHandler) {
        tryInvoke(dragHandler, "move", v2);
      }
    };

    let onUp = (e2: MouseEvent) => {
      let v2 = valueForEvent(e2);

      isDown = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);

      if (dragHandler) {
        tryInvoke(dragHandler, "up", e2);
        dragHandler = undefined;
      } else {
        if (handlers.click) {
          tryInvoke(handlers, "click", v2);
        }
      }
    };

    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("mouseup", onUp, { passive: true });
  };

  const onMouseMove = (e: MouseEvent) => {
    if (!isDown) {
      // If no entry, this means the pointer is hovering.
      if (handlers.hover) {
        tryInvoke(handlers, "hover", valueForEvent(e));
        isHovering = true;
      }
      return;
    }
  };

  const onMouseLeave = (e: MouseEvent) => {
    if (isHovering && handlers.hover) {
      tryInvoke(handlers, "hover", null);
      isHovering = false;
    }
  };

  element.addEventListener("mousedown", onMouseDown as any, { passive: false });
  element.addEventListener("mousemove", onMouseMove as any, { passive: true });
  element.addEventListener("mouseleave", onMouseLeave as any, { passive: true });

  return {
    update: (props: InteractionHandlerProps) => {
      handlers = props;
    },
    destroy: () => {
      element.removeEventListener("mousedown", onMouseDown as any);
      element.removeEventListener("mousemove", onMouseMove as any);
      element.removeEventListener("mouseleave", onMouseLeave as any);
    },
  };
}

function blurActiveElement() {
  try {
    if (document.activeElement && document.activeElement !== document.body) {
      (document.activeElement as any).blur?.();
    }
  } catch (_) {}
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { FocusRequest, Overlay } from "../charts/chart.js";

/**
 * Resolves which overlay a view should render.
 *
 * An external overlay (provided through the viewer's `overlay` prop) takes
 * precedence over the internal overlay (e.g. produced by search) whenever it
 * is set to an object, even an empty one. When the external overlay is
 * `undefined` or `null`, the internal overlay is used, so clearing the prop
 * restores the internal overlay if one exists.
 */
export function resolveEffectiveOverlay(
  externalOverlay: Overlay | null | undefined,
  internalOverlay: Overlay | null,
): Overlay | null {
  return externalOverlay ?? internalOverlay;
}

/**
 * Tracks focus requests (see the viewer's `focus` prop) for one chart context.
 *
 * A request is acted upon only when its `key` differs from the key of the
 * previously accepted request, which makes it possible to request the same
 * point again by providing a new key. Each accepted request gets a sequence
 * token; because resolving a request is asynchronous, callers should check
 * `isLatest` after awaiting and drop the result if a newer request arrived
 * in the meantime.
 */
export class FocusRequestTracker {
  private lastKey: string | number | null = null;
  private latestSeq = 0;

  /**
   * Registers a focus request. Returns a sequence token if the request should
   * be acted upon, or `null` if it should be ignored (no request, or a request
   * whose key was already handled).
   */
  accept(request: FocusRequest | null): number | null {
    if (request == null || request.key === this.lastKey) {
      return null;
    }
    this.lastKey = request.key;
    return ++this.latestSeq;
  }

  /** Whether the given sequence token still belongs to the latest accepted request. */
  isLatest(seq: number): boolean {
    return seq === this.latestSeq;
  }
}

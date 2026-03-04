// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import DOMPurify from "dompurify";

// Add a hook to make all links open a new window
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if ("target" in node) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

export function sanitizeHTML(html: string): string {
  return DOMPurify.sanitize(html);
}

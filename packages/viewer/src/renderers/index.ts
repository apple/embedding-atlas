// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { CustomCell } from "@embedding-atlas/table";

import { ImageRenderer } from "./image.js";
import { JSONRenderer, safeJSONStringify } from "./json.js";
import { MarkdownRenderer } from "./markdown.js";
import { MessagesRenderer } from "./messages.js";
import { URLRenderer } from "./url.js";

/** A type describing how to display a column in the table, tooltip, and search results */
export interface ColumnStyle {
  /** The renderer type */
  renderer?: string;
  /** Props passed to the renderer class */
  rendererOptions?: any;
  /** Display style */
  display?: "full" | "badge" | "hidden";
}

export let textRendererClasses: Record<string, any> = {
  markdown: MarkdownRenderer,
  image: ImageRenderer,
  url: URLRenderer,
  json: JSONRenderer,
  messages: MessagesRenderer,
};

export let renderersList = [
  { renderer: "markdown", label: "Markdown" },
  { renderer: "image", label: "Image" },
  { renderer: "url", label: "Link" },
  { renderer: "json", label: "JSON" },
  { renderer: "messages", label: "Messages" },
];

export function getRenderer(value: string | CustomCell | null | undefined) {
  if (value == null) {
    return undefined;
  }
  if (typeof value == "string") {
    return textRendererClasses[value];
  }
  return value; // value is a CustomCell
}

export function isLink(value: any): boolean {
  return typeof value == "string" && (value.startsWith("http://") || value.startsWith("https://"));
}

export function isImage(value: any): boolean {
  if (value == null) {
    return false;
  }
  if (typeof value == "string" && value.startsWith("data:image/")) {
    return true;
  }
  if (value.bytes && value.bytes instanceof Uint8Array) {
    // TODO: check if the bytes are actually an image.
    return true;
  }
  return false;
}

export function stringify(value: any): string {
  if (value == null) {
    return "(null)";
  } else if (typeof value == "string") {
    return value.toString();
  } else if (typeof value == "number") {
    return value.toLocaleString();
  } else if (Array.isArray(value)) {
    return "[" + value.map((x) => stringify(x)).join(", ") + "]";
  }
  try {
    return safeJSONStringify(value);
  } catch (e) {
    return value.toString();
  }
}

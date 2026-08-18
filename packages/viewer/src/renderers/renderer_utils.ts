// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { detectAudioMimeType, detectBase64MimeType, detectImageMimeType } from "@embedding-atlas/utils";

const imageExtensions = new Set(["png", "jpg", "jpeg", "tiff", "tif", "gif"]);
const audioExtensions = new Set(["wav", "wave", "mp3", "aac"]);

/** Detect displayable media from a column value. Strings are checked by prefix
 * (URLs, data: URLs) and then probed as raw base64; binary values are detected based
 * on magic bytes, with a fallback on the path's
 * file extension. */
export function valueKind(value: any): "link" | "image" | "audio" | undefined {
  if (typeof value == "string") {
    if (value.startsWith("http://") || value.startsWith("https://")) {
      return "link";
    } else if (value.startsWith("data:image/")) {
      return "image";
    } else if (value.startsWith("data:audio/")) {
      return "audio";
    }
    let mimeType = detectBase64MimeType(value);
    if (mimeType != null) {
      if (mimeType.startsWith("image/")) {
        return "image";
      } else if (mimeType.startsWith("audio/")) {
        return "audio";
      }
    }
    return undefined;
  }

  let bytes: Uint8Array | null = null;
  let path: string | undefined = undefined;
  if (value instanceof Uint8Array) {
    bytes = value;
  } else if (typeof value == "object" && value != null && value.bytes) {
    if (value.bytes instanceof Uint8Array) {
      bytes = value.bytes;
    }
    if (typeof value.path == "string") {
      path = value.path;
    }
  }

  if (bytes != null) {
    if (detectImageMimeType(bytes) != null) {
      return "image";
    }
    if (detectAudioMimeType(bytes) != null) {
      return "audio";
    }
  }
  if (path != null) {
    let ext = path.split(".").pop()?.toLowerCase() ?? "";
    if (imageExtensions.has(ext)) {
      return "image";
    } else if (audioExtensions.has(ext)) {
      return "audio";
    }
  }
  return undefined;
}

/**
 * Prevents large binary values from being expanded into a JSON array of numbers.
 * For eg. this is enough to keep typical embedding vectors visible and  small enough that an
 * undetected image can never dump megabytes of digits into the DOM. */

const MAX_INLINE_BINARY_BYTES = 16384;

export function safeJSONStringify(value: any, space?: number): string {
  try {
    return JSON.stringify(
      value,
      (_, value) => {
        if (value instanceof Object && ArrayBuffer.isView(value)) {
          if (value.byteLength > MAX_INLINE_BINARY_BYTES) {
            return `(${value.constructor.name}, ${formatByteSize(value.byteLength)})`;
          }
          return Array.from(value as any);
        }
        return value;
      },
      space,
    );
  } catch (e) {
    return "(invalid)";
  }
}

function formatByteSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} kB`;
  } else {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
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
  } else if (value instanceof Date) {
    return value.toISOString();
  }
  try {
    return safeJSONStringify(value);
  } catch (e) {
    return value.toString();
  }
}

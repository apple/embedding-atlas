// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

const imageExtensions = new Set(["png", "jpg", "jpeg", "tiff", "tif", "gif"]);
const audioExtensions = new Set(["wav", "wave", "mp3", "aac"]);

// Quick heuristic check based on string prefixes and file extensions.
// May not capture all formats (e.g., raw base64 strings, binary objects without a path).
export function valueKind(value: any): "link" | "image" | "audio" | undefined {
  if (typeof value == "string") {
    if (value.startsWith("http://") || value.startsWith("https://")) {
      return "link";
    } else if (value.startsWith("data:image/")) {
      return "image";
    } else if (value.startsWith("data:audio/")) {
      return "audio";
    }
  } else if (typeof value == "object" && value != null && value.bytes) {
    if (typeof value.path == "string") {
      let ext = value.path.split(".").pop()?.toLowerCase();
      if (imageExtensions.has(ext)) {
        return "image";
      } else if (audioExtensions.has(ext)) {
        return "audio";
      }
    }
  }
  return undefined;
}

export function safeJSONStringify(value: any, space?: number): string {
  try {
    return JSON.stringify(
      value,
      (_, value) => {
        if (value instanceof Object && ArrayBuffer.isView(value)) {
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

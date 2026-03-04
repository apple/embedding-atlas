// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

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

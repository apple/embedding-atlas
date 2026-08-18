// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { safeJSONStringify, valueKind } from "../src/renderers/renderer_utils.js";

const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d];
const WAV_HEADER = [0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45];
const UNKNOWN_BYTES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

function bytes(values: number[]): Uint8Array {
  return new Uint8Array(values);
}

function base64(values: number[]): string {
  return btoa(String.fromCharCode(...values));
}

function padded(values: number[], length: number = 48): number[] {
  return values.concat(new Array(length - values.length).fill(0));
}

describe("valueKind", () => {
  it("detects links and data URLs from string prefixes", () => {
    expect(valueKind("https://example.com/a.png")).toBe("link");
    expect(valueKind("http://example.com")).toBe("link");
    expect(valueKind("data:image/png;base64,abcd")).toBe("image");
    expect(valueKind("data:audio/wav;base64,abcd")).toBe("audio");
  });

  it("detects raw base64-encoded media strings", () => {
    expect(valueKind(base64(padded(PNG_HEADER)))).toBe("image");
    expect(valueKind(base64(padded(WAV_HEADER)))).toBe("audio");
  });

  it("does not misdetect ordinary strings", () => {
    expect(valueKind("hello, world")).toBeUndefined();
    expect(valueKind("")).toBeUndefined();
    expect(valueKind("//foobar")).toBeUndefined();
    expect(valueKind("//TODOxx")).toBeUndefined();
    expect(valueKind("SUQzTEST")).toBeUndefined();
    expect(valueKind("d2d2f1a4b8c09e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f")).toBeUndefined();
    expect(valueKind(base64(padded(UNKNOWN_BYTES)))).toBeUndefined();
    expect(valueKind(base64(PNG_HEADER))).toBeUndefined();
  });

  it("detects a bare Uint8Array by magic bytes", () => {
    expect(valueKind(bytes(PNG_HEADER))).toBe("image");
    expect(valueKind(bytes(WAV_HEADER))).toBe("audio");
    expect(valueKind(bytes(UNKNOWN_BYTES))).toBeUndefined();
  });

  it("detects a {bytes} struct without a path", () => {
    expect(valueKind({ bytes: bytes(PNG_HEADER), path: null })).toBe("image");
    expect(valueKind({ bytes: bytes(PNG_HEADER) })).toBe("image");
    expect(valueKind({ bytes: bytes(WAV_HEADER) })).toBe("audio");
  });

  it("falls back to the path extension when magic bytes are not recognized", () => {
    expect(valueKind({ bytes: bytes(UNKNOWN_BYTES), path: "scan.tif" })).toBe("image");
    expect(valueKind({ bytes: bytes(UNKNOWN_BYTES), path: "clip.wav" })).toBe("audio");
    expect(valueKind({ bytes: bytes(UNKNOWN_BYTES), path: "notes.txt" })).toBeUndefined();
  });

  it("returns undefined for non-media values", () => {
    expect(valueKind(null)).toBeUndefined();
    expect(valueKind(undefined)).toBeUndefined();
    expect(valueKind(42)).toBeUndefined();
    expect(valueKind({ a: 1 })).toBeUndefined();
    expect(valueKind([1, 2, 3])).toBeUndefined();
  });
});

describe("safeJSONStringify", () => {
  it("inlines small binary values as number arrays", () => {
    expect(safeJSONStringify(new Uint8Array([1, 2, 3]))).toBe("[1,2,3]");
    expect(safeJSONStringify({ bytes: new Uint8Array([1, 2]), path: "a.bin" })).toBe('{"bytes":[1,2],"path":"a.bin"}');
  });

  it("keeps embedding vectors like data visible", () => {
    let vector = new Float32Array(384);
    let result = safeJSONStringify(vector);
    expect(result.startsWith("[")).toBe(true);
    expect(JSON.parse(result).length).toBe(384);
  });
});

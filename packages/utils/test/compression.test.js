// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { base64Decode, base64Encode, compress, decompress } from "@embedding-atlas/utils";

import { describe, expect, it } from "vitest";

describe("compression", () => {
  describe("compress and decompress", () => {
    it("should compress and decompress data correctly", async () => {
      const original = new TextEncoder().encode("Hello, World!");
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
      expect(new TextDecoder().decode(decompressed)).toBe("Hello, World!");
    });

    it("should compress and decompress empty data", async () => {
      const original = new Uint8Array(0);
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
    });

    it("should compress and decompress large data", async () => {
      const original = new Uint8Array(100000).fill(42);
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
      expect(compressed.length).toBeLessThan(original.length);
    });

    it("should compress and decompress binary data", async () => {
      const original = new Uint8Array([0, 1, 2, 3, 255, 254, 253]);
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
    });

    it("should support deflate format", async () => {
      const original = new TextEncoder().encode("Test data for deflate");
      const compressed = await compress(original, "deflate");
      const decompressed = await decompress(compressed, "deflate");

      expect(decompressed).toEqual(original);
    });

    it("should support deflate-raw format", async () => {
      const original = new TextEncoder().encode("Test data for deflate-raw");
      const compressed = await compress(original, "deflate-raw");
      const decompressed = await decompress(compressed, "deflate-raw");

      expect(decompressed).toEqual(original);
    });

    it("should use gzip format by default", async () => {
      const original = new TextEncoder().encode("Default format test");
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
    });

    it("should handle repeated data efficiently", async () => {
      const original = new TextEncoder().encode("a".repeat(10000));
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
      expect(compressed.length).toBeLessThan(original.length / 10);
    });

    it("should handle unicode text", async () => {
      const original = new TextEncoder().encode("Hello 世界 🌍");
      const compressed = await compress(original);
      const decompressed = await decompress(compressed);

      expect(decompressed).toEqual(original);
      expect(new TextDecoder().decode(decompressed)).toBe("Hello 世界 🌍");
    });
  });

  describe("base64Encode and base64Decode", () => {
    it("should encode and decode data correctly", () => {
      const original = new TextEncoder().encode("Hello, World!");
      const encoded = base64Encode(original);
      const decoded = base64Decode(encoded);

      expect(decoded).toEqual(original);
      expect(new TextDecoder().decode(decoded)).toBe("Hello, World!");
    });

    it("should encode to valid base64 string", () => {
      const original = new Uint8Array([1, 2, 3, 4, 5]);
      const encoded = base64Encode(original);

      expect(typeof encoded).toBe("string");
      expect(/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)).toBe(true);
    });

    it("should encode and decode empty data", () => {
      const original = new Uint8Array(0);
      const encoded = base64Encode(original);
      const decoded = base64Decode(encoded);

      expect(encoded).toBe("");
      expect(decoded).toEqual(original);
    });

    it("should encode and decode binary data", () => {
      const original = new Uint8Array([0, 1, 2, 3, 255, 254, 253]);
      const encoded = base64Encode(original);
      const decoded = base64Decode(encoded);

      expect(decoded).toEqual(original);
    });

    it("should handle large data (chunking)", () => {
      const original = new Uint8Array(100000).fill(42);
      const encoded = base64Encode(original);
      const decoded = base64Decode(encoded);

      expect(decoded).toEqual(original);
    });

    it("should handle all byte values", () => {
      const original = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        original[i] = i;
      }
      const encoded = base64Encode(original);
      const decoded = base64Decode(encoded);

      expect(decoded).toEqual(original);
    });

    it("should encode and decode unicode text", () => {
      const original = new TextEncoder().encode("Hello 世界 🌍");
      const encoded = base64Encode(original);
      const decoded = base64Decode(encoded);

      expect(decoded).toEqual(original);
      expect(new TextDecoder().decode(decoded)).toBe("Hello 世界 🌍");
    });

    it("should handle data that requires padding", () => {
      for (let length = 0; length < 1000; length++) {
        const original = new Uint8Array(length);
        for (let i = 0; i < length; i++) {
          original[i] = Math.floor(Math.random() * 256);
        }
        const encoded = base64Encode(original);
        const decoded = base64Decode(encoded);
        expect(decoded).toEqual(original);
      }
    });
  });

  describe("integration: compress and encode", () => {
    it("should compress then base64 encode and decode then decompress", async () => {
      const original = new TextEncoder().encode("Compress and encode test data");
      const compressed = await compress(original);
      const encoded = base64Encode(compressed);
      const decoded = base64Decode(encoded);
      const decompressed = await decompress(decoded);

      expect(decompressed).toEqual(original);
      expect(new TextDecoder().decode(decompressed)).toBe("Compress and encode test data");
    });

    it("should handle large data through full pipeline", async () => {
      const original = new TextEncoder().encode("Lorem ipsum ".repeat(1000));
      const compressed = await compress(original);
      const encoded = base64Encode(compressed);
      const decoded = base64Decode(encoded);
      const decompressed = await decompress(decoded);

      expect(decompressed).toEqual(original);
      expect(encoded.length).toBeLessThan(original.length);
    });
  });
});

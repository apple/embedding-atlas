// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import {
  audioToDataUrl,
  detectAudioMimeType,
  detectBase64MimeType,
  detectImageMimeType,
  imageToDataUrl,
} from "@embedding-atlas/utils";

import { describe, expect, it } from "vitest";

const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d];
const JPEG_HEADER = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46];
const GIF89_HEADER = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00];
const TIFF_LE_HEADER = [0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00];
const BMP_HEADER = [0x42, 0x4d, 0x9a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7a, 0x00];
const WAV_HEADER = [0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45];
const ID3_HEADER = [0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x07, 0x76];
const MP3_SYNC_HEADER = [0xff, 0xfb, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00];
const AAC_ADTS_HEADER = [0xff, 0xf1, 0x50, 0x80, 0x00, 0x00, 0x00, 0x00];
const OGG_HEADER = [0x4f, 0x67, 0x67, 0x53, 0x00, 0x02, 0x00, 0x00];
const FLAC_HEADER = [0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22];
const M4A_HEADER = [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70];
const UNKNOWN_BYTES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

function bytes(values) {
  return new Uint8Array(values);
}

function base64(values) {
  return btoa(String.fromCharCode(...values));
}

function padded(values, length = 48) {
  return values.concat(new Array(length - values.length).fill(0));
}

describe("detectImageMimeType", () => {
  it("detects known image formats from magic bytes", () => {
    expect(detectImageMimeType(bytes(PNG_HEADER))).toBe("image/png");
    expect(detectImageMimeType(bytes(JPEG_HEADER))).toBe("image/jpeg");
    expect(detectImageMimeType(bytes(GIF89_HEADER))).toBe("image/gif");
    expect(detectImageMimeType(bytes(TIFF_LE_HEADER))).toBe("image/tiff");
    expect(detectImageMimeType(bytes(BMP_HEADER))).toBe("image/bmp");
  });

  it("returns null for unrecognized data", () => {
    expect(detectImageMimeType(bytes(UNKNOWN_BYTES))).toBeNull();
    expect(detectImageMimeType(bytes([]))).toBeNull();
    expect(detectImageMimeType(bytes(WAV_HEADER))).toBeNull();
  });

  it("rejects 'BM' data whose header bytes are not zero", () => {
    expect(detectImageMimeType(bytes([0x42, 0x4d, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]))).toBeNull();
  });
});

describe("detectAudioMimeType", () => {
  it("detects known audio formats from magic bytes", () => {
    expect(detectAudioMimeType(bytes(WAV_HEADER))).toBe("audio/wav");
    expect(detectAudioMimeType(bytes(ID3_HEADER))).toBe("audio/mpeg");
    expect(detectAudioMimeType(bytes(MP3_SYNC_HEADER))).toBe("audio/mpeg");
    expect(detectAudioMimeType(bytes(AAC_ADTS_HEADER))).toBe("audio/aac");
    expect(detectAudioMimeType(bytes(OGG_HEADER))).toBe("audio/ogg");
    expect(detectAudioMimeType(bytes(FLAC_HEADER))).toBe("audio/flac");
    expect(detectAudioMimeType(bytes(M4A_HEADER))).toBe("audio/mp4");
  });

  it("falls back to the  extensions for unrecognized", () => {
    expect(detectAudioMimeType(bytes(UNKNOWN_BYTES), "clip.mp3")).toBe("audio/mpeg");
    expect(detectAudioMimeType(bytes(UNKNOWN_BYTES), "clip.wav")).toBe("audio/wav");
  });
});

describe("detectBase64MimeType", () => {
  it("detects media in base64-encoded strings", () => {
    expect(detectBase64MimeType(base64(padded(PNG_HEADER)))).toBe("image/png");
    expect(detectBase64MimeType(base64(padded(JPEG_HEADER)))).toBe("image/jpeg");
    expect(detectBase64MimeType(base64(padded(WAV_HEADER)))).toBe("audio/wav");
    expect(detectBase64MimeType(base64(padded(ID3_HEADER)))).toBe("audio/mpeg");
  });

  it("returns null for non-base64 strings", () => {
    expect(detectBase64MimeType("hello, world!")).toBeNull();
    expect(detectBase64MimeType("short")).toBeNull();
    expect(detectBase64MimeType("")).toBeNull();
    expect(detectBase64MimeType("data:image/png;base64,abcd")).toBeNull();
  });

  it("returns null for strings very short to be real media", () => {
    expect(detectBase64MimeType(base64(PNG_HEADER))).toBeNull();
    expect(detectBase64MimeType(base64(WAV_HEADER))).toBeNull();
  });

  it("returns null for base64-shaped strings with unrecognized content", () => {
    // For eg. hex hashes are valid base64 characters but they must never decode to media magic bytes.
    expect(detectBase64MimeType("d2d2f1a4b8c09e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f")).toBeNull();
    expect(detectBase64MimeType(base64(padded(UNKNOWN_BYTES)))).toBeNull();
  });

  it("does not misdetect base64-shaped text as audio", () => {
    expect(detectBase64MimeType("//foobar")).toBeNull();
    expect(detectBase64MimeType("//TODOxx")).toBeNull();
    expect(detectBase64MimeType("SUQzTEST")).toBeNull();
    expect(detectBase64MimeType("/".repeat(64))).toBeNull();
    expect(detectBase64MimeType(base64(padded(MP3_SYNC_HEADER)))).toBeNull();
    expect(detectBase64MimeType("SUQz" + "A".repeat(60))).toBeNull();
  });
});

describe("imageToDataUrl", () => {
  it("converts a bare Uint8Array", () => {
    expect(imageToDataUrl(bytes(PNG_HEADER))).toBe("data:image/png;base64," + base64(PNG_HEADER));
  });

  it("converts a {bytes, path} object", () => {
    expect(imageToDataUrl({ bytes: bytes(JPEG_HEADER), path: "photo.jpg" })).toBe(
      "data:image/jpeg;base64," + base64(JPEG_HEADER),
    );
  });

  it("converts a {bytes} object without a path", () => {
    expect(imageToDataUrl({ bytes: bytes(PNG_HEADER), path: null })).toBe(
      "data:image/png;base64," + base64(PNG_HEADER),
    );
  });

  it("passes through data: and http URLs", () => {
    expect(imageToDataUrl("data:image/png;base64,abcd")).toBe("data:image/png;base64,abcd");
    expect(imageToDataUrl("https://example.com/a.png")).toBe("https://example.com/a.png");
    expect(imageToDataUrl("http://example.com/a.png")).toBe("http://example.com/a.png");
  });

  it("wraps a raw base64 string with a sniffed MIME type", () => {
    let value = base64(PNG_HEADER);
    expect(imageToDataUrl(value)).toBe("data:image/png;base64," + value);
  });

  it("labels unrecognized binary as application/octet-stream", () => {
    expect(imageToDataUrl(bytes(UNKNOWN_BYTES))).toBe("data:application/octet-stream;base64," + base64(UNKNOWN_BYTES));
  });

  it("returns null for non-media strings and null values", () => {
    expect(imageToDataUrl("not base64 at all!")).toBeNull();
    expect(imageToDataUrl(null)).toBeNull();
    expect(imageToDataUrl(undefined)).toBeNull();
    expect(imageToDataUrl(42)).toBeNull();
  });
});

describe("audioToDataUrl", () => {
  it("converts audio bytes", () => {
    expect(audioToDataUrl(bytes(WAV_HEADER))).toBe("data:audio/wav;base64," + base64(WAV_HEADER));
  });

  it("uses the path extension for unrecognized data", () => {
    expect(audioToDataUrl({ bytes: bytes(UNKNOWN_BYTES), path: "clip.mp3" })).toBe(
      "data:audio/mpeg;base64," + base64(UNKNOWN_BYTES),
    );
  });
});

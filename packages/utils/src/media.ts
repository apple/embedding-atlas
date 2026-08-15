// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export function imageToDataUrl(value: any): string | null {
  return mediaToDataUrl(value, detectImageMimeType);
}

export function audioToDataUrl(value: any): string | null {
  return mediaToDataUrl(value, detectAudioMimeType);
}

function mediaToDataUrl(value: any, detectType: (bytes: Uint8Array, path?: string) => string | null): string | null {
  if (value == null) {
    return null;
  }
  try {
    if (typeof value == "string") {
      if (value.startsWith("data:") || value.startsWith("http://") || value.startsWith("https://")) {
        return value;
      } else if (looksLikeBase64(value)) {
        let type = detectType(base64DecodePrefix(value)) ?? "application/octet-stream";
        return `data:${type};base64,` + value;
      } else {
        return null;
      }
    } else {
      let bytes: Uint8Array<ArrayBuffer> | null = null;
      let path: string | undefined = undefined;
      if (value.bytes && value.bytes instanceof Uint8Array) {
        bytes = value.bytes;
        if (typeof value.path == "string") {
          path = value.path;
        }
      }
      if (value instanceof Uint8Array) {
        bytes = value as any;
      }
      if (bytes != null) {
        let type = detectType(bytes, path) ?? "application/octet-stream";
        return `data:${type};base64,` + base64Encode(bytes);
      }
    }
  } catch (_) {
    return null;
  }
  return null;
}

function startsWith(data: Uint8Array, prefix: number[]): boolean {
  if (data.length < prefix.length) {
    return false;
  }
  for (let i = 0; i < prefix.length; i++) {
    if (data[i] != prefix[i]) {
      return false;
    }
  }
  return true;
}

/** Detect the image MIME type from magic bytes. Returns null for unrecognized data. */
export function detectImageMimeType(data: Uint8Array): string | null {
  if (startsWith(data, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) {
    return "image/png";
  } else if (startsWith(data, [0xff, 0xd8, 0xff])) {
    return "image/jpeg";
  } else if (startsWith(data, [0x49, 0x49, 0x2a, 0x00])) {
    return "image/tiff";
  } else if (
    // Require the header's reserved
    // bytes (offset 6-9) to be zero so arbitrary binary doesn't misdetect.
    startsWith(data, [0x42, 0x4d]) &&
    data.length >= 10 &&
    data[6] === 0 &&
    data[7] === 0 &&
    data[8] === 0 &&
    data[9] === 0
  ) {
    return "image/bmp";
  } else if (
    startsWith(data, [0x47, 0x49, 0x46, 0x38, 0x37, 0x61]) ||
    startsWith(data, [0x47, 0x49, 0x46, 0x38, 0x39, 0x61])
  ) {
    return "image/gif";
  }
  return null;
}

/** Detect the audio MIME type from magic bytes, falling back to the file
 * extension of `path` when the content is unrecognized. Returns null if
 * neither matches. */
export function detectAudioMimeType(data: Uint8Array, path?: string): string | null {
  let container = detectAudioContainer(data);
  if (container != null) return container;
  // Check for MPEG audio / AAC ADTS frames (both share the 0xff sync byte).
  // Top 11 bits = sync word (0xffe0 mask). Layer bits (bits 1-2 of second byte):
  //   00 = AAC (ADTS), non-zero = MPEG audio (MP3/MP2/MP1).
  if (data.length >= 2 && data[0] === 0xff && (data[1] & 0xe0) === 0xe0) {
    const layer = (data[1] >> 1) & 0x03;
    return layer === 0 ? "audio/aac" : "audio/mpeg";
  }
  // Attempt to infering from path extension
  if (path) {
    const ext = path.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "mp3":
        return "audio/mpeg";
      case "wav":
        return "audio/wav";
      case "ogg":
        return "audio/ogg";
      case "flac":
        return "audio/flac";
      case "aac":
        return "audio/aac";
      case "m4a":
        return "audio/mp4";
      case "webm":
        return "audio/webm";
    }
  }
  return null;
}

function detectAudioContainer(data: Uint8Array): string | null {
  // Check for MP3 with an ID3v2 .
  if (
    startsWith(data, [0x49, 0x44, 0x33]) &&
    data.length >= 10 &&
    (data[3] === 2 || data[3] === 3 || data[3] === 4) &&
    data[6] < 0x80 &&
    data[7] < 0x80 &&
    data[8] < 0x80 &&
    data[9] < 0x80
  ) {
    return "audio/mpeg";
  }
  // Check for WAV (RIFF....WAVE)
  if (startsWith(data, [0x52, 0x49, 0x46, 0x46]) && data.length >= 12) {
    if (data[8] === 0x57 && data[9] === 0x41 && data[10] === 0x56 && data[11] === 0x45) {
      return "audio/wav";
    }
  }
  // Check for OGG
  if (startsWith(data, [0x4f, 0x67, 0x67, 0x53])) {
    return "audio/ogg";
  }
  // Check for FLAC
  if (startsWith(data, [0x66, 0x4c, 0x61, 0x43])) {
    return "audio/flac";
  }
  // Check for WebM/Matroska
  if (startsWith(data, [0x1a, 0x45, 0xdf, 0xa3])) {
    return "audio/webm";
  }
  // Check for M4A/MP4 audio
  // High bytes of 32bit size is 0
  if (
    data.length >= 8 &&
    data[0] === 0 &&
    data[1] === 0 &&
    data[2] === 0 &&
    data[4] === 0x66 &&
    data[5] === 0x74 &&
    data[6] === 0x79 &&
    data[7] === 0x70
  ) {
    return "audio/mp4";
  }
  return null;
}

// Real base64-encoded media is far longer,
// so we omit short base-64 shaped strings entirely
const MIN_BASE64_DETECT_LENGTH = 64;

/** Detect the MIME type of a raw base64-encoded string (no `data:` prefix) by
 * sniffing the magic bytes of its decoded prefix. Used to detect
 * media in string columns. Returns null unless the string is base64-shaped,
 * at least 64 characters long, and decodes to a multi-byte media signature */
export function detectBase64MimeType(value: string): string | null {
  if (value.length < MIN_BASE64_DETECT_LENGTH || !looksLikeBase64(value)) return null;
  let bytes = base64DecodePrefix(value);
  return detectImageMimeType(bytes) ?? detectAudioContainer(bytes);
}

const BASE64_SHAPE = /^[A-Za-z0-9+/]+={0,2}$/;

function looksLikeBase64(value: string): boolean {
  return value.length >= 8 && value.length % 4 != 1 && BASE64_SHAPE.test(value);
}

/** Decode only the first few bytes of a base64 string — enough for magic-byte
 * sniffing without materializing multi-megabyte payloads. */
function base64DecodePrefix(base64: string): Uint8Array {
  const binaryString = atob(base64.length > 32 ? base64.slice(0, 32) : base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function base64Encode(data: Uint8Array): string {
  const chunkSize = 0x8000; // 32kb chunk
  let binary = "";
  for (let i = 0; i < data.length; i += chunkSize) {
    binary += String.fromCharCode(...data.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

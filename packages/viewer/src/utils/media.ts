// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export function imageToDataUrl(value: any): string | null {
  return mediaToDataUrl(value, detectImageType);
}

export function audioToDataUrl(value: any): string | null {
  return mediaToDataUrl(value, detectAudioType);
}

function mediaToDataUrl(value: any, detectType: (bytes: Uint8Array, path?: string) => string): string | null {
  if (value == null) {
    return null;
  }
  if (typeof value == "string") {
    if (value.startsWith("data:") || value.startsWith("http://") || value.startsWith("https://")) {
      return value;
    } else {
      let type = detectType(base64Decode(value));
      return `data:${type};base64,` + value;
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
      let type = detectType(bytes, path);
      return `data:${type};base64,` + base64Encode(bytes);
    }
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

function detectImageType(data: Uint8Array): string {
  if (startsWith(data, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) {
    return "image/png";
  } else if (startsWith(data, [0xff, 0xd8, 0xff])) {
    return "image/jpeg";
  } else if (startsWith(data, [0x49, 0x49, 0x2a, 0x00])) {
    return "image/tiff";
  } else if (startsWith(data, [0x42, 0x4d])) {
    return "image/bmp";
  } else if (
    startsWith(data, [0x47, 0x49, 0x46, 0x38, 0x37, 0x61]) ||
    startsWith(data, [0x47, 0x49, 0x46, 0x38, 0x39, 0x61])
  ) {
    return "image/gif";
  }
  // Unknown, fallback to generic type
  return "application/octet-stream";
}

function detectAudioType(data: Uint8Array, path?: string): string {
  // Check for MP3 (ID3 tag)
  if (startsWith(data, [0x49, 0x44, 0x33])) {
    return "audio/mpeg";
  }
  // Check for MPEG audio / AAC ADTS frames (both share the 0xff sync byte).
  // Top 11 bits = sync word (0xffe0 mask). Layer bits (bits 1-2 of second byte):
  //   00 = AAC (ADTS), non-zero = MPEG audio (MP3/MP2/MP1).
  if (data.length >= 2 && data[0] === 0xff && (data[1] & 0xe0) === 0xe0) {
    const layer = (data[1] >> 1) & 0x03;
    return layer === 0 ? "audio/aac" : "audio/mpeg";
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
  // Check for WebM/Matroska (can contain audio)
  if (startsWith(data, [0x1a, 0x45, 0xdf, 0xa3])) {
    return "audio/webm";
  }
  // Check for M4A/MP4 audio (ftyp box)
  if (data.length >= 8 && data[4] === 0x66 && data[5] === 0x74 && data[6] === 0x79 && data[7] === 0x70) {
    return "audio/mp4";
  }
  // Fallback: try to infer from path extension
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
  // Unknown, fallback to generic type
  return "application/octet-stream";
}

function base64Encode(data: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < data.length; i++) {
    binary += String.fromCharCode(data[i]);
  }
  return btoa(binary);
}

function base64Decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  return new Uint8Array([...binaryString].map((char) => char.charCodeAt(0)));
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

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
    startsWith(data, [0x47, 0x49, 0x46, 0x38, 0x37, 0x61])
  ) {
    return "image/gif";
  }
  // Unknown, fallback to generic type
  return "application/octet-stream";
}

export function imageToDataUrl(img: any): string | null {
  if (img == null) {
    return null;
  }
  if (typeof img == "string") {
    if (img.startsWith("data:")) {
      return img;
    } else {
      let type = detectImageType(base64Decode(img));
      return `data:${type};base64,` + img;
    }
  } else {
    let bytes: Uint8Array | null = null;
    if (img.bytes && img.bytes instanceof Uint8Array) {
      bytes = img.bytes;
    }
    if (img instanceof Uint8Array) {
      bytes = img;
    }
    if (bytes != null) {
      let type = detectImageType(bytes);
      return `data:${type};base64,` + base64Encode(bytes);
    }
  }
  return null;
}

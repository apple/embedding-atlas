// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export async function compress(
  data: Uint8Array<ArrayBuffer>,
  format: CompressionFormat = "gzip",
): Promise<Uint8Array<ArrayBuffer>> {
  let stream = new CompressionStream(format);
  let inputBlob = new Blob([data]);
  let compressedStream = inputBlob.stream().pipeThrough(stream);
  let compressedBlob = await new Response(compressedStream).blob();
  let buf = await compressedBlob.arrayBuffer();
  return new Uint8Array(buf);
}

export async function decompress(
  data: Uint8Array<ArrayBuffer>,
  format: CompressionFormat = "gzip",
): Promise<Uint8Array<ArrayBuffer>> {
  let stream = new DecompressionStream(format);
  let inputBlob = new Blob([data]);
  let compressedStream = inputBlob.stream().pipeThrough(stream);
  let compressedBlob = await new Response(compressedStream).blob();
  let buf = await compressedBlob.arrayBuffer();
  return new Uint8Array(buf);
}

function toStandardBase64(str: string) {
  let out = str.replace(/-/g, "+").replace(/_/g, "/");
  // Add padding if missing
  const pad = out.length % 4;
  if (pad === 2) out += "==";
  else if (pad === 3) out += "=";
  else if (pad === 1) {
    // Invalid length for base64, but we try to fix minimally
    out += "===";
  }
  return out;
}

function toUrlSafeBase64(str: string) {
  // Replace chars not allowed in URL-safe base64
  let out = str.replace(/\+/g, "-").replace(/\//g, "_");

  // Remove padding (=) since URL-safe base64 typically omits it
  out = out.replace(/=+$/, "");

  return out;
}

export function base64Encode(bytes: Uint8Array<ArrayBuffer>): string {
  const chunkSize = 0x8000; // 32 KB
  let result = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    result += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return toUrlSafeBase64(btoa(result));
}

export function base64Decode(base64: string): Uint8Array<ArrayBuffer> {
  let binaryString = atob(toStandardBase64(base64));
  let bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

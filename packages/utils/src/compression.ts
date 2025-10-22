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

export function base64Encode(bytes: Uint8Array<ArrayBuffer>): string {
  const chunkSize = 0x8000; // 32 KB
  let result = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    result += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(result);
}

export function base64Decode(base64: string): Uint8Array<ArrayBuffer> {
  let binaryString = atob(base64);
  let bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

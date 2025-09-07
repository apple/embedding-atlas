// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

async function compress(
  data: Uint8Array<ArrayBuffer>,
  format: CompressionFormat = "gzip",
): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new CompressionStream(format);
  const inputBlob = new Blob([data]);
  const compressedStream = inputBlob.stream().pipeThrough(stream);
  const compressedBlob = await new Response(compressedStream).blob();
  let buf = await compressedBlob.arrayBuffer();
  return new Uint8Array(buf);
}

async function decompress(
  data: Uint8Array<ArrayBuffer>,
  format: CompressionFormat = "gzip",
): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new DecompressionStream(format);
  const inputBlob = new Blob([data]);
  const compressedStream = inputBlob.stream().pipeThrough(stream);
  const compressedBlob = await new Response(compressedStream).blob();
  let buf = await compressedBlob.arrayBuffer();
  return new Uint8Array(buf);
}

function base64Encode(bytes: Uint8Array): string {
  let binary = "";
  const len = bytes.length;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64Decode(base64: string): Uint8Array<ArrayBuffer> {
  const binaryString = atob(base64);
  return new Uint8Array([...binaryString].map((char) => char.charCodeAt(0)));
}

async function serializePayload(object: any) {
  let encoder = new TextEncoder();
  let data = encoder.encode(JSON.stringify(object));
  let compressed = await compress(data);
  return base64Encode(compressed);
}

async function deserializePayload(data: string) {
  let buffer = base64Decode(data);
  let result = await decompress(buffer);
  let decoder = new TextDecoder();
  return JSON.parse(decoder.decode(result));
}

export async function getQueryPayload(): Promise<any | null> {
  try {
    let hash = window.location.hash;
    let lastMarkIndex = hash.lastIndexOf("?");
    if (lastMarkIndex < 0) {
      return null;
    }
    return await deserializePayload(hash.slice(lastMarkIndex + 1));
  } catch (e) {
    return null;
  }
}

export async function setQueryPayload(object: any) {
  let hash = window.location.hash;
  let lastMarkIndex = hash.lastIndexOf("?");
  let querystring = await serializePayload(object);
  if (lastMarkIndex < 0) {
    window.location.hash += "?" + querystring;
  } else {
    window.location.hash = window.location.hash.slice(0, lastMarkIndex) + "?" + querystring;
  }
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

async function compress(data, format = "gzip") {
  let stream = new CompressionStream(format);
  let inputBlob = new Blob([data]);
  let compressedStream = inputBlob.stream().pipeThrough(stream);
  let compressedBlob = await new Response(compressedStream).blob();
  let buf = await compressedBlob.arrayBuffer();
  return new Uint8Array(buf);
}

function toUrlSafeBase64(str) {
  // Replace chars not allowed in URL-safe base64
  let out = str.replace(/\+/g, "-").replace(/\//g, "_");

  // Remove padding (=) since URL-safe base64 typically omits it
  out = out.replace(/=+$/, "");

  return out;
}

function base64Encode(bytes) {
  const chunkSize = 0x8000; // 32 KB
  let result = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    result += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return toUrlSafeBase64(btoa(result));
}

export async function encode(obj) {
  let binary = new TextEncoder().encode(JSON.stringify(obj));
  return base64Encode(await compress(binary, "deflate-raw"));
}

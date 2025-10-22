// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { base64Decode, base64Encode, compress, decompress } from "@embedding-atlas/utils";

const format: CompressionFormat = "deflate-raw";

async function serializePayload(object: any) {
  let encoder = new TextEncoder();
  let data = encoder.encode(JSON.stringify(object));
  let compressed = await compress(data, format);
  return base64Encode(compressed);
}

async function deserializePayload(data: string) {
  let buffer = base64Decode(data);
  let result = await decompress(buffer, format);
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

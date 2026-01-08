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

function parseHash(hash: string): Record<string, string> {
  let lastMarkIndex = hash.lastIndexOf("?");
  if (lastMarkIndex < 0) {
    return {};
  }
  let params = new URLSearchParams(hash.slice(lastMarkIndex + 1));
  return Object.fromEntries(params.entries());
}

function updateHash(hash: string, records: Record<string, string>): string {
  let lastMarkIndex = hash.lastIndexOf("?");
  let params = new URLSearchParams(records);
  params.sort();
  let queryString = params.toString();
  if (queryString == "") {
    if (lastMarkIndex < 0) {
      return hash;
    } else {
      return hash.slice(0, lastMarkIndex);
    }
  } else {
    if (lastMarkIndex < 0) {
      return hash + "?" + queryString;
    } else {
      return hash.slice(0, lastMarkIndex) + "?" + params.toString();
    }
  }
}

export async function getQueryPayload(key: string, type: "text" | "object" = "object"): Promise<any | undefined> {
  try {
    let records = parseHash(window.location.hash);
    if (records[key] == undefined) {
      return undefined;
    } else {
      if (type == "text") {
        return records[key];
      } else if (type == "object") {
        return await deserializePayload(records[key]);
      } else {
        return undefined;
      }
    }
  } catch (e) {
    return undefined;
  }
}

export async function setQueryPayload(key: string, object: any, type: "text" | "object" = "object") {
  let value: string | undefined = undefined;
  if (object !== undefined) {
    if (type == "text") {
      if (typeof object != "string") {
        return;
      }
      value = object;
    } else if (type == "object") {
      value = await serializePayload(object);
    } else {
      return;
    }
  }

  let records = parseHash(window.location.hash);
  if (value === undefined) {
    delete records[key];
  } else {
    records[key] = value;
  }
  window.location.hash = updateHash(window.location.hash, records);
}

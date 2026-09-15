// Copyright (c) 2026. Licensed under MIT License.

/** Extract 2D Point coordinates from geometry values commonly produced by Parquet and JSON readers. */
export function pointCoordinatesFromGeometry(value: unknown): [number, number] | null {
  if (typeof value === "string") {
    const text = value.trim();
    if (text === "") return null;

    try {
      return pointCoordinatesFromGeometry(JSON.parse(text));
    } catch {
      const match = /^POINT(?:\s+Z(?:M)?|\s+M)?\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)/i.exec(text);
      if (match == null) return null;
      return finitePoint(Number(match[1]), Number(match[2]));
    }
  }

  const bytes = geometryBytes(value);
  if (bytes != null) {
    return pointCoordinatesFromWKB(bytes);
  }

  if (value == null || typeof value !== "object") return null;

  const object = value as Record<string, unknown>;
  if (String(object.type).toLowerCase() === "feature") {
    return pointCoordinatesFromGeometry(object.geometry);
  }
  if (String(object.type).toLowerCase() !== "point") return null;

  const coordinates = numericValues(object.coordinates);
  if (coordinates == null || coordinates.length < 2) return null;
  return finitePoint(coordinates[0], coordinates[1]);
}

function pointCoordinatesFromWKB(bytes: Uint8Array): [number, number] | null {
  if (bytes.byteLength < 21) return null;

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const byteOrder = view.getUint8(0);
  if (byteOrder !== 0 && byteOrder !== 1) return null;

  const littleEndian = byteOrder === 1;
  const rawType = view.getUint32(1, littleEndian);
  const typeWithoutEwkbFlags = rawType & 0x0fffffff;
  const geometryType = typeWithoutEwkbFlags >= 1000 ? typeWithoutEwkbFlags % 1000 : typeWithoutEwkbFlags;
  if (geometryType !== 1) return null;

  let coordinateOffset = 5;
  if ((rawType & 0x20000000) !== 0) coordinateOffset += 4;
  if (bytes.byteLength < coordinateOffset + 16) return null;

  return finitePoint(
    view.getFloat64(coordinateOffset, littleEndian),
    view.getFloat64(coordinateOffset + 8, littleEndian),
  );
}

function geometryBytes(value: unknown): Uint8Array | null {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (value instanceof DataView) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }

  const values = numericValues(value);
  if (values == null || values.length < 21) return null;
  if (values.some((item) => !Number.isInteger(item) || item < 0 || item > 255)) return null;
  return Uint8Array.from(values);
}

function numericValues(value: unknown): number[] | null {
  let values: unknown;
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    values = Array.from(value as ArrayLike<unknown>);
  } else if (value != null && typeof value === "object" && "toArray" in value) {
    const toArray = (value as { toArray?: unknown }).toArray;
    if (typeof toArray !== "function") return null;
    values = Array.from(toArray.call(value) as ArrayLike<unknown>);
  } else {
    return null;
  }

  const numbers = (values as unknown[]).map((item) => Number(item));
  return numbers.every(Number.isFinite) ? numbers : null;
}

function finitePoint(x: number, y: number): [number, number] | null {
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

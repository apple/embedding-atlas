// Copyright (c) 2025 Apple Inc. Licensed under MIT License.
//
// Mosaic-core REST connector that streams the arrow response body and
// hands flechette a per-message ``Uint8Array[]`` instead of one giant
// allocation.
//
// Why: Chromium caps a single ``new Uint8Array(n)`` at ~2 GB on
// macOS (verified empirically: 1.6 GB OK, 2.15 GB and up throw
// ``Array buffer allocation failed`` regardless of free RAM). The
// 322 M-row eubucco scatter response is 2.58 GB raw u32×2, so neither
// ``Response.arrayBuffer()`` nor ``new Uint8Array(contentLength)`` can
// hold it in one block. Server-side zstd doesn't help here — the
// pyarrow zstd codec produces a payload fzstd 0.1.1 cannot decode AND
// silently breaks the 8-byte buffer alignment ``BigInt64Array`` views
// require. So we ship the body uncompressed and split by IPC message
// instead.
//
// Flechette's ``tableFromIPC`` accepts ``Uint8Array | Uint8Array[]``;
// when given an array each entry must contain whole IPC messages
// (messages may NOT span entries — the decode loop processes each
// entry independently). The server cooperates by emitting many small
// batches (~256 K rows per ``RecordBatch`` = ~2 MB body for u32×2),
// so this connector's per-message ``Uint8Array``s never grow large
// enough to hit the 2 GB cap.
//
// Falls back to the upstream ``restConnector`` for non-arrow queries
// (json/exec) where the response is small.

import {
  restConnector,
  type ArrowQueryRequest,
  type Connector,
  type ConnectorQueryRequest,
  type ExecQueryRequest,
  type JSONQueryRequest,
} from "@uwdata/mosaic-core";
import { tableFromIPC, type Table } from "@uwdata/flechette";

/** Same shape as Mosaic's ``restConnector`` options. */
export interface StreamingRestConnectorOptions {
  /** Base URL of the DuckDB REST endpoint (e.g. ``http://127.0.0.1:5088/data/query``). */
  uri: string;
  /** Flechette IPC extraction options. Forwarded to ``tableFromIPC``. */
  ipc?: Parameters<typeof tableFromIPC>[1];
}

/** Sliding window over a queue of network chunks. ``peek``/``take`` work
 *  across chunk boundaries by copying — fast for the small reads we do
 *  (header probes are 8-bytes, metadata is a few KB at most). For the
 *  large per-message body copy we fall through to ``take`` once which
 *  allocates a single right-sized ``Uint8Array``. */
class ChunkQueue {
  private chunks: Uint8Array[] = [];
  /** Bytes already consumed from ``chunks[0]``. */
  private headOffset = 0;
  /** Live byte count = sum of unconsumed bytes across chunks. */
  private buffered = 0;

  push(chunk: Uint8Array): void {
    if (chunk.length === 0) return;
    this.chunks.push(chunk);
    this.buffered += chunk.length;
  }

  available(): number {
    return this.buffered;
  }

  /** Copy the first ``n`` bytes into a fresh ``Uint8Array`` without
   *  consuming. Throws if ``n`` exceeds available. */
  peek(n: number): Uint8Array {
    if (n > this.buffered) {
      throw new Error(`peek(${n}) exceeds available ${this.buffered}`);
    }
    const first = this.chunks[0];
    const firstAvail = first.length - this.headOffset;
    // Hot path: entirely inside the first chunk → return a view, no copy.
    if (n <= firstAvail) {
      return first.subarray(this.headOffset, this.headOffset + n);
    }
    const out = new Uint8Array(n);
    let written = 0;
    let i = 0;
    let off = this.headOffset;
    while (written < n) {
      const c = this.chunks[i];
      const a = c.length - off;
      const take = Math.min(n - written, a);
      out.set(c.subarray(off, off + take), written);
      written += take;
      if (take === a) {
        i++;
        off = 0;
      } else {
        off += take;
      }
    }
    return out;
  }

  /** Consume the first ``n`` bytes and return them as a single
   *  ``Uint8Array`` (copying across chunks if needed). */
  take(n: number): Uint8Array {
    const out = this.peek(n);
    this.advance(n);
    // peek may have returned a subarray view of chunks[0]; advance
    // doesn't invalidate it because we don't mutate the underlying
    // storage, only drop our reference. So returning the view is safe.
    return out;
  }

  /** Drop the first ``n`` bytes from the queue. */
  advance(n: number): void {
    if (n > this.buffered) {
      throw new Error(`advance(${n}) exceeds available ${this.buffered}`);
    }
    this.buffered -= n;
    let remaining = n;
    while (remaining > 0) {
      const c = this.chunks[0];
      const a = c.length - this.headOffset;
      if (remaining < a) {
        this.headOffset += remaining;
        return;
      }
      remaining -= a;
      this.chunks.shift();
      this.headOffset = 0;
    }
  }
}

function readInt32LE(b: Uint8Array, o: number): number {
  return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24);
}

function readInt16LE(b: Uint8Array, o: number): number {
  return ((b[o] | (b[o + 1] << 8)) << 16) >> 16;
}

/** Read the ``bodyLength`` field from an Arrow IPC ``Message`` flatbuffer.
 *
 *  The Message table layout (from the Arrow ``Message.fbs`` schema, in
 *  vtable byte-offset order):
 *    4: version           (int16)
 *    6: header_type       (uint8)  — the "MessageHeader" union tag
 *    8: header            (table)  — union payload offset
 *   10: bodyLength        (int64)
 *   12: custom_metadata   (vector)
 *
 *  Mirrors flechette's ``readObject(metadata, 0)``-based reads in
 *  ``decode/message.js``: vtable[index] gives a uint16 byte offset
 *  inside the table; if non-zero, read the value at ``tablePos + off``.
 *  We need this here (rather than calling flechette's decodeMessage)
 *  because we want the body LENGTH only — not the body bytes — so we
 *  can decide if the next message is fully buffered. */
function readMessageBodyLength(metadata: Uint8Array): number {
  const tablePos = readInt32LE(metadata, 0);
  const vtablePos = tablePos - readInt32LE(metadata, tablePos);
  const vtableSize = readInt16LE(metadata, vtablePos);
  const FIELD_OFFSET = 10;
  if (FIELD_OFFSET >= vtableSize) return 0;
  const off = readInt16LE(metadata, vtablePos + FIELD_OFFSET);
  if (off === 0) return 0;
  // bodyLength is int64 LE; safe to coalesce to JS Number — Arrow
  // record-batch bodies max out well below 2^53.
  const at = tablePos + off;
  const lo = readInt32LE(metadata, at) >>> 0;
  const hi = readInt32LE(metadata, at + 4);
  return hi * 0x1_0000_0000 + lo;
}

/** Stream the response body as a sequence of complete IPC messages.
 *  Each returned ``Uint8Array`` contains exactly one message
 *  (continuation marker + metadata length + metadata + body) — small
 *  enough that even on the 322 M-row scatter we never allocate more
 *  than ~2 MB at once on the JS heap. */
async function readBodyAsMessages(res: Response): Promise<Uint8Array[]> {
  const reader = res.body!.getReader();
  const queue = new ChunkQueue();
  const messages: Uint8Array[] = [];
  let eos = false;

  // Try to extract one message at the head of the queue. Returns
  // false if not enough buffered yet, true if a message was emitted
  // (or the stream EOS marker was consumed).
  const tryExtract = (): boolean => {
    // Need at least 4 bytes to read the prefix.
    if (queue.available() < 4) return false;
    const prefix = queue.peek(8 <= queue.available() ? 8 : 4);
    let metadataLen = readInt32LE(prefix, 0);
    let prefixBytes = 4;
    if (metadataLen === -1) {
      // Modern format: 4-byte continuation marker + 4-byte length.
      if (queue.available() < 8) return false;
      metadataLen = readInt32LE(prefix, 4);
      prefixBytes = 8;
    }
    if (metadataLen === 0) {
      // End-of-stream marker (zero-length metadata after the
      // continuation prefix). Consume and signal EOS.
      queue.advance(prefixBytes);
      eos = true;
      return true;
    }
    if (queue.available() < prefixBytes + metadataLen) return false;
    const head = queue.peek(prefixBytes + metadataLen);
    const metadata = head.subarray(prefixBytes);
    const bodyLen = readMessageBodyLength(metadata);
    const total = prefixBytes + metadataLen + bodyLen;
    if (queue.available() < total) return false;
    messages.push(queue.take(total));
    return true;
  };

  for (;;) {
    const r = await reader.read();
    if (r.value && r.value.length > 0) queue.push(r.value);
    while (tryExtract()) {
      if (eos) break;
    }
    if (eos) break;
    if (r.done) break;
  }

  return messages;
}

/** Mosaic-core connector with the streaming-arrow override. Initialise
 *  it the same way as ``restConnector`` and pass the result to
 *  ``coordinator.databaseConnector(...)``. */
export function streamingRestConnector(options: StreamingRestConnectorOptions): Connector {
  const baseConnector = restConnector({ uri: options.uri, ipc: options.ipc });
  const ipcOptions = options.ipc;
  async function query(query: ArrowQueryRequest): Promise<Table>;
  async function query(query: ExecQueryRequest): Promise<void>;
  async function query(query: JSONQueryRequest): Promise<Record<string, unknown>[]>;
  async function query(query: ConnectorQueryRequest): Promise<unknown> {
    // Non-arrow queries (small JSON / exec) go through the upstream
    // connector unchanged.
    if (query.type !== "arrow") {
      return baseConnector.query(query as any);
    }
    const res = await fetch(options.uri, {
      method: "POST",
      mode: "cors",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(query),
    });
    if (!res.ok) {
      throw new Error(`[atlas-stream] arrow query failed with HTTP ${res.status}: ${await res.text()}`);
    }
    const messages = await readBodyAsMessages(res);
    return tableFromIPC(messages, ipcOptions);
  }
  return {
    query,
  };
}

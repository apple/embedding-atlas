// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export function downloadBuffer(arrayBuffer: ArrayBuffer | Uint8Array<ArrayBuffer>, fileName: string) {
  let a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([arrayBuffer], { type: "application/octet-stream" }));
  a.download = fileName;
  a.click();
}

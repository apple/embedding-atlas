// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

let colorCanvasContext: CanvasRenderingContext2D | null = null;
let colorCache = new Map<string, { r: number; g: number; b: number; a: number }>();

export function parseCSSColor(color: string): { r: number; g: number; b: number; a: number } {
  let cached = colorCache.get(color);
  if (cached != null) {
    return cached;
  }
  if (colorCanvasContext == null) {
    let canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    colorCanvasContext = canvas.getContext("2d")!;
  }
  colorCanvasContext.fillStyle = color;
  colorCanvasContext.clearRect(0, 0, 1, 1);
  colorCanvasContext.fillRect(0, 0, 1, 1);
  let result = [...colorCanvasContext.getImageData(0, 0, 1, 1).data];
  let parsed = { r: result[0], g: result[1], b: result[2], a: result[3] };
  colorCache.set(color, parsed);
  return parsed;
}

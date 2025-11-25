// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { type Matrix3 } from "./matrix.js";
import type { Point, ViewportState } from "./utils.js";

export class Viewport {
  private viewport: ViewportState;
  private width: number;
  private height: number;

  private _matrix: Matrix3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  private _pixel_kx: number = 0;
  private _pixel_bx: number = 0;
  private _pixel_ky: number = 0;
  private _pixel_by: number = 0;

  constructor(viewport: ViewportState, width: number, height: number) {
    this.viewport = viewport;
    this.width = width;
    this.height = height;
    this.updateCoefficients();
  }

  update(viewport: ViewportState, width: number, height: number) {
    this.viewport = viewport;
    this.width = width;
    this.height = height;
    this.updateCoefficients();
  }

  private updateCoefficients() {
    let { x, y, scale } = this.viewport;
    let sx = scale;
    let sy = scale;
    if (this.width < this.height) {
      sx *= this.height / this.width;
    } else {
      sy *= this.width / this.height;
    }
    this._matrix = [sx, 0, 0, 0, sy, 0, -x * sx, -y * sy, 1];
    this._pixel_kx = (this._matrix[0] * this.width) / 2;
    this._pixel_bx = ((this._matrix[6] + 1) * this.width) / 2;
    this._pixel_ky = (-this._matrix[4] * this.height) / 2;
    this._pixel_by = ((-this._matrix[7] + 1) * this.height) / 2;
  }

  matrix(): Matrix3 {
    return this._matrix;
  }

  scale(): number {
    return Math.abs(this._pixel_kx);
  }

  pixelLocation(x: number, y: number): Point {
    return { x: x * this._pixel_kx + this._pixel_bx, y: y * this._pixel_ky + this._pixel_by };
  }

  coordinateAtPixel(px: number, py: number): Point {
    return { x: (px - this._pixel_bx) / this._pixel_kx, y: (py - this._pixel_by) / this._pixel_ky };
  }

  pixelLocationFunction(): (x: number, y: number) => Point {
    let kx = this._pixel_kx;
    let ky = this._pixel_ky;
    let bx = this._pixel_bx;
    let by = this._pixel_by;
    return (x, y) => ({ x: x * kx + bx, y: y * ky + by });
  }

  coordinateAtPixelFunction(): (px: number, py: number) => Point {
    let kx = this._pixel_kx;
    let ky = this._pixel_ky;
    let bx = this._pixel_bx;
    let by = this._pixel_by;
    return (px, py) => ({ x: (px - bx) / kx, y: (py - by) / ky });
  }
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export class OccupancyMap {
  private numColumns: number;
  private rows: Uint32Array;

  constructor(numColumns: number) {
    if (numColumns > 32) {
      throw new Error("numColumns must be <= 32");
    }
    this.numColumns = numColumns;
    this.rows = new Uint32Array(128);
  }

  private ensureSize(count: number) {
    if (this.rows.length < count) {
      let newLength = Math.ceil(this.rows.length * 1.5);
      while (newLength < count) newLength = Math.ceil(newLength * 1.5);
      let newRows = new Uint32Array(newLength);
      newRows.set(this.rows, 0);
      this.rows = newRows;
    }
  }

  clone(): OccupancyMap {
    let r = new OccupancyMap(this.numColumns);
    r.rows = new Uint32Array(this.rows);
    return r;
  }

  /** Get the bit at (x, y) */
  get(x: number, y: number): boolean {
    if (x < 0 || x >= this.numColumns || y < 0 || y >= this.rows.length) {
      return false;
    }
    return (this.rows[y] & (1 << x)) != 0;
  }

  /** Set the bit at (x, y) */
  set(x: number, y: number, value: number) {
    if (x < 0 || x >= this.numColumns || y < 0) {
      return;
    }
    this.ensureSize(y + 1);
    if (value == 0) {
      this.rows[y] &= ~(1 << x);
    } else if (value == 1) {
      this.rows[y] |= 1 << x;
    }
  }

  /** Check if the given rect is occupied, return true if any bit is true */
  check(x: number, y: number, width: number, height: number): boolean {
    if (x < 0) {
      width += x;
      x = 0;
    }
    if (y < 0) {
      height += y;
      y = 0;
    }
    if (x + width > this.numColumns) {
      width = this.numColumns - x;
    }
    if (width <= 0 || height <= 0) {
      return true;
    }
    let mask = ((1 << width) - 1) << x;
    for (let dy = 0; dy < height; dy++) {
      if (y + dy < this.rows.length && (this.rows[y + dy] & mask) != 0) {
        return false;
      }
    }
    return true;
  }

  /** Fill the given rect as occupied */
  fill(x: number, y: number, width: number, height: number) {
    if (x < 0) {
      width += x;
      x = 0;
    }
    if (y < 0) {
      height += y;
      y = 0;
    }
    if (x + width > this.numColumns) {
      width = this.numColumns - x;
    }
    if (width <= 0 || height <= 0) {
      return;
    }
    const mask = ((1 << width) - 1) << x;
    this.ensureSize(y + height);
    for (let dy = 0; dy < height; dy++) {
      this.rows[y + dy] |= mask;
    }
  }

  /** Find a start location (x, y) with a rect of size (width, height) can be placed */
  find(width: number, height: number): { x: number; y: number } {
    if (width <= 0 || height <= 0 || width > this.numColumns) {
      throw new Error("invalid dimensions");
    }
    let maxY = this.rows.length;
    for (let y = 0; y < maxY; y++) {
      for (let x = 0; x <= this.numColumns - width; x++) {
        if (this.check(x, y, width, height)) {
          return { x, y };
        }
      }
    }
    let newY = this.rows.length;
    return { x: 0, y: newY };
  }

  maxOccupiedY(): number {
    for (let i = this.rows.length - 1; i >= 0; i--) {
      if (this.rows[i] != 0) {
        return i;
      }
    }
    return -1;
  }

  /** Return unsued rects with the given size constraints */
  unusedRects(
    minWidth: number,
    minHeight: number,
    maxWidth: number,
    maxHeight: number,
  ): { x: number; y: number; width: number; height: number }[] {
    let map = this.clone();
    let result: { x: number; y: number; width: number; height: number }[] = [];
    let maxY = this.maxOccupiedY();
    // Find all unused rectangular areas that meet the size constraints
    for (let y = 0; y <= maxY; y++) {
      for (let x = 0; x < map.numColumns; x++) {
        // Skip if this position is already occupied
        if (map.get(x, y)) {
          continue;
        }
        // Find the maximum width and height for a rectangle starting at (x, y)
        let maxW = 0;
        for (let w = 1; w <= Math.min(maxWidth, map.numColumns - x) && !map.get(x + w - 1, y); w++) {
          maxW = w;
        }

        // If we can't meet minimum width, skip
        if (maxW < minWidth) {
          continue;
        }

        let maxH = 0;

        // Find maximum height for each possible width
        for (let w = maxW; w >= minWidth; w--) {
          maxH = 0;
          for (let h = 1; h <= Math.min(maxHeight, map.rows.length - y); h++) {
            // Check if the entire row segment is free
            let rowFree = true;
            for (let dx = 0; dx < w; dx++) {
              if (map.get(x + dx, y + h - 1)) {
                rowFree = false;
                break;
              }
            }
            if (!rowFree) {
              break;
            }
            maxH = h;
          }

          if (y + maxH > maxY + 1) {
            maxH = maxY + 1 - y;
          }

          // If we found a valid rectangle, add it and mark the area as used
          if (maxH >= minHeight) {
            result.push({ x, y, width: w, height: maxH });

            // Mark this rectangle as occupied in our working map
            map.fill(x, y, w, maxH);
            break; // Move to next position since we've used this area
          }
        }
      }
    }

    return result;
  }
}

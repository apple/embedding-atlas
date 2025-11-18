export function solveExtents(
  data: [number, number][],
  length: number,
  lengthMode: "inner" | "outer",
): [number, number] {
  if (data.length == 0) {
    return [0, 0];
  }

  if (lengthMode == "inner") {
    let start = 0;
    let end = 0;
    for (let [pos, offset] of data) {
      let x = pos * length + offset;
      if (x < 0) {
        start = Math.max(start, -x);
      }
      if (x > length) {
        end = Math.max(end, x - length);
      }
    }
    return [start, end];
  } else {
    // Simple greedy algorithm.
    let start = 0;
    let end = 0;
    for (let [pos, offset] of data) {
      // x = (length - start - end) * pos + start + offset
      // x >= 0 and x <= length
      let x = (length - start - end) * pos + start + offset;
      if (x < 0 && pos != 1) {
        // (length - end) * pos + (1 - pos) * start + offset = 0
        start = ((length - end) * pos + offset) / (pos - 1);
      }
      if (x > length && pos != 0) {
        // (length - start) * pos - end * pos + start + offset = length
        end = ((length - start) * pos + start + offset - length) / pos;
      }
    }
    return [start, end];
  }
}

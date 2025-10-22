// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ViewportState } from "@embedding-atlas/component";

function mix(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export function interpolateViewport(s1: ViewportState, s2: ViewportState, t: number): ViewportState {
  let ls1 = Math.log(s1.scale);
  let ls2 = Math.log(s2.scale);
  if (Math.abs(ls2 - ls1) < 1e-5) {
    return {
      x: mix(s1.x, s2.x, t),
      y: mix(s1.y, s2.y, t),
      scale: mix(s1.scale, s2.scale, t),
    };
  }
  let scale = Math.exp(mix(ls1, ls2, t));
  // Find a path such that a point at the center of the view moves in a constant speed,
  // while the scale interpolates in the log domain.
  //
  // px = (x - x0) * scale
  // dpx/dt = d(x - x0)/dt * scale + (x - x0) * dscale/dt
  //        = -dx0/dt * scale + (x - x0) * dscale/dt
  // let x = x0, we have:
  // dpx/dt@x=x0 = -dx0/dt * scale
  // Therefore the goal is to find a x0(t) such that dx0/dt * scale is constant.
  //
  // dx0/dt = constant * exp(-ls1 - (ls2 - ls1) * t)
  // x0(t) = A + B * exp(-ls1 - (ls2 - ls1) * t) = A + B / scale(t)
  // s1.x * s1.scale = A * s1.scale + B
  // s2.x * s2.scale = A * s2.scale + B
  // A = (s2.x * s2.scale - s1.x * s1.scale) / (s2.scale - s1.scale)
  // B = (s1.x - s2.x) * s1.scale * s2.scale / (s2.scale - s1.scale)
  return {
    x: (s2.x * s2.scale - s1.x * s1.scale + (s1.x - s2.x) * ((s1.scale * s2.scale) / scale)) / (s2.scale - s1.scale),
    y: (s2.y * s2.scale - s1.y * s1.scale + (s1.y - s2.y) * ((s1.scale * s2.scale) / scale)) / (s2.scale - s1.scale),
    scale: scale,
  };
}

// Copyright (c) 2025 Apple Inc. Licensed under MIT License.
//
// Pass 2: ribbons (linear density accumulation).
// For each non-empty bin texel, draw a ribbon connecting the left axis value-range to the
// right axis value-range, and additively accumulate the bin's (Σ linear color, count) into
// a single screen target. Accumulation is LINEAR (the order-independent-transparency tone-map
// is applied once, in the composite pass, to the accumulated total). Because accumulation is
// linear, the accumulated density at a pixel depends only on the total rows covering it, not on
// how they are partitioned by the neighbor axis. This makes a vertical slice of density
// continuous across a shared axis (both fans converge to the same per-axis marginal), and keeps
// the OIT opacity partition-invariant. The density -> color/alpha tone-mapping happens once, in
// the composite pass.
//
// Density modulation == anti-aliasing. Within a bin we assume uniform density over the
// (left value, right value) unit square, so a row at sub-bin position (u, v) sits at band
// fraction o = (1-t)*u + t*v at horizontal fraction t. For u,v ~ U[0,1] the line density
// across the band is f(o; t) = pdf of o -- a trapezoid: flat (==1) at the axes, narrowing to
// a triangle peaking at 2 at mid-span (t=0.5), and ->0 at both band edges. We deposit that f
// (not a flat fill) and obtain it by box-filtering its analytic CDF F over each pixel's
// footprint: (F(o+h) - F(o-h)) / (2h). In the interior this -> f(o) (the modulation); at the
// band edges F saturates at 0/1, so the same expression ramps coverage to 0 -- i.e. it is
// also the edge AA, no separate feather needed.
//
// Adjacent bins mesh exactly. Diagonal neighbors share an edge (bottom of (bx,by) == top of
// (bx+1,by+1)) and tile with identical slope. A pixel straddling that edge gets, from each
// bin, exactly the mass of that bin's density falling inside the pixel (F clamps past the
// shared edge); additive blending sums them to the true area-average regardless of the two
// counts, and the shared slope means an identical 2h divisor -- so neighbors join seamlessly
// under OIT/additive blending with no dark seam and no double-bright line.
//
// Per-ribbon contributions are scaled by 1/num_rows to keep the accumulator small. The
// modulation can push a pixel's total above 1 (up to ~2x near mid-span centers); the
// composite's log tone-map + clamp absorbs it.

struct Shared {
  bin_count: u32,
  num_fields: u32,
  num_rows: u32,
  _pad0: u32,
  opacity: f32,
  max_count: f32,
  gamma: f32,
  _pad1: f32,
  plot_y1: f32,
  plot_y2: f32,
  width: f32,   // framebuffer width in device pixels (for the AA pixel footprint)
  height: f32,  // framebuffer height in device pixels
  background: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Shared;

@group(1) @binding(0) var bins: texture_2d_array<f32>;     // per-pair (Σ linear color, count)
@group(1) @binding(1) var<storage, read> axis_x: array<f32>; // NDC x per axis

struct RibbonVSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) color_sum: vec3<f32>, // Σ linear color for the bin
  @location(1) @interpolate(flat) weight: f32,          // bin count
  @location(2) ov: f32,                                 // band fraction: 0 at top edge, 1 at bottom edge
  @location(3) tx: f32,                                 // horizontal fraction: 0 at left axis, 1 at right axis
}

@vertex
fn ribbon_vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> RibbonVSOut {
  var out: RibbonVSOut;

  let bc = u.bin_count;
  let cells = bc * bc;
  let p = iid / cells;
  let texel = iid % cells;
  let bx = texel % bc;
  let by = texel / bc;

  let t = textureLoad(bins, vec2<i32>(i32(bx), i32(by)), i32(p), 0);
  let count = t.a;
  if (count <= 0.0) {
    out.position = vec4(2.0, 2.0, 0.0, 1.0);
    out.color_sum = vec3(0.0);
    out.weight = 0.0;
    out.ov = 0.0;
    out.tx = 0.0;
    return out;
  }

  let left_x = axis_x[p];
  let right_x = axis_x[p + 1u];

  // The bin spans a value range on each axis; the ribbon is a parallelogram connecting them
  // (top and bottom edges share one slope, since each bin has the same vertical thickness on
  // both axes).
  let l_top = mix(u.plot_y1, u.plot_y2, f32(bx) / f32(bc));
  let l_bot = mix(u.plot_y1, u.plot_y2, f32(bx + 1u) / f32(bc));
  let r_top = mix(u.plot_y1, u.plot_y2, f32(by) / f32(bc));
  let r_bot = mix(u.plot_y1, u.plot_y2, f32(by + 1u) / f32(bc));

  // Expand the band outward by ~1px perpendicular so the rasterizer generates the fragments
  // the edge AA needs; the fragment shader's box-filtered CDF ramps coverage to 0 across that
  // pad (and deposits 0 beyond the band, so overlap into a neighbor is harmless). x stays
  // pinned to the axes (no left/right expansion).
  let dxp = (right_x - left_x) * 0.5 * u.width;     // edge vector in device px
  let dyp = (r_top - l_top) * 0.5 * u.height;       // top-edge slope == bottom-edge slope
  let len = max(sqrt(dxp * dxp + dyp * dyp), 1e-6);
  let cos_t = abs(dxp) / len;                       // cos of the edge angle from horizontal
  let pad = 1.0;                                    // perpendicular pad in device px
  // Vertical NDC offset that expands the band by `pad` px PERPENDICULAR to the edge. A vertical shift dv
  // contributes dv*cos_t in the perpendicular direction, so dv = pad/cos_t keeps the perpendicular pad at
  // ~1px for ANY slope (steep edges just get a taller vertical expansion). cos_t is floored only at a tiny
  // epsilon to guard divide-by-zero for a degenerate (zero-width) axis gap. The extra vertical extent is
  // zero-deposit tail (trap_cdf clamps beyond [0,1]), and `ov` still encodes the true band fraction
  // independent of dv, so this neither bleeds into neighbors nor changes adjacent-band meshing.
  let dv = (pad / max(cos_t, 1e-3)) * 2.0 / u.height;
  let band_ndc = max(abs(l_top - l_bot), 1e-6);     // band vertical thickness in NDC
  let e = dv / band_ndc;                            // pad expressed in band-fraction units

  let lb = vec2(left_x, l_bot - dv);
  let lt = vec2(left_x, l_top + dv);
  let rb = vec2(right_x, r_bot - dv);
  let rt = vec2(right_x, r_top + dv);

  // Band fraction `ov` is 0 on the top edge and 1 on the bottom edge; the expanded corners get
  // -e / 1+e so the interpolated value stays exact inside the AA pad. `tx` is the horizontal
  // fraction (0 at the left axis, 1 at the right). Both are affine over the parallelogram, so
  // triangle interpolation reproduces them exactly.
  var pos: vec2<f32>;
  var ov: f32;
  var tx: f32;
  switch vid {
    case 0u: { pos = lb; ov = 1.0 + e; tx = 0.0; }
    case 1u: { pos = lt; ov = -e;      tx = 0.0; }
    case 2u: { pos = rb; ov = 1.0 + e; tx = 1.0; }
    case 3u: { pos = rb; ov = 1.0 + e; tx = 1.0; }
    case 4u: { pos = lt; ov = -e;      tx = 0.0; }
    case 5u: { pos = rt; ov = -e;      tx = 1.0; }
    default: { pos = lb; ov = 1.0 + e; tx = 0.0; }
  }

  out.position = vec4(pos, 0.0, 1.0);
  out.color_sum = t.rgb;
  out.weight = count;
  out.ov = ov;
  out.tx = tx;
  return out;
}

// CDF of the cross-section density f(o; t) for a uniformly-filled square bin, where
// o = (1-t)*u + t*v with u,v ~ U[0,1]. f is the trapezoid (a triangle at t=0.5); F is its
// integral, clamped to [0,1] outside the band [0,1]. Box-filtering F over a pixel both
// reproduces f (modulation) and ramps coverage at the band edges (AA), and because F conserves
// mass, edge-sharing neighbors deposit complementary amounts that mesh under additive blending.
fn trap_cdf(o: f32, t: f32) -> f32 {
  if (o <= 0.0) { return 0.0; }
  if (o >= 1.0) { return 1.0; }
  let a = 1.0 - t;
  let b = t;
  let ab = a * b;                  // = t(1-t)
  if (ab < 1e-6) { return o; }     // at an axis the cross-section is uniform: F(o) = o
  let M = max(a, b);
  let m = 1.0 - M;                 // = min(a, b); note ab == m*M
  if (o < m) {
    return o * o / (2.0 * ab);
  } else if (o > M) {
    let r = 1.0 - o;
    return 1.0 - r * r / (2.0 * ab);
  }
  return (o - 0.5 * m) / M;
}

struct RibbonFSOut {
  @location(0) accum: vec4<f32>, // Σ (premultiplied linear color, count) × density / num_rows
  @location(1) coverage: f32,    // Σ geometric edge coverage (for anti-aliasing, independent of density)
}

@fragment
fn ribbon_fs(in: RibbonVSOut) -> RibbonFSOut {
  // Box-filter over the pixel's footprint in band-fraction units.
  let half = max(0.5 * fwidth(in.ov), 1e-5);
  let lo = in.ov - half;
  let hi = in.ov + half;
  // Data density (trapezoid modulation, box-filtered): drives the OIT opacity. -> f(o) in the interior,
  // ramps at the band edges. `hi - lo` matches for edge-sharing neighbors, so bins mesh.
  let density = (trap_cdf(hi, in.tx) - trap_cdf(lo, in.tx)) / (hi - lo);
  // Geometric coverage: fraction of the footprint inside the band [0,1] -- the pure edge AA ramp (1 in
  // the interior, ramping to 0 at the band edges). Adjacent bins' ramps sum to 1, so seams stay covered;
  // the composite uses this to apply AA as a linear multiplier on top of the capped opacity.
  let coverage = (clamp(hi, 0.0, 1.0) - clamp(lo, 0.0, 1.0)) / (hi - lo);

  // Scale by 1/num_rows so the additive accumulator stays small. Both color_sum and weight get the
  // same factor so the composite's mean color (rgb/a) is unchanged.
  let scale = select(0.0, 1.0 / f32(u.num_rows), u.num_rows > 0u) * density;
  var out: RibbonFSOut;
  out.accum = vec4(in.color_sum * scale, in.weight * scale);
  out.coverage = coverage;
  return out;
}

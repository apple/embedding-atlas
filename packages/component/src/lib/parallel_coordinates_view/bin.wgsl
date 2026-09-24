// Copyright (c) 2025 Apple Inc. Licensed under MIT License.
//
// Pass 1: bin-space color blend.
// Render every row as a 1px point into a binCount x binCount texture (one array layer
// per axis pair), accumulating (sum premultiplied color, count) with hardware additive
// blending. No atomics, no per-category counters -- the cost is independent of the number
// of colors and works for continuous color scales. Colors are accumulated in linear space.

struct Shared {
  bin_count: u32,
  num_fields: u32,
  num_rows: u32,
  lut_size: u32,        // number of real color LUT entries (<= 256)
  opacity: f32,
  max_count: f32,
  gamma: f32,
  color_mode: u32,      // 0 = index (categorical), 1 = interpolate (continuous value)
  plot_y1: f32,
  plot_y2: f32,
  _pad2: f32,
  _pad3: f32,
  background: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Shared;

@group(1) @binding(0) var<storage, read> values: array<f32>;       // row-major, numRows * numFields, in [0,1]
@group(1) @binding(1) var<storage, read> color_value: array<f32>;  // per row: LUT index (categorical) or [0,1] value (continuous)
@group(1) @binding(2) var color_lut: texture_2d<f32>;              // 256 x 1, sRGB; first lut_size entries are real

struct PairParams {
  dim_left: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(2) @binding(0) var<uniform> pair: PairParams;

struct BinVSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) color: vec4<f32>,
}

@vertex
fn bin_vs(@builtin(instance_index) row: u32) -> BinVSOut {
  var out: BinVSOut;

  let nf = u.num_fields;
  let base = row * nf;
  let dl = pair.dim_left;
  let v_left = values[base + dl];
  let v_right = values[base + dl + 1u];

  // Guard non-finite (NaN fails the comparisons) and out-of-range values: skip the row
  // for this pair by emitting a vertex outside the clip volume.
  if (!(v_left >= 0.0) || !(v_left <= 1.0) || !(v_right >= 0.0) || !(v_right <= 1.0)) {
    out.position = vec4(2.0, 2.0, 0.0, 1.0);
    out.color = vec4(0.0);
    return out;
  }

  let bc = u.bin_count;
  let bx = min(u32(v_left * f32(bc)), bc - 1u);
  let by = min(u32(v_right * f32(bc)), bc - 1u);

  // Place a 1px point at the center of bin (bx, by). Framebuffer y is flipped vs NDC.
  let nx = (f32(bx) + 0.5) / f32(bc) * 2.0 - 1.0;
  let ny = 1.0 - (f32(by) + 0.5) / f32(bc) * 2.0;
  out.position = vec4(nx, ny, 0.0, 1.0);

  // Look up the row's color from the LUT and accumulate the linear color (composited in bin space).
  //  - index (categorical): the value is a direct LUT texel index.
  //  - interpolate (continuous): the [0,1] value maps across the LUT's real entries (0 -> first real
  //    texel, 1 -> last real texel), linearly interpolated and clamped.
  let cv = color_value[row];
  let last = max(0, i32(u.lut_size) - 1);
  var srgb: vec3<f32>;
  if (u.color_mode == 0u) {
    let idx = clamp(i32(round(cv)), 0, last);
    srgb = textureLoad(color_lut, vec2<i32>(idx, 0), 0).rgb;
  } else {
    let f = clamp(cv, 0.0, 1.0) * f32(last);
    let i0 = i32(floor(f));
    let i1 = min(i0 + 1, last);
    let c0 = textureLoad(color_lut, vec2<i32>(i0, 0), 0).rgb;
    let c1 = textureLoad(color_lut, vec2<i32>(i1, 0), 0).rgb;
    srgb = mix(c0, c1, f - f32(i0));
  }
  let lin = pow(srgb, vec3(u.gamma));
  out.color = vec4(lin, 1.0); // premultiplied linear color + unit weight
  return out;
}

@fragment
fn bin_fs(in: BinVSOut) -> @location(0) vec4<f32> {
  return in.color;
}

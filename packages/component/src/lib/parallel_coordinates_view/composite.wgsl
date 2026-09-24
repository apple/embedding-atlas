// Copyright (c) 2025 Apple Inc. Licensed under MIT License.
//
// Pass 3: composite.
// The accumulation target holds, per pixel, (Σ linear color, Σ count) scaled by 1/num_rows.
// Recover the density-weighted mean color and the order-independent-transparency opacity from
// the total density, then composite over the theme background and convert from linear back to
// sRGB. The opacity uses the embedding view's `1 - exp(Σ log(1-α))` recovery (see below).

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
  _pad2: f32,
  _pad3: f32,
  background: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Shared;

@group(1) @binding(0) var accum_tex: texture_2d<f32>;
@group(1) @binding(1) var coverage_tex: texture_2d<f32>;

struct CompVSOut {
  @builtin(position) position: vec4<f32>,
}

@vertex
fn composite_vs(@builtin(vertex_index) vid: u32) -> CompVSOut {
  var positions = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var out: CompVSOut;
  out.position = vec4(positions[vid], 0.0, 1.0);
  return out;
}

@fragment
fn composite_fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(i32(frag.x), i32(frag.y));
  let acc = textureLoad(accum_tex, p, 0);
  let cov = textureLoad(coverage_tex, p, 0).r;    // Σ geometric coverage (~1 in the interior, ramps at edges)

  // Composite in LINEAR light: colors are accumulated linear and the background is linearized CPU-side, so
  // the line-over-background blend (a light-mixing operation) is done linearly and encoded to the screen's
  // sRGB gamma once, at the end. Anti-aliased edges still ramp smoothly because the ramp lives in `alpha`
  // (the coverage multiplier), independent of the encode.
  let bg_lin = u.background.rgb;
  var rgb = bg_lin;

  if (acc.a > 0.0 && cov > 0.0) {
    let mean_lin = acc.rgb / acc.a;                 // density-weighted mean color (linear)
    let total_count = acc.a * f32(u.num_rows);      // recovered count (density-weighted; includes the edge ramp)
    let k = -log(1.0 - clamp(u.opacity, 0.0, 0.9999)); // per-row extinction

    // Anti-aliasing: keep geometric coverage separate from the density-driven opacity so opacity can
    // SATURATE (cap at 1 via the exp) BEFORE anti-aliasing, then apply AA as a LINEAR multiply. `covc` is
    // the pixel's geometric coverage in [0,1]; `count_solid = total_count / covc` divides the edge ramp
    // back out to get the density as if the pixel were fully covered. Opacity saturates on count_solid;
    // final alpha = covc * opacity. Interior / meshed regions have covc = 1 (adjacent ramps sum to 1) so
    // they're unchanged; only true silhouette edges (covc < 1) ramp -- linearly, so they stay smooth even
    // at opacity 1 (the original `1 - exp(-k * count * coverage)` crushed that ramp when k*count was large).
    let covc = min(cov, 1.0);
    let count_solid = total_count / max(covc, 1e-4);
    let alpha = covc * clamp(1.0 - exp(-k * count_solid), 0.0, 1.0);

    rgb = mix(bg_lin, mean_lin, alpha);
  }

  let srgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / u.gamma));
  return vec4(srgb, 1.0);
}

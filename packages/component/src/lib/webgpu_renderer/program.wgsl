// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

enable f16;

// Pipeline-overridable workgroup sizes. Apple Silicon GPUs use 32-wide
// SIMDs and 8 SIMDs per core → 256 concurrent threads max. Smaller
// threadgroups yield higher occupancy for latency hiding; bigger ones
// amortize dispatch overhead. `override` lets the host swap values at
// pipeline-creation time without recompiling the WGSL.
override wg_downsample_cull: u32 = 256u;
override wg_density_sample: u32 = 256u;
override wg_compact: u32 = 256u;
override wg_accumulate: u32 = 64u;
override wg_gaussian_blur: u32 = 64u;

// Row-strides used for 2D dispatch. Must satisfy
// stride == wg_size * workgroups_x, i.e. the host must recompute
// workgroups_x = stride / wg_size when it changes the wg_*. Strides
// themselves are independent tunables (smaller → more rows, bigger y
// dimension in dispatch).
override ACCUMULATE_STRIDE: u32 = 4096u;

struct Uniforms {
  count: u32,
  category_count: u32,
  framebuffer_width: i32,
  framebuffer_height: i32,
  density_width: i32,
  density_height: i32,
  gamma: f32,
  point_size: f32,
  point_alpha: f32,
  points_alpha: f32,
  density_scaler: f32,
  quantization_step: f32,
  density_alpha: f32,
  contours_alpha: f32,
  matrix: mat3x3<f32>,
  view_xy_scaler: vec2<f32>,
  kde_causal: vec4<f32>,
  kde_anticausal: vec4<f32>,
  kde_a: vec4<f32>,
  background_color: vec4<f32>,
  category_colors: array<vec4<f32>, 256>,
}

struct DownsampleUniforms {
  render_limit: u32,
  frame_seed: u32,
  density_weight: f32,
  // Chunk offset in workgroup-Y units. Host splits each downsample
  // pass into K command-buffer-sized chunks (each ~count/K threads)
  // so no single MTLCommandBuffer iterates the full 322M-row dataset
  // — keeps each cmd buffer well under Metal's 5 s wall-clock
  // watchdog regardless of how many points are in viewport. Per-thread
  // index = (id.y + chunk_offset_y) * DOWNSAMPLE_STRIDE + id.x.
  chunk_offset_y: u32,
}

struct PointData {
  position: vec3<f32>,
  category: u32,
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) log1malpha: f32, // log(1 - alpha)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read> x_buffer: array<f32>;
@group(1) @binding(1) var<storage, read> y_buffer: array<f32>;
@group(1) @binding(2) var<storage, read> category_buffer: array<u32>;

@group(2) @binding(0) var<storage, read_write> count_buffer: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> blur_buffer: array<f16>;
@group(2) @binding(2) var<storage, read_write> blur_swap_buffer: array<f16>;

@group(3) @binding(0) var framebuffer_sampler: sampler;
@group(3) @binding(1) var color_texture: texture_2d<f32>;
@group(3) @binding(2) var log1malpha_texture: texture_2d<f32>;

// Downsampling bind groups (group 3 for compute shaders)
// WebGPU has a default limit of 4 bind groups, so we use group 3 (not 4)
// 3 storage buffers to stay within 8-buffer limit (3 from group1 + 1 from group2 + 3 from group3 = 7)
@group(3) @binding(0) var<uniform> downsample_uniforms: DownsampleUniforms;
@group(3) @binding(1) var<storage, read_write> downsample_counters: array<atomic<u32>>; // [visible_count, max_density_fixed]
@group(3) @binding(2) var<storage, read_write> point_data: array<f32>; // density (>= 0 means visible with density, < 0 means not visible or not accepted)
// Compaction outputs: a tight list of accepted point indices, plus a 16-byte
// indirect-draw args buffer ([vertexCount, instanceCount, firstVertex, firstInstance]).
// Only `instanceCount` (slot 1) is mutated by the compact_accepted pass via
// atomicAdd; the other slots are pre-initialized on the host.
@group(3) @binding(3) var<storage, read_write> compact_indices: array<u32>;
@group(3) @binding(4) var<storage, read_write> indirect_args: array<atomic<u32>>;

// Separate binding for vertex shader in downsampled draw pipeline
// Uses group 2 since the pipeline only needs groups 0, 1, 2
// (read-only access required by WebGPU for vertex shaders)
@group(2) @binding(0) var<storage, read> point_data_read: array<f32>; // same as point_data above
// Compaction read view used by the indirect-draw vertex shader.
// Bound to a different bind group than point_data_read so the two draw paths
// can share group2 binding0 without conflicting.
@group(2) @binding(1) var<storage, read> compact_indices_read: array<u32>;

fn get_point(index: u32) -> PointData {
  var result: PointData;
  result.position = vec3(x_buffer[index], y_buffer[index], 1.0);
  if (uniforms.category_count > 1) {
    result.category = (category_buffer[index >> 2] >> ((index & 3) << 3)) & 0xff;
  } else {
    result.category = 0;
  }
  return result;
}

const ACCUMULATE_UNIT: u32 = 4096;

fn increment_count(x: i32, y: i32, category: u32, value: u32) {
  let width = uniforms.density_width;
  let height = uniforms.density_height;
  if (x < 0 || x >= width || y < 0 || y >= height || category >= uniforms.category_count || value == 0) {
    return;
  }
  let offset = (y * width + x) + i32(category) * (width * height);
  atomicAdd(&count_buffer[offset], value);
}

@compute @workgroup_size(wg_accumulate, 1)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = uniforms.density_width;
  let height = uniforms.density_height;
  let index = id.y * ACCUMULATE_STRIDE + id.x;
  if (index >= uniforms.count) { return; }
  let point = get_point(index);
  let pos = uniforms.matrix * point.position;
  let x = (pos.x + 1.0) / 2.0 * f32(width) - 0.5;
  let y = (pos.y + 1.0) / 2.0 * f32(height) - 0.5;
  let ix = i32(x);
  let iy = i32(y);
  let tx = x - f32(ix);
  let ty = y - f32(iy);
  let w1: u32 = u32((1 - tx) * (1 - ty) * f32(ACCUMULATE_UNIT));
  let w2: u32 = u32(tx * (1 - ty) * f32(ACCUMULATE_UNIT));
  let w3: u32 = u32((1 - tx) * ty * f32(ACCUMULATE_UNIT));
  let w123 = w1 + w2 + w3;
  var w4: u32 = select(0, ACCUMULATE_UNIT - w123, w123 < ACCUMULATE_UNIT);
  increment_count(ix, iy, point.category, w1);
  increment_count(ix + 1, iy, point.category, w2);
  increment_count(ix, iy + 1, point.category, w3);
  increment_count(ix + 1, iy + 1, point.category, w4);
}

// =====================================================
// Draw Discrete Points
// =====================================================

struct PointsVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) dp: vec3<f32>,
  @location(1) color: vec4<f32>,
}

@vertex
fn points_vs(
  @builtin(instance_index) index: u32,
  @builtin(vertex_index) part: u32,
) -> PointsVertexOutput {
  let framebuffer_size = vec2(f32(uniforms.framebuffer_width), f32(uniforms.framebuffer_height));
  let alpha = uniforms.point_alpha * uniforms.points_alpha;
  let dp = vec2<f32>(f32(part % 2), f32(part / 2)) * 2.0 - 1.0;
  let point = get_point(index);
  let pos = uniforms.matrix * point.position;

  var out: PointsVertexOutput;
  out.position = vec4<f32>(pos.xy + dp * uniforms.point_size / framebuffer_size * 2.0, 0.0, 1.0);
  out.dp = vec3(dp, uniforms.point_size);
  out.color = uniforms.category_colors[point.category] * alpha;
  return out;
}

@fragment
fn points_fs(in: PointsVertexOutput) -> FragmentOutput {
  let r = length(in.dp.xy) * in.dp.z;
  let a = max(0.0, min(1.0, in.dp.z - r));
  var out: FragmentOutput;
  out.color = in.color * a;
  out.log1malpha = log(1 - out.color.a);
  return out;
}

// =====================================================
// Draw Density Map
// =====================================================

struct DrawDensityMapVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) texture_coord: vec2<f32>,
}

@vertex
fn draw_density_map_vs(
  @builtin(vertex_index) part: u32,
) -> DrawDensityMapVertexOutput {
  let framebuffer_size = vec2(f32(uniforms.framebuffer_width), f32(uniforms.framebuffer_height));
  let dp = vec2<f32>(f32(part % 2), f32(part / 2)) * 2.0 - 1.0;
  var out: DrawDensityMapVertexOutput;
  out.position = vec4(dp, 0.0, 1.0);
  out.texture_coord = (vec2(dp.x, dp.y) + 1.0) / 2.0 * framebuffer_size;
  return out;
}

fn get_density_raw(x: i32, y: i32, category: u32) -> f32 {
  let width = uniforms.density_width;
  let height = uniforms.density_height;
  let density_scaler = uniforms.density_scaler;
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return 0.0;
  }
  let offset = (y * width + x) + i32(category) * (width * height);
  return max(0.0, f32(blur_buffer[offset]) * density_scaler);
}

fn get_density(x: f32, y: f32, category: u32) -> f32 {
  let px = x / f32(uniforms.framebuffer_width) * f32(uniforms.density_width) - 0.5;
  let py = y / f32(uniforms.framebuffer_height) * f32(uniforms.density_height) - 0.5;
  let ix = i32(px);
  let iy = i32(py);
  let tx = px - f32(ix);
  let ty = py - f32(iy);
  let v00 = get_density_raw(ix, iy, category);
  let v10 = get_density_raw(ix + 1, iy, category);
  let v01 = get_density_raw(ix, iy + 1, category);
  let v11 = get_density_raw(ix + 1, iy + 1, category);
  return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

fn get_density_quantized(x: f32, y: f32, category: u32) -> f32 {
  let v = get_density(x, y, category);
  return floor(clamp(v, 0, 1) / uniforms.quantization_step);
}

fn get_density_quantized_sobel(x: f32, y: f32, category: u32) -> vec2<f32> {
  let v11 = get_density_quantized(x - 1, y - 1, category);
  let v21 = get_density_quantized(x, y - 1, category);
  let v31 = get_density_quantized(x + 1, y - 1, category);
  let v12 = get_density_quantized(x - 1, y, category);
  let v22 = get_density_quantized(x, y, category);
  let v32 = get_density_quantized(x + 1, y, category);
  let v13 = get_density_quantized(x - 1, y + 1, category);
  let v23 = get_density_quantized(x, y + 1, category);
  let v33 = get_density_quantized(x + 1, y + 1, category);
  let gx = v11 + v12 * 2.0 + v13 - v31 - v32 * 2.0 - v33;
  let gy = v11 + v21 * 2.0 + v31 - v13 - v23 * 2.0 - v33;
  return vec2(gx, gy);
}

@fragment
fn draw_density_map_fs(in: DrawDensityMapVertexOutput) -> FragmentOutput {
  let px = in.texture_coord.x;
  let py = in.texture_coord.y;
  let quantization_step: f32 = uniforms.quantization_step;

  var sum_color: vec4<f32> = vec4(0);
  var sum_log1malpha: f32 = 0.0;

  for (var i: u32 = 0; i < uniforms.category_count; i++) {
    let density = get_density(px, py, i);
    var alpha = min(1.0, floor(density / quantization_step) * quantization_step);
    alpha *= uniforms.density_alpha;
    let color = uniforms.category_colors[i] * alpha;
    sum_color += color;
    sum_log1malpha += log(1 - color.a);
  }

  if (uniforms.contours_alpha > 0.0) {
    for (var i: u32 = 0; i < uniforms.category_count; i++) {
      let sobel = get_density_quantized_sobel(px, py, i);
      let alpha = clamp(length(sobel) * 0.2, 0.0, 1.0) * uniforms.contours_alpha;
      let color = uniforms.category_colors[i] * alpha;
      sum_color += color;
      sum_log1malpha += log(1 - color.a);
    }
  }

  var out: FragmentOutput;
  out.color = sum_color;
  out.log1malpha = sum_log1malpha;
  return out;
}

// =====================================================
// Gamma Correction
// =====================================================

struct GammaCorrectionVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) texture_coord: vec2<f32>,
}

@vertex
fn gamma_correction_vs(
  @builtin(vertex_index) part: u32,
) -> GammaCorrectionVertexOutput {
  let dp = vec2<f32>(f32(part % 2), f32(part / 2)) * 2.0 - 1.0;
  var out: GammaCorrectionVertexOutput;
  out.position = vec4(dp * uniforms.view_xy_scaler, 0.0, 1.0);
  out.texture_coord = (vec2(dp.x, -dp.y) + 1.0) / 2.0;
  return out;
}

@fragment
fn gamma_correction_fs(in: GammaCorrectionVertexOutput) -> @location(0) vec4<f32> {
  let sum_color = textureSample(color_texture, framebuffer_sampler, in.texture_coord);
  let sum_log_one_minus_alpha = textureSample(log1malpha_texture, framebuffer_sampler, in.texture_coord).r;
  var color: vec4<f32>;
  if (sum_color.a > 0.0) {
    color = sum_color / sum_color.a * (1.0 - exp(sum_log_one_minus_alpha));
    color = color + uniforms.background_color * (1 - color.a);
  } else {
    color = uniforms.background_color;
  }
  let rgb = pow(color.rgb, vec3(1.0 / uniforms.gamma));
  return vec4(rgb, color.a);
}

// =====================================================
// Gaussian Blur
// =====================================================

@compute @workgroup_size(wg_gaussian_blur, 1)
fn gaussian_blur_stage_1(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = uniforms.density_width;
  let height = uniforms.density_height;
  let x = id.x;
  if (x >= u32(width)) { return; }
  let start = x + id.y * u32(width * height);
  let count = u32(height);
  let stride = u32(width);

  deriche_conv_1d_forward(
    start, stride, count,
    uniforms.kde_causal, uniforms.kde_anticausal, uniforms.kde_a
  );
}

@compute @workgroup_size(wg_gaussian_blur, 1)
fn gaussian_blur_stage_2(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = uniforms.density_width;
  let height = uniforms.density_height;
  let y = id.x;
  if (y >= u32(height)) { return; }
  let start = y * u32(width) + id.y * u32(width * height);
  let count = u32(width);
  let stride = u32(1);

  deriche_conv_1d_backward(
    start, stride, count,
    uniforms.kde_causal, uniforms.kde_anticausal, uniforms.kde_a
  );
}

// The gaussian blur is a two-pass Deriche recursive filter (horizontal, then
// vertical) implemented as a ping-pong between two storage buffers.
//
// The forward pass reads the fixed-point accumulated densities. These live in
// `count_buffer` (an array<atomic<u32>>) which physically aliases `blur_buffer`.
// We read them through the u32 view rather than reinterpreting `blur_buffer`'s
// f16 pairs, because bitcasting a vec2<f16> to a u32 is not supported by all
// WGSL implementations (e.g. Firefox / naga).
//
// The two passes are kept as separate functions (rather than one function with
// a `forward` flag) so that each blur entry point statically references only the
// buffers it uses. That keeps `count_buffer`/`blur_buffer` (which alias the same
// GPU buffer) out of a single dispatch's usage scope at the same time, avoiding
// WebGPU storage-buffer aliasing validation errors.

// Forward pass: read u32 counts from count_buffer, write horizontal blur to blur_swap_buffer.
fn deriche_conv_1d_forward(
    start: u32, stride: u32, count: u32,
    kde_causal: vec4<f32>, kde_anticausal: vec4<f32>, kde_a: vec4<f32>
) {
  var s: vec4<f32> = vec4(0.0);
  var y0: f32 = 0.0;
  var y1234: vec4<f32> = vec4(0.0);

  var first_nonzero: u32 = count;
  var last_nonzero: u32 = 0;

  for (var i: u32 = 0; i < count; i++) {
    let offset = start + i * stride;
    var input: f32 = f32(atomicLoad(&count_buffer[offset])) / f32(ACCUMULATE_UNIT);
    if (input != 0.0) {
      first_nonzero = min(i, first_nonzero);
      last_nonzero = max(i, last_nonzero);
    }
    s = vec4(input, s.xyz);
    y1234 = vec4(y0, y1234.xyz);
    y0 = dot(kde_causal, s) - dot(kde_a, y1234);
    blur_swap_buffer[offset] = f16(y0);
  }

  if (first_nonzero > last_nonzero) {
    return;
  }

  s = vec4(0.0);
  y0 = 0.0;
  y1234 = vec4(0.0);

  for (var i: u32 = count - 1 - last_nonzero; i < count; i++) {
    let p = count - 1 - i;
    let offset = start + p * stride;
    var input: f32 = 0.0;
    if (p >= first_nonzero) {
      input = f32(atomicLoad(&count_buffer[offset])) / f32(ACCUMULATE_UNIT);
    }
    y1234 = vec4(y0, y1234.xyz);
    y0 = dot(kde_anticausal, s) - dot(kde_a, y1234);
    s = vec4(input, s.xyz);
    if (y0 != 0.0) {
      blur_swap_buffer[offset] = f16(f32(blur_swap_buffer[offset]) + y0);
    }
  }
}

// Backward pass: read f16 from blur_swap_buffer, write vertical blur to blur_buffer.
fn deriche_conv_1d_backward(
    start: u32, stride: u32, count: u32,
    kde_causal: vec4<f32>, kde_anticausal: vec4<f32>, kde_a: vec4<f32>
) {
  var s: vec4<f32> = vec4(0.0);
  var y0: f32 = 0.0;
  var y1234: vec4<f32> = vec4(0.0);

  var first_nonzero: u32 = count;
  var last_nonzero: u32 = 0;

  for (var i: u32 = 0; i < count; i++) {
    let offset = start + i * stride;
    var input: f32 = f32(blur_swap_buffer[offset]);
    if (input != 0.0) {
      first_nonzero = min(i, first_nonzero);
      last_nonzero = max(i, last_nonzero);
    }
    s = vec4(input, s.xyz);
    y1234 = vec4(y0, y1234.xyz);
    y0 = dot(kde_causal, s) - dot(kde_a, y1234);
    blur_buffer[offset] = f16(y0);
  }

  if (first_nonzero > last_nonzero) {
    return;
  }

  s = vec4(0.0);
  y0 = 0.0;
  y1234 = vec4(0.0);

  for (var i: u32 = count - 1 - last_nonzero; i < count; i++) {
    let p = count - 1 - i;
    let offset = start + p * stride;
    var input: f32 = 0.0;
    if (p >= first_nonzero) {
      input = f32(blur_swap_buffer[offset]);
    }
    y1234 = vec4(y0, y1234.xyz);
    y0 = dot(kde_anticausal, s) - dot(kde_a, y1234);
    s = vec4(input, s.xyz);
    if (y0 != 0.0) {
      blur_buffer[offset] = f16(f32(blur_buffer[offset]) + y0);
    }
  }
}

// =====================================================
// Downsampling: PCG hash for deterministic randomness
// =====================================================

fn pcg_hash(input: u32) -> u32 {
  var state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random_float(seed: u32) -> f32 {
  return f32(pcg_hash(seed)) / 4294967295.0;
}

// =====================================================
// Downsampling Pass 1: Viewport culling + density lookup
// =====================================================
// Uses 2D dispatch for large point counts (>65K workgroups)
// Stride: 256 workgroups * 256 threads = 65536 threads per row

// Pipeline override so host stays in sync with wg_downsample_cull
// (stride == wg_size * workgroups_x). Default keeps the previous layout
// of 256 workgroups × 256 threads = 65536.
override DOWNSAMPLE_STRIDE: u32 = 65536u;

// Workgroup-local accumulators for viewport_cull. Each workgroup tallies
// its visible count + max density in workgroup memory (which is L1 / fast),
// then ONE thread folds the totals into the global ``downsample_counters``
// via TWO global atomics. This collapses the global-atomic hotspot from
// N (visible threads, up to 200M+ at world zoom on the eubucco 322M-row
// dataset) to ceil(N / wg_size) — at wg_size=256 that's ~781K global atomic
// ops instead of 200M.
//
// Why this matters: M-series GPU sustained throughput on a single contended
// atomic address is ~50-100 Mops/s. 200M atomicAdd on counters[0] +
// 200M atomicMax on counters[1] takes 2-3 s EACH and the two cannot fully
// overlap (same warp emits both), so a single viewport_cull dispatch
// exceeds 5 s and trips ``kIOGPUCommandBufferCallbackErrorTimeout`` —
// which then poisons the device with cascading
// ``kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`` rejections.
//
// ``compact_accepted`` already uses this same workgroup-reduction pattern
// (see ``wg_local_count`` below); we mirror it here for the cull pass.
// Variable names must be globally unique across compute entry-points in
// the module — hence the ``wg_cull_*`` prefix.
var<workgroup> wg_cull_visible: atomic<u32>;
var<workgroup> wg_cull_max_density: atomic<u32>;

@compute @workgroup_size(wg_downsample_cull)
fn downsample_viewport_cull(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  // Initialise workgroup accumulators on thread 0 then sync.
  if (lid == 0u) {
    atomicStore(&wg_cull_visible, 0u);
    atomicStore(&wg_cull_max_density, 0u);
  }
  workgroupBarrier();

  let actual_y = id.y + downsample_uniforms.chunk_offset_y;
  let index = actual_y * DOWNSAMPLE_STRIDE + id.x;
  // Use a flag rather than early-return — every thread must reach the
  // workgroupBarrier below, even tail-of-buffer threads that are out of
  // range. (The original `if (index >= count) return;` was safe ONLY when
  // there were no workgroup barriers in this pass.)
  let in_range = index < uniforms.count;

  var in_viewport: bool = false;
  var density: f32 = -1.0;
  var density_fixed: u32 = 0u;

  if (in_range) {
    let point = get_point(index);
    let pos = uniforms.matrix * point.position;

    // Check if point is in viewport [-1, 1]
    in_viewport = pos.x >= -1.0 && pos.x <= 1.0 && pos.y >= -1.0 && pos.y <= 1.0;

    if (in_viewport) {
      // Lookup density at this point's location from blur_buffer
      let width = uniforms.density_width;
      let height = uniforms.density_height;
      let dx = (pos.x + 1.0) / 2.0 * f32(width) - 0.5;
      let dy = (pos.y + 1.0) / 2.0 * f32(height) - 0.5;
      let ix = clamp(i32(dx), 0, width - 1);
      let iy = clamp(i32(dy), 0, height - 1);

      // Sum density across all categories at this grid cell
      var d: f32 = 0.0;
      for (var c: u32 = 0; c < uniforms.category_count; c++) {
        let offset = iy * width + ix + i32(c) * (width * height);
        d += f32(blur_buffer[offset]);
      }
      // Store density (positive = visible). Add small epsilon to ensure > 0.
      density = min(max(d, 0.0001), 65535.0);
      density_fixed = u32(density * 65536.0);
    }
  }

  // Per-visible-point updates land in workgroup-scope atomics (L1, ~1 ns).
  // No L2 ping-pong, no global serialization across millions of threads.
  if (in_viewport) {
    atomicAdd(&wg_cull_visible, 1u);
    atomicMax(&wg_cull_max_density, density_fixed);
  }
  workgroupBarrier();

  // ONE thread per workgroup folds workgroup totals into the global
  // counters. Two global atomics per workgroup × ~781K workgroups at
  // 200M visible = 1.56M global atomics total — finishes in ~20 ms
  // even under heavy contention. Compare 400M global atomics in the
  // original code.
  if (lid == 0u) {
    let total = atomicLoad(&wg_cull_visible);
    if (total > 0u) {
      atomicAdd(&downsample_counters[0], total);
    }
    let max_d = atomicLoad(&wg_cull_max_density);
    if (max_d > 0u) {
      atomicMax(&downsample_counters[1], max_d);
    }
  }

  // Write per-thread output AFTER the workgroup-fold so the barriers
  // above sequence consistently across all threads.
  if (in_range) {
    if (in_viewport) {
      point_data[index] = density;
    } else {
      point_data[index] = -1.0;
    }
  }
}

// =====================================================
// Downsampling Pass 2: Probabilistic acceptance
// =====================================================

@compute @workgroup_size(wg_density_sample)
fn downsample_density_sample(@builtin(global_invocation_id) id: vec3<u32>) {
  let actual_y = id.y + downsample_uniforms.chunk_offset_y;
  let index = actual_y * DOWNSAMPLE_STRIDE + id.x;
  if (index >= uniforms.count) { return; }

  let density = point_data[index];

  // Not visible (density < 0)
  if (density < 0.0) {
    return;
  }

  let visible_count = atomicLoad(&downsample_counters[0]);
  let render_limit = downsample_uniforms.render_limit;

  // If visible count is within limit, accept all visible points (keep positive density)
  if (visible_count <= render_limit) {
    return; // Keep positive value = accepted
  }

  // Compute acceptance probability based on density
  let max_density_fixed = atomicLoad(&downsample_counters[1]);
  let max_density = f32(max_density_fixed) / 65536.0;

  // Base acceptance rate
  let base_rate = f32(render_limit) / f32(visible_count);

  // Density-based modulation: lower density = higher acceptance
  let normalized_density = select(0.0, density / max_density, max_density > 0.0001);
  let density_weight = downsample_uniforms.density_weight;

  // Inverse density weighting: sparse areas get higher probability
  let inverse_weight = 1.0 / (1.0 + normalized_density * density_weight);

  // Compute final probability (scale by ~2 to compensate for average inverse_weight)
  let final_prob = min(1.0, base_rate * inverse_weight * 2.0);

  // Deterministic random for frame stability (based on point index + frame seed)
  let seed = index ^ downsample_uniforms.frame_seed;
  let rand = random_float(seed);

  // If not accepted, set to negative (marks as rejected)
  if (rand >= final_prob) {
    point_data[index] = -1.0;
  }
}

// =====================================================
// Draw points with downsampling
// =====================================================

@vertex
fn points_downsampled_vs(
  @builtin(instance_index) instance: u32,
  @builtin(vertex_index) part: u32,
) -> PointsVertexOutput {
  var out: PointsVertexOutput;

  let point_data = point_data_read[instance];
  if (point_data < 0.0) {
    // To discard a point, we set a out-of-viewport position. This avoids fragment costs.
    out.position = vec4<f32>(-1000, -1000, 0.0, 1.0);
    return out;
  }

  let framebuffer_size = vec2(f32(uniforms.framebuffer_width), f32(uniforms.framebuffer_height));
  let alpha = uniforms.point_alpha * uniforms.points_alpha;
  let dp = vec2<f32>(f32(part % 2), f32(part / 2)) * 2.0 - 1.0;
  let point = get_point(instance);
  let pos = uniforms.matrix * point.position;
  out.position = vec4<f32>(pos.xy + dp * uniforms.point_size / framebuffer_size * 2.0, 0.0, 1.0);
  out.dp = vec3(dp, uniforms.point_size);
  out.color = uniforms.category_colors[point.category] * alpha;
  return out;
}

// =====================================================
// Compaction pass: build a tight list of accepted indices
// =====================================================
// Read point_data and, for each accepted point (>= 0), atomically reserve
// a slot in compact_indices and write the *original* point index there.
// Also bumps indirect_args[1] (instanceCount) so a subsequent drawIndirect
// pass renders exactly accepted_count instances. The render_limit guard
// handles the case where density_sample over-accepts in low-density
// viewports: extra hits are dropped on the floor instead of overflowing
// the buffer.

// Workgroup-level reduction: each workgroup tallies its accepted count
// in shared memory, then a SINGLE thread does ONE global atomicAdd to
// reserve a contiguous range of slots. This collapses the serialization
// hotspot from N atomics on indirect_args[1] (millions when cap is large)
// to N/workgroup_size atomics — typically 100x-300x fewer.
var<workgroup> wg_local_count: atomic<u32>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(wg_compact)
fn compact_accepted(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  if (lid == 0u) {
    atomicStore(&wg_local_count, 0u);
  }
  workgroupBarrier();

  let actual_y = id.y + downsample_uniforms.chunk_offset_y;
  let index = actual_y * DOWNSAMPLE_STRIDE + id.x;
  let in_range = index < uniforms.count;
  let accepted = in_range && (point_data[index] >= 0.0);

  var local_slot: u32 = 0u;
  if (accepted) {
    local_slot = atomicAdd(&wg_local_count, 1u);
  }
  workgroupBarrier();

  if (lid == 0u) {
    let total = atomicLoad(&wg_local_count);
    wg_base = atomicAdd(&indirect_args[1], total);
  }
  workgroupBarrier();

  if (accepted) {
    let global_slot = wg_base + local_slot;
    if (global_slot < downsample_uniforms.render_limit) {
      compact_indices[global_slot] = index;
    }
  }

  // Cap the drawIndirect instance count at render_limit. Without this
  // cap, ``density_sample``'s probabilistic acceptance over-shoots
  // (the legacy ``base_rate * inverse_weight * 2.0`` formula targets
  // ~2× render_limit on average so density weighting can rebalance),
  // and at huge N the drawIndirect issues ~2× the planned instances —
  // the per-instance vertex shader iteration easily exceeds Metal's
  // 5 s wall-clock watchdog. ``compact_indices`` itself is already
  // bounded by the ``global_slot < render_limit`` check above, so the
  // post-cap drawIndirect lands on the populated slice only. Each WG
  // races to the same min; the last write wins and the final value is
  // ``min(actual_accepted, render_limit)``.
  if (lid == 0u) {
    atomicMin(&indirect_args[1], downsample_uniforms.render_limit);
  }
}

// =====================================================
// Draw compacted points (indirect-draw target)
// =====================================================
// Reads from the tight compact_indices list — instance_index ranges from 0
// to accepted_count - 1 instead of 0 to total_count - 1, so the vertex
// shader runs only for accepted points. This is the win against the
// 75M-instance vertex iteration that bottlenecks downsampled draws.

@vertex
fn points_compacted_vs(
  @builtin(instance_index) instance: u32,
  @builtin(vertex_index) part: u32,
) -> PointsVertexOutput {
  let real_index = compact_indices_read[instance];
  let framebuffer_size = vec2(f32(uniforms.framebuffer_width), f32(uniforms.framebuffer_height));
  let alpha = uniforms.point_alpha * uniforms.points_alpha;
  let dp = vec2<f32>(f32(part % 2), f32(part / 2)) * 2.0 - 1.0;
  let point = get_point(real_index);
  let pos = uniforms.matrix * point.position;
  var out: PointsVertexOutput;
  out.position = vec4<f32>(pos.xy + dp * uniforms.point_size / framebuffer_size * 2.0, 0.0, 1.0);
  out.dp = vec3(dp, uniforms.point_size);
  out.color = uniforms.category_colors[point.category] * alpha;
  return out;
}

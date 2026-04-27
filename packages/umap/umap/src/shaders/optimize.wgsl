// UMAP SGD optimization kernel with atomic gradient accumulation.
//
// Two-pass approach:
//   1. accumulate_grads: each thread processes one edge, computing attractive
//      and repulsive gradients, accumulating them via atomicAdd into a
//      fixed-point i32 buffer. This avoids lost updates from GPU write conflicts.
//   2. apply_grads: each thread reads accumulated gradient for one embedding
//      element, converts from fixed-point, applies with learning rate alpha,
//      and clears the accumulator.

struct OptimizeParams {
    n_vertices: u32,
    dim: u32,
    n_edges: u32,
    _pad0: u32,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    grad_clamp: f32,
    apply_offset: u32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read_write> embedding: array<f32>;
@group(0) @binding(1) var<storage, read> edges: array<u32>;
@group(0) @binding(2) var<uniform> params: OptimizeParams;
@group(0) @binding(3) var<storage, read_write> grad_accum: array<atomic<i32>>;

// Fixed-point scale for atomic gradient accumulation.
// Per-edge gradients are clamped to [-4, 4]. Max contributions per vertex
// per epoch ≈ 7*K (K=n_neighbors). Worst case (K=100, all max):
// 700 * 4 * 65536 = 183M, well within i32 ±2.1B.
const FIXED_SCALE: f32 = 65536.0;
// Per-edge gradient clamp, matching the CPU implementation.
const EDGE_CLAMP: f32 = 4.0;

fn to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_SCALE);
}

fn from_fixed(v: i32) -> f32 {
    return f32(v) / FIXED_SCALE;
}

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Pass 1: Accumulate gradients for one edge into grad_accum using atomicAdd.
@compute @workgroup_size(256)
fn accumulate_grads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_edges) {
        return;
    }

    let j = edges[idx * 4u];
    let k = edges[idx * 4u + 1u];
    let n_neg = edges[idx * 4u + 2u];
    let rng_seed = edges[idx * 4u + 3u];
    let j_off = j * params.dim;
    let k_off = k * params.dim;

    // Compute squared distance between j and k
    var dist_sq: f32 = 0.0;
    for (var d = 0u; d < params.dim; d++) {
        let diff = embedding[j_off + d] - embedding[k_off + d];
        dist_sq += diff * diff;
    }

    // Attractive force (positive sample)
    if (dist_sq > 0.0) {
        let pow_b = pow(dist_sq, params.b);
        let grad_coeff = -2.0 * params.a * params.b * (pow_b / dist_sq) / (params.a * pow_b + 1.0);
        for (var d = 0u; d < params.dim; d++) {
            let diff = embedding[j_off + d] - embedding[k_off + d];
            let grad = clamp(grad_coeff * diff, -EDGE_CLAMP, EDGE_CLAMP);
            atomicAdd(&grad_accum[j_off + d], to_fixed(grad));
            atomicAdd(&grad_accum[k_off + d], to_fixed(-grad));
        }
    }

    // Repulsive forces (negative sampling)
    for (var neg_i = 0u; neg_i < n_neg; neg_i++) {
        let neg_k = pcg_hash(rng_seed ^ (neg_i + 1u)) % params.n_vertices;
        if (neg_k == j) {
            continue;
        }
        let neg_off = neg_k * params.dim;

        var neg_dist_sq: f32 = 0.0;
        for (var d = 0u; d < params.dim; d++) {
            let diff = embedding[j_off + d] - embedding[neg_off + d];
            neg_dist_sq += diff * diff;
        }

        if (neg_dist_sq > 0.0) {
            let grad_coeff = 2.0 * params.gamma * params.b / ((0.001 + neg_dist_sq) * (params.a * pow(neg_dist_sq, params.b) + 1.0));
            if (grad_coeff > 0.0) {
                for (var d = 0u; d < params.dim; d++) {
                    let diff = embedding[j_off + d] - embedding[neg_off + d];
                    let grad = clamp(grad_coeff * diff, -EDGE_CLAMP, EDGE_CLAMP);
                    atomicAdd(&grad_accum[j_off + d], to_fixed(grad));
                }
            }
        }
    }
}

// Pass 2: Apply accumulated gradients to embedding and clear accumulator.
//
// The accumulated gradient is the sum of per-edge contributions (each clamped
// to [-4, 4]). We clamp the total to [-grad_clamp, grad_clamp] to prevent
// flyaway outliers. grad_clamp is derived from n_neighbors on the CPU side:
//   grad_clamp = EDGE_CLAMP * sqrt(7 * n_neighbors)
// This models the expected magnitude as a random walk over ~7K gradient
// contributions (K attractive as head, K as tail, 5K repulsive), where
// partially-canceling directions give sqrt(N) scaling.
@compute @workgroup_size(256)
fn apply_grads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + params.apply_offset;
    let total = params.n_vertices * params.dim;
    if (idx >= total) {
        return;
    }

    let g = atomicExchange(&grad_accum[idx], 0);
    if (g != 0) {
        let grad = clamp(from_fixed(g), -params.grad_clamp, params.grad_clamp);
        embedding[idx] += params.alpha * grad;
    }
}

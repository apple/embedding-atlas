// UMAP velocity-SGD optimization kernel with atomic gradient accumulation.
//
// Resident, self-scheduling design: the edge list and per-edge sampling schedule
// counters live in GPU buffers across the whole run. Three entry points:
//
//   1. accumulate_grads: one thread per edge. Each edge reads its own schedule
//      counters, decides if it is due this epoch, advances its own counters (it
//      is the sole writer of slot `i`, so no atomics on counters are needed),
//      then atomicAdds its attractive + repulsive gradients into a fixed-point
//      i32 buffer (avoids lost updates from GPU write conflicts). A sub-batch
//      processes edges in [edge_offset, edge_end).
//   2. apply_grads: one thread per embedding element. Reads + clears the
//      accumulated gradient and applies it. SGD clamps the summed delta;
//      Momentum integrates velocity (which bounds its own step).
//   3. rebase_schedule: subtract a baseline from both counter buffers so the f32
//      counters stay precise over indefinitely long runs.

struct OptimizeParams {
    n_vertices: u32,
    dim: u32,
    n_edges: u32,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    grad_clamp: f32,
    negative_sample_rate: f32,
    base_seed: u32,
    apply_offset: u32,
    edge_offset: u32,
    edge_end: u32,
    optimizer: u32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read_write> embedding: array<f32>;
@group(0) @binding(1) var<storage, read> head: array<u32>;
@group(0) @binding(2) var<storage, read> tail: array<u32>;
@group(0) @binding(3) var<storage, read> eps: array<f32>;
@group(0) @binding(4) var<storage, read_write> eons: array<f32>;
@group(0) @binding(5) var<storage, read_write> eonns: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_accum: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(8) var<uniform> params: OptimizeParams;

// Fixed-point scale for atomic gradient accumulation. Per-edge gradients are
// clamped to [-4, 4]; max contributions per vertex per epoch ≈ 7*K, so worst
// case (K=100): 700 * 4 * 65536 = 183M, well within i32 ±2.1B.
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

// Pass 1: self-schedule, then accumulate this edge's gradients via atomicAdd.
@compute @workgroup_size(256)
fn accumulate_grads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = params.edge_offset + gid.x;
    if (i >= params.edge_end) {
        return;
    }

    let epoch_f = f32(params.epoch);
    if (eons[i] > epoch_f) {
        return; // not due to be sampled this epoch
    }

    // Self-schedule: advance own counters (sole owner of slot i, no atomics).
    var n_neg = 0u;
    if (params.negative_sample_rate > 0.0) {
        let epn = eps[i] / params.negative_sample_rate;
        let raw = (epoch_f - eonns[i]) / epn;
        if (raw > 0.0) {
            n_neg = u32(raw);
        }
        eonns[i] = eonns[i] + f32(n_neg) * epn;
    }
    eons[i] = eons[i] + eps[i];

    let j = head[i];
    let k = tail[i];
    let j_off = j * params.dim;
    let k_off = k * params.dim;

    // Attractive force (positive sample).
    var dist_sq: f32 = 0.0;
    for (var d = 0u; d < params.dim; d++) {
        let diff = embedding[j_off + d] - embedding[k_off + d];
        dist_sq += diff * diff;
    }
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

    // Repulsive forces (negative sampling). Per-edge RNG seed is deterministic
    // from (base_seed, epoch, edge index).
    let seed = params.base_seed * (params.epoch + 1u) + i * 2654435761u;
    for (var neg_i = 0u; neg_i < n_neg; neg_i++) {
        let neg_k = pcg_hash(seed ^ (neg_i + 1u)) % params.n_vertices;
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

// Pass 2: apply accumulated gradients to embedding and clear the accumulator.
//
// `g` is the summed SGD position delta. SGD applies `α·clamp(g, ±grad_clamp)`
// (grad_clamp bounds the random-walk accumulation of ~7K per-edge contributions).
// Momentum integrates `v = β·v + g; x += α·v` — velocity bounds the step, so no
// clamp is applied on that path.
@compute @workgroup_size(256)
fn apply_grads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + params.apply_offset;
    let total = params.n_vertices * params.dim;
    if (idx >= total) {
        return;
    }

    let g = from_fixed(atomicExchange(&grad_accum[idx], 0));
    if (params.optimizer == 1u) {
        let v = params.beta * velocity[idx] + g;
        velocity[idx] = v;
        embedding[idx] += params.alpha * v;
    } else {
        embedding[idx] += params.alpha * clamp(g, -params.grad_clamp, params.grad_clamp);
    }
}

// Pass 3: subtract a baseline from both schedule counters (called when the f32
// epoch coordinate grows large) so precision is retained over long runs.
@compute @workgroup_size(256)
fn rebase_schedule(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = params.edge_offset + gid.x;
    if (i >= params.n_edges) {
        return;
    }
    let base = f32(params.epoch);
    eons[i] = eons[i] - base;
    eonns[i] = eonns[i] - base;
}

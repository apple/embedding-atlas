// Euclidean distance computation with threshold filtering.
// Each invocation computes the distance for one (p, q) pair, and only
// outputs results that beat the current heap threshold for p or q.

struct Params {
    n_features: u32,
    n_pairs: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<storage, read> thresholds: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_pairs) {
        return;
    }

    let p = pairs[idx * 2u];
    let q = pairs[idx * 2u + 1u];
    let p_off = p * params.n_features;
    let q_off = q * params.n_features;

    // Compute squared euclidean distance with loop unrolling
    var sum0 = 0.0;
    var sum1 = 0.0;
    let n4 = params.n_features - (params.n_features % 4u);
    var i = 0u;
    for (; i < n4; i += 4u) {
        let d0 = data[p_off + i] - data[q_off + i];
        let d1 = data[p_off + i + 1u] - data[q_off + i + 1u];
        let d2 = data[p_off + i + 2u] - data[q_off + i + 2u];
        let d3 = data[p_off + i + 3u] - data[q_off + i + 3u];
        sum0 += d0 * d0 + d1 * d1;
        sum1 += d2 * d2 + d3 * d3;
    }
    var sum = sum0 + sum1;
    for (; i < params.n_features; i++) {
        let d = data[p_off + i] - data[q_off + i];
        sum += d * d;
    }
    // Squared euclidean distance (no sqrt) — matches CPU's squared_euclidean
    let dist = sum;

    // Filter: only output if distance beats threshold for p or q
    if (dist < thresholds[p]) {
        let pos = atomicAdd(&counter, 1u);
        output[pos * 3u] = p;
        output[pos * 3u + 1u] = q;
        output[pos * 3u + 2u] = bitcast<u32>(dist);
    }
    if (dist < thresholds[q]) {
        let pos = atomicAdd(&counter, 1u);
        output[pos * 3u] = q;
        output[pos * 3u + 1u] = p;
        output[pos * 3u + 2u] = bitcast<u32>(dist);
    }
}

// Alternative cosine distance matching the CPU's alternative_cosine function:
// log2(sqrt(norm_p * norm_q) / dot)
// This is equivalent to 0.5 * log2(norm_p * norm_q) - log2(dot)

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

    // Compute dot product, norm_p^2, norm_q^2
    var dot_sum = 0.0;
    var norm_p = 0.0;
    var norm_q = 0.0;
    for (var i = 0u; i < params.n_features; i++) {
        let pv = data[p_off + i];
        let qv = data[q_off + i];
        dot_sum += pv * qv;
        norm_p += pv * pv;
        norm_q += qv * qv;
    }

    // Match CPU alternative_cosine: log2(sqrt(norm_p * norm_q) / dot)
    // Handle edge cases same as CPU
    var dist: f32;
    if (norm_p == 0.0 && norm_q == 0.0) {
        dist = 0.0;
    } else if (norm_p == 0.0 || norm_q == 0.0 || dot_sum <= 0.0) {
        // CPU returns FLOAT32_MAX for these cases
        dist = 0x1.fffffep+127;
    } else {
        dist = log2(sqrt(norm_p * norm_q) / dot_sum);
    }

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

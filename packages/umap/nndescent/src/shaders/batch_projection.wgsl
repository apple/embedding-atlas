// Batched dot-product projection: processes multiple hyperplanes in one dispatch.
// Each invocation computes margin for one (point, node) assignment:
//   margins[idx] = dot(hyperplanes[node_id * n_features ..], data[point_idx * n_features ..]) + offsets[node_id]

struct BatchProjectionParams {
    n_features: u32,
    n_total_points: u32,
    base_idx: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> hyperplanes: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<f32>;
@group(0) @binding(3) var<storage, read> point_indices: array<u32>;
@group(0) @binding(4) var<storage, read> node_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> margins: array<f32>;
@group(0) @binding(6) var<uniform> params: BatchProjectionParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + params.base_idx;
    if (idx >= params.n_total_points) {
        return;
    }

    let point_idx = point_indices[idx];
    let node_id = node_ids[idx];
    let hp_off = node_id * params.n_features;
    let pt_off = point_idx * params.n_features;

    // Compute dot product with loop unrolling
    var sum0 = 0.0;
    var sum1 = 0.0;
    let n4 = params.n_features - (params.n_features % 4u);
    var i = 0u;
    for (; i < n4; i += 4u) {
        let d0 = data[pt_off + i] * hyperplanes[hp_off + i];
        let d1 = data[pt_off + i + 1u] * hyperplanes[hp_off + i + 1u];
        let d2 = data[pt_off + i + 2u] * hyperplanes[hp_off + i + 2u];
        let d3 = data[pt_off + i + 3u] * hyperplanes[hp_off + i + 3u];
        sum0 += d0 + d1;
        sum1 += d2 + d3;
    }
    var sum = sum0 + sum1;
    for (; i < params.n_features; i++) {
        sum += data[pt_off + i] * hyperplanes[hp_off + i];
    }

    margins[idx] = sum + offsets[node_id];
}

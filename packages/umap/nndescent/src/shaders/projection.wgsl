// Batch dot-product projection: margin[i] = dot(hyperplane, data[indices[i]]) + offset
// Used by RP tree construction to project points onto a splitting hyperplane.

struct ProjectionParams {
    n_features: u32,
    n_points: u32,
    offset: f32,
    base_idx: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> hyperplane: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> margins: array<f32>;
@group(0) @binding(4) var<uniform> params: ProjectionParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + params.base_idx;
    if (idx >= params.n_points) {
        return;
    }

    let point_idx = indices[idx];
    let point_off = point_idx * params.n_features;

    // Compute dot product with loop unrolling
    var sum0 = 0.0;
    var sum1 = 0.0;
    let n4 = params.n_features - (params.n_features % 4u);
    var i = 0u;
    for (; i < n4; i += 4u) {
        let d0 = data[point_off + i] * hyperplane[i];
        let d1 = data[point_off + i + 1u] * hyperplane[i + 1u];
        let d2 = data[point_off + i + 2u] * hyperplane[i + 2u];
        let d3 = data[point_off + i + 3u] * hyperplane[i + 3u];
        sum0 += d0 + d1;
        sum1 += d2 + d3;
    }
    var sum = sum0 + sum1;
    for (; i < params.n_features; i++) {
        sum += data[point_off + i] * hyperplane[i];
    }

    margins[idx] = sum + params.offset;
}

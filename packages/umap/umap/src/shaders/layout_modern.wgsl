// umap-learn 0.6 Momentum and Adam layouts (see layout_modern.rs, the CPU
// reference). Entry points, per epoch:
//
//   accumulate: one thread per source row of the active block. Advances the
//     schedules of the row's CSR edges and adds attraction plus hashed negative
//     samples to the row's update slot, reading positions frozen for the pass.
//   apply_adam / apply_momentum, decay_momentum: the position updates.
//   project, sort_global, sort_local, build_orders: the negative-selection
//     order (a random-projection ranking or a host-shuffled permutation).
//   adaptation_resultants, adapt_scale: the Adam repulsion-scale adaptation.
//
// Sampling schedules are f32 epochs relative to a base epoch that the host
// moves forward with rebase_schedule, so they stay precise on long runs.

const ALL_BLOCKS: u32 = 0xffffffffu;
const ADAPTATION_CANDIDATES: u32 = 5u;
const SCALE_HEADER: u32 = 8u;
const SORT_TILE: u32 = 512u;
const NEGATIVE_INTERVAL_STRETCH: f32 = 1.5;

struct LayoutParams {
    n_vertices: u32,
    dim: u32,
    epoch: u32,
    active_block: u32,
    n_edges: u32,
    selection_range: u32,
    small: u32,
    exclude_graph_neighbors: u32,
    n_sorted: u32,
    block_size: u32,
    adaptation_sources: u32,
    adaptation_range: u32,
    // Current epoch minus the schedule base epoch.
    schedule_epoch: f32,
    negative_sample_rate: f32,
    a: f32,
    b: f32,
    gamma: f32,
    attraction_factor: f32,
    repulsion_factor: f32,
    alpha: f32,
    momentum: f32,
    beta1: f32,
    beta2: f32,
    one_minus_beta1: f32,
    one_minus_beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    epsilon: f32,
    max_repulsion_norm: f32,
    rebase: f32,
    padding0: u32,
    padding1: u32,
}

struct SortPass {
    block: u32,
    distance: u32,
}

@group(0) @binding(0) var<storage, read_write> embedding: array<f32>;
// CSR column indices of all edges, followed by the n_vertices + 1 row offsets.
@group(0) @binding(1) var<storage, read> graph: array<u32>;
@group(0) @binding(2) var<storage, read> epochs_per_sample: array<f32>;
@group(0) @binding(3) var<storage, read_write> next_sample: array<f32>;
@group(0) @binding(4) var<storage, read_write> next_negative: array<f32>;
// Per index i, four words: to_node_order[i], source_ranks[i], block of node i,
// unused. Scalar words, because different threads write different fields.
@group(0) @binding(5) var<storage, read_write> orders: array<u32>;
// Per coordinate: (update, Adam m, Adam v, unused).
@group(0) @binding(6) var<storage, read_write> state: array<vec4<f32>>;
// [scale, prior, anchor, has anchor, -, -, -, -], the projection direction
// (dim values), per adaptation source (global, local resultant), then scratch.
@group(0) @binding(7) var<storage, read_write> scale: array<f32>;
@group(0) @binding(8) var<uniform> params: LayoutParams;
// Per index i: (host to_node_order[i], node_order[i]).
@group(0) @binding(9) var<storage, read> host_orders: array<vec2<u32>>;
// (orderable projection key, node), padded to a power of two.
@group(0) @binding(10) var<storage, read_write> sort_keys: array<vec2<u32>>;
// Adaptation projection ranks: sources, then local, then global candidates.
@group(0) @binding(11) var<storage, read> adaptation_ranks: array<u32>;
@group(0) @binding(12) var<uniform> sort_pass: SortPass;

fn thread_index(gid: vec3<u32>, groups: vec3<u32>, workgroup_size: u32) -> u32 {
    return gid.x + gid.y * groups.x * workgroup_size;
}

fn row_start(node: u32) -> u32 {
    return graph[params.n_edges + node];
}

fn negative_order(rank: u32) -> u32 {
    return orders[4u * rank];
}

fn source_rank(node: u32) -> u32 {
    return orders[4u * node + 1u];
}

fn node_block(node: u32) -> u32 {
    return orders[4u * node + 2u];
}

fn in_active_block(node: u32) -> bool {
    return params.active_block == ALL_BLOCKS || node_block(node) == params.active_block;
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform hash of (edge, epoch, sample) for negative-sample selection.
fn sample_hash(edge: u32, sample: u32) -> u32 {
    return pcg_hash(edge ^ pcg_hash((params.epoch * 0x9e3779b9u) ^ pcg_hash(sample)));
}

// ---------------------------------------------------------------------------
// Gradients.

fn squared_distance(from_offset: u32, to_offset: u32) -> f32 {
    var sum = 0.0;
    for (var d = 0u; d < params.dim; d++) {
        let diff = embedding[from_offset + d] - embedding[to_offset + d];
        sum += diff * diff;
    }
    return sum;
}

fn attraction_coefficient(dist_squared: f32) -> f32 {
    if (dist_squared <= 0.0) {
        return 0.0;
    }
    let a = params.a;
    let b = params.b;
    return -2.0 * a * b * pow(dist_squared, b - 1.0) / (a * pow(dist_squared, b) + 1.0);
}

// Cephes tanhf; the builtin loses relative precision near 0 on some backends.
fn accurate_tanh(x: f32) -> f32 {
    let magnitude = abs(x);
    if (magnitude >= 0.625) {
        return sign(x) * (1.0 - 2.0 / (exp(min(2.0 * magnitude, 80.0)) + 1.0));
    }
    let s = x * x;
    return ((((-5.70498872745e-3 * s + 2.06390887954e-2) * s - 5.37397155531e-2) * s
        + 1.33314422036e-1) * s - 3.33332819422e-1) * s * x + x;
}

fn repulsion_coefficient(dist_squared: f32, scale_factor: f32) -> f32 {
    if (dist_squared <= 0.0) {
        return 0.0;
    }
    let a = params.a;
    let b = params.b;
    let gamma = params.gamma;
    let grad_coeff = scale_factor * 2.0 * gamma * b
        / ((0.001 + dist_squared) * (a * pow(dist_squared, b) + 1.0));
    if (grad_coeff <= 0.0) {
        return 0.0;
    }
    let grad_norm = sqrt(grad_coeff * grad_coeff * dist_squared);
    let clipped = accurate_tanh(grad_norm / gamma);
    return grad_coeff * min(gamma, params.max_repulsion_norm) * clipped / grad_norm;
}

fn is_graph_neighbor(source: u32, to: u32) -> bool {
    if (to == source) {
        return true;
    }
    var low = row_start(source);
    var high = row_start(source + 1u);
    while (low < high) {
        let middle = low + (high - low) / 2u;
        let value = graph[middle];
        if (value == to) {
            return true;
        }
        if (value < to) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return false;
}

fn negative_interval(edge: u32) -> f32 {
    return epochs_per_sample[edge] / params.negative_sample_rate * NEGATIVE_INTERVAL_STRETCH;
}

// Advances the schedule of `edge`; returns -1 if the edge is not due, else the
// (non-negative) number of negative samples.
fn advance_edge_schedule(edge: u32) -> i32 {
    let epoch = params.schedule_epoch;
    if (next_sample[edge] > epoch) {
        return -1i;
    }
    next_sample[edge] += epochs_per_sample[edge];
    if (params.negative_sample_rate <= 0.0) {
        return 0i;
    }
    let interval = negative_interval(edge);
    let n_negative = i32(clamp(trunc((epoch - next_negative[edge]) / interval), -1e6, 1e6));
    next_negative[edge] += f32(n_negative) * interval;
    return max(n_negative, 0i);
}

fn add_scaled_difference(factor: f32, source: u32, other: u32) {
    if (factor == 0.0) {
        return;
    }
    let dim = params.dim;
    for (var d = 0u; d < dim; d++) {
        let difference = embedding[source * dim + d] - embedding[other * dim + d];
        state[source * dim + d].x += factor * difference;
    }
}

fn repel(source: u32, other: u32, scale_factor: f32) {
    let dim = params.dim;
    let repulsion = repulsion_coefficient(
        squared_distance(source * dim, other * dim),
        scale_factor,
    );
    add_scaled_difference(params.repulsion_factor * repulsion, source, other);
}

fn accumulate_edge(source: u32, edge: u32, n_negative: u32) {
    let dim = params.dim;
    let to = graph[edge];
    let attraction = attraction_coefficient(squared_distance(source * dim, to * dim));
    add_scaled_difference(params.attraction_factor * attraction, source, to);

    let n_vertices = params.n_vertices;
    let exclude = params.exclude_graph_neighbors == 1u;
    if (params.small == 1u) {
        let max_attempts = max(n_negative, 1u) * n_vertices;
        var accepted = 0u;
        var attempts = 0u;
        while (accepted < n_negative && attempts < max_attempts) {
            let other = negative_order(sample_hash(edge, attempts) % n_vertices);
            attempts += 1u;
            if (exclude && is_graph_neighbor(source, other)) {
                continue;
            }
            accepted += 1u;
            repel(source, other, 1.0);
        }
        return;
    }

    let range = params.selection_range;
    let rank = source_rank(source);
    let range_start = u32(clamp(i32(rank) - i32(range / 2u), 0i, i32(n_vertices - range)));
    for (var sample = 0u; sample < n_negative; sample++) {
        let other = negative_order(range_start + sample_hash(edge, sample) % range);
        if (exclude && is_graph_neighbor(source, other)) {
            continue;
        }
        repel(source, other, scale[0]);
    }
}

@compute @workgroup_size(64)
fn accumulate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let source = thread_index(gid, groups, 64u);
    if (source >= params.n_vertices || !in_active_block(source)) {
        return;
    }
    for (var edge = row_start(source); edge < row_start(source + 1u); edge++) {
        let n_negative = advance_edge_schedule(edge);
        if (n_negative >= 0i) {
            accumulate_edge(source, edge, u32(n_negative));
        }
    }
}

@compute @workgroup_size(256)
fn apply_adam(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = thread_index(gid, groups, 256u);
    if (index >= params.n_vertices * params.dim) {
        return;
    }
    var coordinate = state[index];
    let gradient = coordinate.x;
    if (gradient == 0.0) {
        return;
    }
    coordinate.x = 0.0;
    coordinate.y = params.beta1 * coordinate.y + params.one_minus_beta1 * gradient;
    coordinate.z = params.beta2 * coordinate.z + params.one_minus_beta2 * gradient * gradient;
    state[index] = coordinate;
    let m_estimate = coordinate.y / params.bias_correction1;
    let v_estimate = coordinate.z / params.bias_correction2;
    embedding[index] += params.alpha * m_estimate / (sqrt(v_estimate) + params.epsilon);
}

@compute @workgroup_size(256)
fn apply_momentum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = thread_index(gid, groups, 256u);
    if (index >= params.n_vertices * params.dim || !in_active_block(index / params.dim)) {
        return;
    }
    embedding[index] += state[index].x;
}

@compute @workgroup_size(256)
fn decay_momentum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = thread_index(gid, groups, 256u);
    if (index >= params.n_vertices * params.dim) {
        return;
    }
    state[index].x *= params.momentum;
}

// ---------------------------------------------------------------------------
// Sampling schedules.

@compute @workgroup_size(256)
fn restart_schedule(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let edge = thread_index(gid, groups, 256u);
    if (edge >= params.n_edges) {
        return;
    }
    next_sample[edge] = params.schedule_epoch + epochs_per_sample[edge];
    if (params.negative_sample_rate > 0.0) {
        next_negative[edge] = params.schedule_epoch + negative_interval(edge);
    }
}

@compute @workgroup_size(256)
fn reseed_negative_schedule(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let edge = thread_index(gid, groups, 256u);
    if (edge >= params.n_edges || params.negative_sample_rate <= 0.0) {
        return;
    }
    next_negative[edge] = params.schedule_epoch + negative_interval(edge);
}

@compute @workgroup_size(256)
fn rebase_schedule(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let edge = thread_index(gid, groups, 256u);
    if (edge >= params.n_edges) {
        return;
    }
    next_sample[edge] -= params.rebase;
    next_negative[edge] -= params.rebase;
}

// ---------------------------------------------------------------------------
// Negative-selection order.

fn projection_key(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}

fn key_less(x: vec2<u32>, y: vec2<u32>) -> bool {
    return x.x < y.x || (x.x == y.x && x.y < y.y);
}

@compute @workgroup_size(256)
fn project(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = thread_index(gid, groups, 256u);
    if (index >= params.n_sorted) {
        return;
    }
    if (index >= params.n_vertices) {
        sort_keys[index] = vec2<u32>(0xffffffffu, index);
        return;
    }
    var projection = 0.0;
    for (var d = 0u; d < params.dim; d++) {
        projection += embedding[index * params.dim + d] * scale[SCALE_HEADER + d];
    }
    sort_keys[index] = vec2<u32>(projection_key(projection), index);
}

// One bitonic merge step with distance >= SORT_TILE.
@compute @workgroup_size(256)
fn sort_global(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let pair = thread_index(gid, groups, 256u);
    if (pair >= params.n_sorted / 2u) {
        return;
    }
    let step = sort_pass.distance;
    let lower = 2u * step * (pair / step) + pair % step;
    let upper = lower + step;
    let ascending = (lower & sort_pass.block) == 0u;
    let x = sort_keys[lower];
    let y = sort_keys[upper];
    if (key_less(y, x) == ascending) {
        sort_keys[lower] = y;
        sort_keys[upper] = x;
    }
}

var<workgroup> tile: array<vec2<u32>, SORT_TILE>;

// All bitonic steps with distance < SORT_TILE inside one tile. `block` 0 runs
// the full tile sort (blocks 2 .. SORT_TILE); otherwise the steps of `block`.
@compute @workgroup_size(256)
fn sort_local(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let tile_start = (group.x + group.y * groups.x) * SORT_TILE;
    let in_range = tile_start < params.n_sorted;
    let lane = local_id.x;
    if (in_range) {
        tile[lane] = sort_keys[tile_start + lane];
        tile[lane + 256u] = sort_keys[tile_start + lane + 256u];
    }
    workgroupBarrier();
    var first_block = 2u;
    var last_block = SORT_TILE;
    if (sort_pass.block != 0u) {
        first_block = sort_pass.block;
        last_block = sort_pass.block;
    }
    for (var block = first_block; block <= last_block; block *= 2u) {
        for (var step = min(block / 2u, SORT_TILE / 2u); step > 0u; step /= 2u) {
            let lower = 2u * step * (lane / step) + lane % step;
            let upper = lower + step;
            let ascending = ((tile_start + lower) & block) == 0u;
            let x = tile[lower];
            let y = tile[upper];
            if (in_range && key_less(y, x) == ascending) {
                tile[lower] = y;
                tile[upper] = x;
            }
            workgroupBarrier();
        }
    }
    if (in_range) {
        sort_keys[tile_start + lane] = tile[lane];
        sort_keys[tile_start + lane + 256u] = tile[lane + 256u];
    }
}

// Rebuild `orders` from the host permutation (`n_sorted` = 0) or from the
// sorted projection keys, plus the block of every node from the node order.
@compute @workgroup_size(256)
fn build_orders(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = thread_index(gid, groups, 256u);
    if (index >= params.n_vertices) {
        return;
    }
    let host = host_orders[index];
    if (params.n_sorted == 0u) {
        orders[4u * index] = host.x;
        orders[4u * index + 1u] = index;
    } else {
        let node = sort_keys[index].y;
        orders[4u * index] = node;
        orders[4u * node + 1u] = index;
    }
    orders[4u * host.y + 2u] = index / params.block_size;
}

// ---------------------------------------------------------------------------
// Adam repulsion-scale adaptation (layout_modern.rs `negative_force_ratio`).

// Component `d` of the unscaled resultant repulsion of `source` against the
// candidates at `first_rank`.
fn resultant_component(source: u32, first_rank: u32, d: u32) -> f32 {
    let dim = params.dim;
    let gamma = params.gamma;
    var resultant = 0.0;
    for (var c = 0u; c < ADAPTATION_CANDIDATES; c++) {
        let candidate = negative_order(adaptation_ranks[first_rank + c]);
        if (params.exclude_graph_neighbors == 1u && is_graph_neighbor(source, candidate)) {
            continue;
        }
        let dist_squared = squared_distance(source * dim, candidate * dim);
        if (dist_squared <= 0.0) {
            continue;
        }
        let coefficient = 2.0 * gamma * params.b
            / ((0.001 + dist_squared) * (params.a * pow(dist_squared, params.b) + 1.0));
        let gradient_norm = coefficient * sqrt(dist_squared);
        let clip = gamma * accurate_tanh(gradient_norm / gamma) / gradient_norm;
        let delta = embedding[source * dim + d] - embedding[candidate * dim + d];
        resultant += coefficient * delta * clip;
    }
    return resultant;
}

fn repulsion_resultant(source: u32, first_rank: u32) -> f32 {
    var norm = 0.0;
    for (var d = 0u; d < params.dim; d++) {
        let component = resultant_component(source, first_rank, d);
        norm += component * component;
    }
    return sqrt(norm);
}

fn adaptation_offset() -> u32 {
    return SCALE_HEADER + params.dim;
}

@compute @workgroup_size(64)
fn adaptation_resultants(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let n_sources = params.adaptation_sources;
    if (index >= n_sources) {
        return;
    }
    let source = negative_order(adaptation_ranks[index]);
    let local_first = n_sources + index * ADAPTATION_CANDIDATES;
    let global_first = n_sources * (1u + ADAPTATION_CANDIDATES) + index * ADAPTATION_CANDIDATES;
    let offset = adaptation_offset() + 2u * index;
    scale[offset] = repulsion_resultant(source, global_first);
    scale[offset + 1u] = repulsion_resultant(source, local_first);
}

fn sorted_quantile(first: u32, count: u32, q: f32) -> f32 {
    let position = q * f32(count - 1u);
    let lower = u32(floor(position));
    let upper = min(lower + 1u, count - 1u);
    let fraction = position - f32(lower);
    return scale[first + lower] + (scale[first + upper] - scale[first + lower]) * fraction;
}

@compute @workgroup_size(1)
fn adapt_scale() {
    let n_sources = params.adaptation_sources;
    let resultants = adaptation_offset();
    let first = resultants + 2u * n_sources;
    var count = 0u;
    for (var i = 0u; i < n_sources; i++) {
        let global_resultant = scale[resultants + 2u * i];
        let local_resultant = scale[resultants + 2u * i + 1u];
        if (global_resultant > 0.0 && local_resultant > 0.0) {
            let ratio = log(global_resultant) - log(local_resultant);
            var position = first + count;
            while (position > first && scale[position - 1u] > ratio) {
                scale[position] = scale[position - 1u];
                position -= 1u;
            }
            scale[position] = ratio;
            count += 1u;
        }
    }
    var force_ratio = 1.0;
    if (count > 0u) {
        let lower = sorted_quantile(first, count, 0.1);
        let upper = sorted_quantile(first, count, 0.9);
        var sum = 0.0;
        for (var i = 0u; i < count; i++) {
            sum += clamp(scale[first + i], lower, upper);
        }
        force_ratio = exp(sum / f32(count));
    }
    let value = scale[0];
    let prior = scale[1];
    if (scale[3] == 0.0) {
        scale[2] = value / force_ratio;
        scale[3] = 1.0;
        return;
    }
    let raw_scale = clamp(scale[2] * force_ratio, 0.5 * prior, 2.0 * prior);
    let log_change = clamp(log(raw_scale / value), log(0.9), log(1.1));
    scale[0] = value * exp(0.1 * log_change);
}

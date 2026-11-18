//! WebGPU port of the umap-learn 0.6 Momentum and Adam layouts
//! ([`crate::layout_modern`]).
//!
//! Positions, optimizer state, per-edge sampling schedules, the
//! negative-selection order, and the repulsion scale live on the GPU. The host
//! owns the RNG-driven parts through the same [`NegativeOrdering`] code as the
//! CPU layout: source-order shuffles, the shuffled negative order when the
//! selection window covers all vertices, projection directions, and the
//! adaptation sample ranks. The GPU computes projections, sorts them (bitonic),
//! and adapts the repulsion scale, so an epoch needs no readback.
//!
//! The GPU follows the same algorithm, schedules, and parameters, but it is not
//! bit-identical to the CPU: sampling schedules are f32, negative samples come
//! from a hash of (edge, epoch, sample) inside the same window, and projections
//! are f32.
//!
//! Per-edge data is 16 bytes in four separate bindings (CSR column, sampling
//! interval, next sample epoch, next negative-sample epoch), so the largest
//! binding is 4 bytes per edge, as for the SGD GPU path.

use crate::gpu_optimize::{map_buffer_read, storage_entry};
use crate::layout_modern::{
    adam_bias_corrections, effective_selection_range, epoch_rates, AdaptationRanks,
    ModernLayoutParams, ModernOptimizerKind, ModernOptions, NegativeOrdering, OrderUpdate,
    RateMode, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, MAX_REPULSION_NORM,
    SCALE_ADAPTATION_NEGATIVES_PER_SOURCE, SMALL_LAYOUT_ATTRACTION_SCALE,
    SMALL_LAYOUT_MAX_VERTICES,
};
use crate::optimize::LiveParams;

const ROW_WORKGROUP_SIZE: u32 = 64;
const ELEMENT_WORKGROUP_SIZE: u32 = 256;
const SORT_TILE: usize = 512;
const ALL_BLOCKS: u32 = u32::MAX;
const SCALE_HEADER: usize = 8;
/// Dynamic-offset alignment of uniform slots.
const UNIFORM_SLOT: u64 = 256;
const UNIFORM_SLOTS: u64 = 16;
/// Epochs between moves of the f32 schedule base epoch.
const SCHEDULE_REBASE_INTERVAL: usize = 1 << 16;

/// Uniform block `LayoutParams` of `shaders/layout_modern.wgsl` (128 bytes).
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct LayoutUniforms {
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
    padding: [u32; 2],
}

/// Uniform block `SortPass` of the shader (`block`, `distance`), padded to a
/// dynamic-offset slot.
type SortPass = [u32; 64];

/// Largest storage binding (bytes) of a layout with these sizes.
fn largest_binding(n_vertices: usize, n_edges: usize, dim: usize, n_adaptation: usize) -> u64 {
    let n_sorted = n_vertices.next_power_of_two().max(SORT_TILE);
    [
        (n_edges + n_vertices + 1) * 4,
        n_vertices * dim * 16,
        n_vertices * 16,
        n_sorted * 8,
        (SCALE_HEADER + dim + 3 * n_adaptation) * 4,
    ]
    .into_iter()
    .max()
    .unwrap() as u64
}

/// Bitonic sort passes over `n_sorted` keys: `(block, distance)` with
/// `block == 0` for the full tile sort and `distance == 0` for a tile pass.
fn sort_passes(n_sorted: usize) -> Vec<(u32, u32)> {
    let mut passes = vec![(0, 0)];
    let mut block = 2 * SORT_TILE;
    while block <= n_sorted {
        let mut distance = block / 2;
        while distance >= SORT_TILE {
            passes.push((block as u32, distance as u32));
            distance /= 2;
        }
        passes.push((block as u32, 0));
        block *= 2;
    }
    passes
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroup_size: u32,
}

struct Pipelines {
    accumulate: Pipeline,
    apply_adam: Pipeline,
    apply_momentum: Pipeline,
    decay_momentum: Pipeline,
    restart_schedule: Pipeline,
    reseed_negative_schedule: Pipeline,
    rebase_schedule: Pipeline,
    project: Pipeline,
    sort_global: Pipeline,
    sort_local: Pipeline,
    build_orders: Pipeline,
    adaptation_resultants: Pipeline,
    adapt_scale: Pipeline,
}

enum Command {
    /// Run a pipeline over a number of threads.
    Dispatch(fn(&Pipelines) -> &Pipeline, usize),
    /// Sort the projection keys.
    Sort,
}

/// Commands of one submission, each with its uniform slot.
#[derive(Default)]
struct Batch {
    uniforms: Vec<LayoutUniforms>,
    commands: Vec<Command>,
}

impl Batch {
    /// Queue `pipeline` over `n_threads` threads with `uniforms`.
    fn push(
        &mut self,
        pipeline: fn(&Pipelines) -> &Pipeline,
        n_threads: usize,
        uniforms: LayoutUniforms,
    ) {
        self.push_command(Command::Dispatch(pipeline, n_threads), uniforms);
    }

    fn push_command(&mut self, command: Command, uniforms: LayoutUniforms) {
        self.uniforms.push(uniforms);
        self.commands.push(command);
    }
}

/// Resident GPU state of a Momentum or Adam layout; mirrors
/// [`crate::layout_modern::ModernLayout`].
pub struct GpuModernLayout {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    embedding_buffer: wgpu::Buffer,
    #[cfg(test)]
    next_sample_buffer: wgpu::Buffer,
    #[cfg(test)]
    next_negative_buffer: wgpu::Buffer,
    #[cfg(test)]
    orders_buffer: wgpu::Buffer,
    state_buffer: wgpu::Buffer,
    scale_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    host_orders_buffer: wgpu::Buffer,
    adaptation_ranks_buffer: wgpu::Buffer,
    embedding_staging: wgpu::Buffer,
    sort_passes: Vec<(u32, u32)>,
    n_sorted: usize,
    max_workgroups: u32,
    params: ModernLayoutParams,
    n_vertices: usize,
    dim: usize,
    n_edges: usize,
    /// Largest number of adaptation sources the buffers hold.
    adaptation_capacity: usize,
    /// Epoch that the f32 schedule epochs count from.
    schedule_base: usize,
    ordering: NegativeOrdering,
    /// Host copy of the embedding; current when `embedding_current` is set.
    embedding: Vec<f32>,
    embedding_current: bool,
    /// Adam steps since the moments were last reset.
    adam_steps: usize,
    epoch: usize,
    /// Error scopes pushed at creation, popped after the first epoch.
    error_scopes_open: bool,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

fn create_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16).next_multiple_of(16),
        usage,
        mapped_at_creation: false,
    })
}

fn dynamic_uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

impl GpuModernLayout {
    /// Upload a layout of the sorted CSR graph (`indptr`, `indices`, per-edge
    /// `epochs_per_sample`) starting at `start_epoch`. Returns `None` if the
    /// buffers exceed the device limits.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        epochs_per_sample: Vec<f64>,
        embedding: &[f32],
        dim: usize,
        params: ModernLayoutParams,
        start_epoch: usize,
    ) -> Option<Self> {
        let n_vertices = indptr.len().saturating_sub(1);
        let n_edges = indices.len();
        assert_eq!(embedding.len(), n_vertices * dim);
        assert_eq!(epochs_per_sample.len(), n_edges);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let device_lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let lost_flag = device_lost.clone();
        device.set_device_lost_callback(move |_, _| {
            lost_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        let n_sorted = n_vertices.next_power_of_two().max(SORT_TILE);
        let n_adaptation = params
            .scale_adaptation_samples
            .max(ModernOptions::default().scale_adaptation_samples)
            .min(n_vertices);
        let limits = device.limits();
        let buffer_limit =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if dim == 0
            || n_vertices == 0
            || u32::try_from(n_edges + n_vertices + 1).is_err()
            || u32::try_from(n_sorted * dim).is_err()
            || largest_binding(n_vertices, n_edges, dim, n_adaptation) > buffer_limit
        {
            drop(device.pop_error_scope());
            drop(device.pop_error_scope());
            return None;
        }
        let element_bytes = (n_vertices * dim) as u64 * 4;
        let edge_bytes = n_edges as u64 * 4;

        use wgpu::BufferUsages as Usage;
        let storage = Usage::STORAGE | Usage::COPY_DST;
        let buffer =
            |label: &str, size: u64, usage: Usage| create_buffer(&device, label, size, usage);
        let embedding_buffer = buffer("modern_embedding", element_bytes, storage | Usage::COPY_SRC);
        let graph_u32: Vec<u32> = indices.iter().chain(&indptr).map(|&i| i as u32).collect();
        let graph_buffer = buffer("modern_graph", graph_u32.len() as u64 * 4, storage);
        queue.write_buffer(&graph_buffer, 0, bytemuck::cast_slice(&graph_u32));
        drop(graph_u32);
        let epochs_buffer = buffer("modern_epochs_per_sample", edge_bytes, storage);
        let epochs_f32: Vec<f32> = epochs_per_sample.iter().map(|&e| e as f32).collect();
        queue.write_buffer(&epochs_buffer, 0, bytemuck::cast_slice(&epochs_f32));
        drop(epochs_f32);
        let next_sample_buffer =
            buffer("modern_next_sample", edge_bytes, storage | Usage::COPY_SRC);
        let next_negative_buffer = buffer(
            "modern_next_negative",
            edge_bytes,
            storage | Usage::COPY_SRC,
        );
        let orders_buffer = buffer(
            "modern_orders",
            n_vertices as u64 * 16,
            storage | Usage::COPY_SRC,
        );
        let state_buffer = buffer("modern_state", 4 * element_bytes, storage);
        let scale_buffer = buffer(
            "modern_scale",
            (SCALE_HEADER + dim + 3 * n_adaptation) as u64 * 4,
            storage | Usage::COPY_SRC,
        );
        let uniforms_buffer = buffer(
            "modern_uniforms",
            UNIFORM_SLOT * UNIFORM_SLOTS,
            Usage::UNIFORM | Usage::COPY_DST,
        );
        let host_orders_buffer = buffer("modern_host_orders", n_vertices as u64 * 8, storage);
        let sort_keys_buffer = buffer("modern_sort_keys", n_sorted as u64 * 8, storage);
        let adaptation_ranks_buffer = buffer(
            "modern_adaptation_ranks",
            (n_adaptation * (1 + 2 * SCALE_ADAPTATION_NEGATIVES_PER_SOURCE)) as u64 * 4,
            storage,
        );
        let sort_passes = sort_passes(n_sorted);
        let sort_uniforms: Vec<SortPass> = sort_passes
            .iter()
            .map(|&(block, distance)| {
                let mut pass = [0; 64];
                pass[0] = block;
                pass[1] = distance;
                pass
            })
            .collect();
        let sort_buffer = buffer(
            "modern_sort_passes",
            (sort_uniforms.len() * std::mem::size_of::<SortPass>()) as u64,
            Usage::UNIFORM | Usage::COPY_DST,
        );
        queue.write_buffer(&sort_buffer, 0, bytemuck::cast_slice(&sort_uniforms));
        let embedding_staging = buffer(
            "modern_embedding_staging",
            element_bytes,
            Usage::MAP_READ | Usage::COPY_DST,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layout_modern_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layout_modern.wgsl").into()),
        });
        let layout_uniform_size = std::mem::size_of::<LayoutUniforms>() as u64;
        let resources: [(u32, &wgpu::Buffer, wgpu::BindGroupLayoutEntry, u64); 13] = [
            (0, &embedding_buffer, storage_entry(0, false), 0),
            (1, &graph_buffer, storage_entry(1, true), 0),
            (2, &epochs_buffer, storage_entry(2, true), 0),
            (3, &next_sample_buffer, storage_entry(3, false), 0),
            (4, &next_negative_buffer, storage_entry(4, false), 0),
            (5, &orders_buffer, storage_entry(5, false), 0),
            (6, &state_buffer, storage_entry(6, false), 0),
            (7, &scale_buffer, storage_entry(7, false), 0),
            (
                8,
                &uniforms_buffer,
                dynamic_uniform_entry(8, layout_uniform_size),
                layout_uniform_size,
            ),
            (9, &host_orders_buffer, storage_entry(9, true), 0),
            (10, &sort_keys_buffer, storage_entry(10, false), 0),
            (11, &adaptation_ranks_buffer, storage_entry(11, true), 0),
            (
                12,
                &sort_buffer,
                dynamic_uniform_entry(12, std::mem::size_of::<SortPass>() as u64),
                std::mem::size_of::<SortPass>() as u64,
            ),
        ];
        let pipeline = |entry: &str, bindings: &[u32], workgroup_size: u32| {
            let used: Vec<_> = resources
                .iter()
                .filter(|(binding, ..)| bindings.contains(binding))
                .collect();
            let entries: Vec<wgpu::BindGroupLayoutEntry> =
                used.iter().map(|(_, _, entry, _)| *entry).collect();
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(entry),
                    entries: &entries,
                });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &bind_group_layout,
                entries: &used
                    .iter()
                    .map(|(binding, buffer, _, size)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(*size),
                        }),
                    })
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            Pipeline {
                pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
                bind_group,
                workgroup_size,
            }
        };
        let pipelines = Pipelines {
            accumulate: pipeline(
                "accumulate",
                &[0, 1, 2, 3, 4, 5, 6, 7, 8],
                ROW_WORKGROUP_SIZE,
            ),
            apply_adam: pipeline("apply_adam", &[0, 6, 8], ELEMENT_WORKGROUP_SIZE),
            apply_momentum: pipeline("apply_momentum", &[0, 5, 6, 8], ELEMENT_WORKGROUP_SIZE),
            decay_momentum: pipeline("decay_momentum", &[6, 8], ELEMENT_WORKGROUP_SIZE),
            restart_schedule: pipeline("restart_schedule", &[2, 3, 4, 8], ELEMENT_WORKGROUP_SIZE),
            reseed_negative_schedule: pipeline(
                "reseed_negative_schedule",
                &[2, 4, 8],
                ELEMENT_WORKGROUP_SIZE,
            ),
            rebase_schedule: pipeline("rebase_schedule", &[3, 4, 8], ELEMENT_WORKGROUP_SIZE),
            project: pipeline("project", &[0, 7, 8, 10], ELEMENT_WORKGROUP_SIZE),
            sort_global: pipeline("sort_global", &[8, 10, 12], ELEMENT_WORKGROUP_SIZE),
            sort_local: pipeline("sort_local", &[8, 10, 12], ELEMENT_WORKGROUP_SIZE),
            build_orders: pipeline("build_orders", &[5, 8, 9, 10], ELEMENT_WORKGROUP_SIZE),
            adaptation_resultants: pipeline(
                "adaptation_resultants",
                &[0, 1, 2, 5, 7, 8, 11],
                ROW_WORKGROUP_SIZE,
            ),
            adapt_scale: pipeline("adapt_scale", &[7, 8], 1),
        };

        let mut layout = GpuModernLayout {
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            device,
            queue,
            pipelines,
            embedding_buffer,
            #[cfg(test)]
            next_sample_buffer: next_sample_buffer.clone(),
            #[cfg(test)]
            next_negative_buffer: next_negative_buffer.clone(),
            #[cfg(test)]
            orders_buffer: orders_buffer.clone(),
            state_buffer,
            scale_buffer,
            uniforms_buffer,
            host_orders_buffer,
            adaptation_ranks_buffer,
            embedding_staging,
            sort_passes,
            n_sorted,
            ordering: NegativeOrdering::new(&params, n_vertices),
            params,
            n_vertices,
            dim,
            n_edges,
            adaptation_capacity: n_adaptation,
            schedule_base: start_epoch,
            embedding: Vec::new(),
            embedding_current: false,
            adam_steps: 0,
            epoch: start_epoch,
            error_scopes_open: true,
            device_lost,
        };
        layout.upload_embedding(embedding);
        layout.restart(start_epoch);
        Some(layout)
    }

    /// Replace the embedding on the GPU and in the host copy.
    pub fn upload_embedding(&mut self, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.n_vertices * self.dim);
        self.queue
            .write_buffer(&self.embedding_buffer, 0, bytemuck::cast_slice(embedding));
        self.embedding.clear();
        self.embedding.extend_from_slice(embedding);
        self.embedding_current = true;
    }

    /// Reset optimizer state, sampling schedules, orderings, and RNGs, and
    /// continue the schedule from `start_epoch` (see
    /// [`crate::layout_modern::ModernLayout::restart`]).
    pub fn restart(&mut self, start_epoch: usize) {
        let state_bytes = self.n_vertices * self.dim * 16;
        self.queue
            .write_buffer(&self.state_buffer, 0, &vec![0u8; state_bytes]);
        self.adam_steps = 0;
        self.epoch = start_epoch;
        self.schedule_base = start_epoch;
        self.reset_ordering();
        let mut batch = Batch::default();
        batch.push(|p| &p.restart_schedule, self.n_edges, self.base_uniforms());
        self.submit(batch);
    }

    fn reset_ordering(&mut self) {
        self.ordering = NegativeOrdering::new(&self.params, self.n_vertices);
        let mut scale = [0.0f32; SCALE_HEADER];
        scale[0] = self.ordering.negative_scale(&self.params, self.n_vertices);
        scale[1] = self.ordering.scale_prior();
        self.queue
            .write_buffer(&self.scale_buffer, 0, bytemuck::cast_slice(&scale));
    }

    /// Workgroup grid covering `n_threads`, wrapped into the y dimension.
    fn workgroups(&self, n_threads: usize, workgroup_size: u32) -> (u32, u32) {
        let groups = n_threads.div_ceil(workgroup_size as usize).max(1) as u32;
        let columns = groups.min(self.max_workgroups);
        (columns, groups.div_ceil(columns))
    }

    fn submit(&self, batch: Batch) {
        if batch.commands.is_empty() {
            return;
        }
        assert!(batch.uniforms.len() as u64 <= UNIFORM_SLOTS);
        for (slot, uniforms) in batch.uniforms.iter().enumerate() {
            self.queue.write_buffer(
                &self.uniforms_buffer,
                slot as u64 * UNIFORM_SLOT,
                bytemuck::bytes_of(uniforms),
            );
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (slot, command) in batch.commands.iter().enumerate() {
                let offset = slot as u32 * UNIFORM_SLOT as u32;
                match *command {
                    Command::Sort => self.encode_sort(&mut pass, offset),
                    Command::Dispatch(_, 0) => {}
                    Command::Dispatch(select, n_threads) => {
                        let pipeline = select(&self.pipelines);
                        pass.set_pipeline(&pipeline.pipeline);
                        pass.set_bind_group(0, &pipeline.bind_group, &[offset]);
                        let (x, y) = self.workgroups(n_threads, pipeline.workgroup_size);
                        pass.dispatch_workgroups(x, y, 1);
                    }
                }
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Encode the full bitonic sort of the projection keys.
    fn encode_sort(&self, pass: &mut wgpu::ComputePass, uniforms_offset: u32) {
        let sort_size = std::mem::size_of::<SortPass>() as u32;
        for (index, &(_, distance)) in self.sort_passes.iter().enumerate() {
            let offsets = [uniforms_offset, index as u32 * sort_size];
            let pipeline = if distance == 0 {
                &self.pipelines.sort_local
            } else {
                &self.pipelines.sort_global
            };
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &pipeline.bind_group, &offsets);
            let (x, y) = self.workgroups(self.n_sorted / 2, ELEMENT_WORKGROUP_SIZE);
            pass.dispatch_workgroups(x, y, 1);
        }
    }

    pub fn params(&self) -> &ModernLayoutParams {
        &self.params
    }

    /// The number of epochs run so far (including the start offset).
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// See [`crate::layout_modern::ModernLayout::set_live_params`].
    pub fn set_live_params(&mut self, live: LiveParams) {
        self.params.initial_alpha = live.alpha;
        self.params.gamma = live.gamma;
        self.params.a = live.a;
        self.params.b = live.b;
        if live.negative_sample_rate != self.params.negative_sample_rate {
            self.params.negative_sample_rate = live.negative_sample_rate;
            let mut batch = Batch::default();
            batch.push(
                |p| &p.reseed_negative_schedule,
                self.n_edges,
                self.base_uniforms(),
            );
            self.submit(batch);
        }
    }

    pub fn set_n_epochs(&mut self, n_epochs: usize) {
        self.params.n_epochs = n_epochs;
    }

    pub fn set_good_initialization(&mut self, good_initialization: bool) {
        self.params.good_initialization = good_initialization;
    }

    /// See [`crate::layout_modern::ModernLayout::set_options`]. The adaptation
    /// sample count is limited to the larger of 128 and the count at creation.
    pub fn set_options(&mut self, options: ModernOptions) {
        self.params.set_options(options);
        self.params.scale_adaptation_samples = options
            .scale_adaptation_samples
            .min(self.adaptation_capacity);
        self.reset_ordering();
    }

    fn small(&self) -> bool {
        self.n_vertices <= SMALL_LAYOUT_MAX_VERTICES
    }

    fn block_size(&self) -> usize {
        match self.params.kind {
            ModernOptimizerKind::Momentum if !self.small() => (self.n_vertices / 2).max(1),
            _ => self.n_vertices,
        }
    }

    fn base_uniforms(&self) -> LayoutUniforms {
        LayoutUniforms {
            n_vertices: self.n_vertices as u32,
            dim: self.dim as u32,
            epoch: self.epoch as u32,
            active_block: ALL_BLOCKS,
            n_edges: self.n_edges as u32,
            schedule_epoch: (self.epoch - self.schedule_base) as f32,
            negative_sample_rate: self.params.negative_sample_rate,
            selection_range: effective_selection_range(&self.params, self.n_vertices) as u32,
            small: self.small() as u32,
            exclude_graph_neighbors: self.params.exclude_graph_neighbors as u32,
            block_size: self.block_size() as u32,
            a: self.params.a,
            b: self.params.b,
            gamma: self.params.gamma,
            max_repulsion_norm: MAX_REPULSION_NORM,
            adaptation_range: self.params.negative_selection_range.max(1) as u32,
            ..LayoutUniforms::default()
        }
    }

    /// Upload the host orders and queue the rebuild of the GPU orders.
    fn queue_order_update(&self, update: OrderUpdate, batch: &mut Batch) {
        let host_orders: Vec<[u32; 2]> = self
            .ordering
            .to_node_order
            .iter()
            .zip(self.ordering.node_order())
            .map(|(&negative, &source)| [negative as u32, source as u32])
            .collect();
        self.queue.write_buffer(
            &self.host_orders_buffer,
            0,
            bytemuck::cast_slice(&host_orders),
        );
        let mut uniforms = self.base_uniforms();
        if let OrderUpdate::Project(direction) = update {
            let direction: Vec<f32> = direction.iter().map(|&d| d as f32).collect();
            self.queue.write_buffer(
                &self.scale_buffer,
                SCALE_HEADER as u64 * 4,
                bytemuck::cast_slice(&direction),
            );
            uniforms.n_sorted = self.n_sorted as u32;
            batch.push(|p| &p.project, self.n_sorted, uniforms);
            batch.push_command(Command::Sort, uniforms);
        }
        batch.push(|p| &p.build_orders, self.n_vertices, uniforms);
    }

    fn queue_adaptation(&self, ranks: AdaptationRanks, batch: &mut Batch) {
        let ranks_u32: Vec<u32> = ranks
            .sources
            .iter()
            .chain(&ranks.local)
            .chain(&ranks.global)
            .map(|&rank| rank as u32)
            .collect();
        self.queue.write_buffer(
            &self.adaptation_ranks_buffer,
            0,
            bytemuck::cast_slice(&ranks_u32),
        );
        let uniforms = LayoutUniforms {
            adaptation_sources: ranks.sources.len() as u32,
            ..self.base_uniforms()
        };
        batch.push(|p| &p.adaptation_resultants, ranks.sources.len(), uniforms);
        batch.push(|p| &p.adapt_scale, 1, uniforms);
    }

    /// See [`crate::layout_modern::ModernLayout::reset_moments`].
    pub fn reset_moments(&mut self) {
        if self.params.kind == ModernOptimizerKind::Adam {
            let state_bytes = self.n_vertices * self.dim * 16;
            self.queue
                .write_buffer(&self.state_buffer, 0, &vec![0u8; state_bytes]);
            self.adam_steps = 0;
        }
    }

    /// See [`crate::layout_modern::ModernLayout::scale_velocity`].
    pub fn scale_velocity(&mut self, factor: f32) {
        if self.params.kind == ModernOptimizerKind::Momentum {
            let uniforms = LayoutUniforms {
                momentum: factor,
                ..self.base_uniforms()
            };
            let mut batch = Batch::default();
            batch.push(|p| &p.decay_momentum, self.n_vertices * self.dim, uniforms);
            self.submit(batch);
        }
    }

    /// Run one epoch on the 0.6 schedule (see
    /// [`crate::layout_modern::ModernLayout::step`]).
    pub async fn step(&mut self) -> Result<(), String> {
        self.step_with(RateMode::Schedule).await
    }

    /// Run one epoch with rates from `mode`. Fails if the device is lost, or
    /// if creating the layout or running its first epoch raised a GPU error;
    /// the embedding is then not advanced and the caller should fall back to
    /// the CPU.
    pub async fn step_with(&mut self, mode: RateMode) -> Result<(), String> {
        if self.device_lost.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("the GPU device was lost".to_string());
        }
        self.run_epoch(mode);
        if self.error_scopes_open {
            self.error_scopes_open = false;
            let validation = self.device.pop_error_scope().await;
            let out_of_memory = self.device.pop_error_scope().await;
            if let Some(error) = validation.or(out_of_memory) {
                return Err(error.to_string());
            }
        }
        Ok(())
    }

    fn run_epoch(&mut self, mode: RateMode) {
        let mut batch = Batch::default();
        if let Some(update) = self.ordering.prepare(self.dim, &self.params) {
            self.queue_order_update(update, &mut batch);
        }
        let kind = self.params.kind;
        let small = self.small();
        if kind == ModernOptimizerKind::Adam && !small {
            if let Some(ranks) = self.ordering.adaptation_ranks(&self.params, self.epoch) {
                self.queue_adaptation(ranks, &mut batch);
            }
        }
        let (alpha, momentum) = epoch_rates(
            mode,
            kind,
            self.epoch,
            self.params.n_epochs,
            self.params.initial_alpha,
            self.params.good_initialization,
        );
        let small_scale = if small {
            SMALL_LAYOUT_ATTRACTION_SCALE
        } else {
            1.0
        };
        let base = self.base_uniforms();
        let elements = self.n_vertices * self.dim;
        match kind {
            ModernOptimizerKind::Adam => {
                let (bias_correction1, bias_correction2) = adam_bias_corrections(self.adam_steps);
                self.adam_steps += 1;
                let uniforms = LayoutUniforms {
                    attraction_factor: small_scale,
                    repulsion_factor: 1.0,
                    alpha,
                    beta1: ADAM_BETA1,
                    beta2: ADAM_BETA2,
                    one_minus_beta1: 1.0 - ADAM_BETA1,
                    one_minus_beta2: 1.0 - ADAM_BETA2,
                    bias_correction1,
                    bias_correction2,
                    epsilon: ADAM_EPSILON,
                    ..base
                };
                batch.push(|p| &p.accumulate, self.n_vertices, uniforms);
                batch.push(|p| &p.apply_adam, elements, uniforms);
            }
            ModernOptimizerKind::Momentum => {
                let uniforms = LayoutUniforms {
                    attraction_factor: small_scale * alpha,
                    repulsion_factor: alpha,
                    momentum,
                    ..base
                };
                let n_blocks = if small {
                    1
                } else {
                    self.n_vertices.div_ceil(self.block_size())
                };
                for block in 0..n_blocks {
                    let active_block = if small { ALL_BLOCKS } else { block as u32 };
                    let block_uniforms = LayoutUniforms {
                        active_block,
                        ..uniforms
                    };
                    batch.push(|p| &p.accumulate, self.n_vertices, block_uniforms);
                    batch.push(|p| &p.apply_momentum, elements, block_uniforms);
                }
                batch.push(|p| &p.decay_momentum, elements, uniforms);
            }
        }
        self.submit(batch);
        self.embedding_current = false;

        let update = self.ordering.advance(self.dim, &self.params);
        let mut batch = Batch::default();
        self.queue_order_update(update, &mut batch);
        self.submit(batch);
        self.epoch += 1;
        if self.epoch - self.schedule_base >= SCHEDULE_REBASE_INTERVAL {
            let rebase = LayoutUniforms {
                rebase: (self.epoch - self.schedule_base) as f32,
                ..self.base_uniforms()
            };
            let mut batch = Batch::default();
            batch.push(|p| &p.rebase_schedule, self.n_edges, rebase);
            self.submit(batch);
            self.schedule_base = self.epoch;
        }
    }

    /// Run epochs until the schedule horizon is reached.
    pub async fn run(&mut self) -> Result<(), String> {
        while self.epoch < self.params.n_epochs {
            self.step().await?;
        }
        Ok(())
    }

    /// Download the embedding into the host copy if the GPU has moved it.
    pub async fn sync_embedding(&mut self) {
        if self.embedding_current {
            return;
        }
        let size = (self.embedding.len() * 4) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.embedding_buffer, 0, &self.embedding_staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));
        map_buffer_read(&self.device, &self.embedding_staging, 0..size).await;
        {
            let data = self.embedding_staging.slice(..size).get_mapped_range();
            self.embedding.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.embedding_staging.unmap();
        self.embedding_current = true;
    }

    /// The embedding after [`sync_embedding`](Self::sync_embedding).
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    #[cfg(test)]
    pub(crate) fn simulate_device_loss(&self) {
        self.device_lost
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    async fn download<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
        let size = (len * std::mem::size_of::<T>()) as u64;
        if size == 0 {
            return Vec::new();
        }
        let staging = create_buffer(
            &self.device,
            "modern_test_staging",
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));
        map_buffer_read(&self.device, &staging, 0..size).await;
        let data = staging.slice(..size).get_mapped_range();
        bytemuck::cast_slice::<u8, T>(&data).to_vec()
    }

    /// Per-edge `(next sample epoch, next negative-sample epoch)`.
    #[cfg(test)]
    async fn download_schedule(&self) -> Vec<(f64, f64)> {
        let base = self.schedule_base as f64;
        let next_sample = self
            .download::<f32>(&self.next_sample_buffer, self.n_edges)
            .await;
        let next_negative = self
            .download::<f32>(&self.next_negative_buffer, self.n_edges)
            .await;
        next_sample
            .iter()
            .zip(&next_negative)
            .map(|(&s, &n)| (base + s as f64, base + n as f64))
            .collect()
    }

    /// GPU `(to_node_order, source_ranks)`.
    #[cfg(test)]
    async fn download_orders(&self) -> (Vec<usize>, Vec<usize>) {
        let orders = self
            .download::<[u32; 4]>(&self.orders_buffer, self.n_vertices)
            .await;
        (
            orders.iter().map(|o| o[0] as usize).collect(),
            orders.iter().map(|o| o[1] as usize).collect(),
        )
    }

    #[cfg(test)]
    async fn negative_scale(&self) -> f32 {
        self.download::<f32>(&self.scale_buffer, 1).await[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CsrMatrix;
    use crate::layout_modern::tests::{gaussian_blobs, pruned_graph, random_init, trustworthiness};
    use crate::layout_modern::ModernLayout;
    use ndarray::Array2;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let device = pollster::block_on(crate::gpu_optimize::request_device());
        if device.is_none() {
            eprintln!("skipping: no GPU adapter");
        }
        device
    }

    fn epochs_per_sample(graph: &CsrMatrix, n_epochs: usize) -> Vec<f64> {
        crate::graph::make_epochs_per_sample(&graph.data, n_epochs)
            .into_iter()
            .map(f64::from)
            .collect()
    }

    fn gpu_layout(
        (device, queue): (wgpu::Device, wgpu::Queue),
        graph: &CsrMatrix,
        embedding: &Array2<f32>,
        params: &ModernLayoutParams,
    ) -> GpuModernLayout {
        GpuModernLayout::new(
            device,
            queue,
            graph.indptr.clone(),
            graph.indices.clone(),
            epochs_per_sample(graph, params.n_epochs),
            embedding.as_slice().unwrap(),
            embedding.ncols(),
            params.clone(),
            0,
        )
        .expect("GPU layout")
    }

    fn gpu_embedding(gpu: &mut GpuModernLayout) -> Array2<f32> {
        pollster::block_on(gpu.sync_embedding());
        Array2::from_shape_vec((gpu.n_vertices, gpu.dim), gpu.embedding().to_vec()).unwrap()
    }

    fn adam() -> ModernLayoutParams {
        ModernLayoutParams::default()
    }

    fn momentum() -> ModernLayoutParams {
        ModernLayoutParams {
            kind: ModernOptimizerKind::Momentum,
            ..ModernLayoutParams::default()
        }
    }

    /// Without negative samples, the first epoch only depends on the schedule
    /// and the attraction, so it matches the CPU up to f32 rounding.
    fn assert_first_epoch_matches_cpu(n_per_blob: usize, params: ModernLayoutParams) {
        let Some(device) = device() else { return };
        let params = ModernLayoutParams {
            negative_sample_rate: 0.0,
            ..params
        };
        let data = gaussian_blobs(n_per_blob, 8, 6.0, 3);
        let graph = pruned_graph(&data, 10, params.n_epochs);
        let init = random_init(data.nrows(), 2, 5);
        let mut cpu = init.clone();
        let mut cpu_layout = ModernLayout::new(&graph, 2, params.clone());
        let mut gpu = gpu_layout(device, &graph, &init, &params);
        cpu_layout.step(&mut cpu);
        pollster::block_on(gpu.step()).unwrap();
        let difference = gpu_embedding(&mut gpu)
            .iter()
            .zip(cpu.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            difference < 1e-4,
            "{:?}: difference {difference}",
            params.kind
        );
    }

    #[test]
    fn first_epoch_matches_cpu() {
        for params in [adam(), momentum()] {
            assert_first_epoch_matches_cpu(200, params.clone());
            assert_first_epoch_matches_cpu(700, params);
        }
    }

    #[test]
    fn gpu_projection_order_matches_cpu_sort() {
        let Some(device) = device() else { return };
        for n_per_blob in [600, 3000] {
            let data = gaussian_blobs(n_per_blob, 4, 6.0, 7);
            let params = ModernLayoutParams {
                negative_selection_range: 300,
                ..adam()
            };
            let graph = pruned_graph(&data, 5, params.n_epochs);
            let init = random_init(data.nrows(), 3, 9);
            let mut gpu = gpu_layout(device.clone(), &graph, &init, &params);
            let mut cpu_ordering = NegativeOrdering::new(&params, data.nrows());
            cpu_ordering.prepare_with(init.as_slice().unwrap(), 3, &params);
            let mut batch = Batch::default();
            let update = gpu.ordering.prepare(3, &params).unwrap();
            gpu.queue_order_update(update, &mut batch);
            gpu.submit(batch);
            let (order, ranks) = pollster::block_on(gpu.download_orders());
            let n = data.nrows();
            assert!((0..n).all(|rank| ranks[order[rank]] == rank));
            let matching = order
                .iter()
                .zip(&cpu_ordering.to_node_order)
                .filter(|(g, c)| g == c)
                .count();
            assert!(matching * 1000 >= n * 999, "{matching} of {n} ranks match");
        }
    }

    /// The f32 schedules track the CPU f64 schedules over long runs, across
    /// live rate changes and schedule rebases.
    #[test]
    fn schedules_follow_cpu_over_long_runs() {
        let Some(device) = device() else { return };
        let n = 120;
        let mut rng = nndescent::rng::Xoshiro256StarStar::seed_from_u64(17);
        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        for row in 0..n {
            let mut columns: Vec<usize> = (0..6)
                .map(|_| rng.random_range_i64(0, n as i64) as usize)
                .filter(|&c| c != row)
                .collect();
            columns.sort_unstable();
            columns.dedup();
            indices.extend(columns);
            indptr.push(indices.len());
        }
        let epochs_per_sample: Vec<f64> = (0..indices.len())
            .map(|_| 1.0 + (rng.random_f32() * 40.0) as f64)
            .collect();
        let params = ModernLayoutParams {
            n_epochs: 3000,
            negative_sample_rate: 1.3,
            ..adam()
        };
        let init = random_init(n, 2, 1);
        let start_epoch = 7;
        let mut cpu_layout = ModernLayout::from_csr_parts(
            indptr.clone(),
            indices.clone(),
            epochs_per_sample.clone(),
            2,
            params.clone(),
            start_epoch,
        );
        let (gpu_device, queue) = device;
        let mut gpu = GpuModernLayout::new(
            gpu_device,
            queue,
            indptr,
            indices,
            epochs_per_sample.clone(),
            init.as_slice().unwrap(),
            2,
            params,
            start_epoch,
        )
        .unwrap();
        let mut cpu = init.clone();
        for (rate, n_steps) in [(1.3, 1500), (5.0, 700), (0.0, 50), (0.7, 750)] {
            let live = LiveParams {
                alpha: 1.0,
                gamma: 1.0,
                negative_sample_rate: rate,
                a: 1.577,
                b: 0.895,
            };
            cpu_layout.set_live_params(live);
            gpu.set_live_params(live);
            for _ in 0..n_steps {
                cpu_layout.step(&mut cpu);
                pollster::block_on(gpu.step()).unwrap();
            }
            let gpu_schedule = pollster::block_on(gpu.download_schedule());
            let pairs = || {
                cpu_layout
                    .edge_schedule()
                    .iter()
                    .zip(&gpu_schedule)
                    .zip(&epochs_per_sample)
            };
            let close = pairs()
                .filter(|((cpu, gpu), _)| (cpu.next_sample - gpu.0).abs() < 0.05)
                .count();
            assert!(
                close * 100 >= gpu_schedule.len() * 97,
                "rate {rate}: {close} close"
            );
            assert!(pairs().all(|((cpu, gpu), &interval)| {
                let negative_interval = interval / rate.max(0.1) as f64 * 1.5;
                (cpu.next_sample - gpu.0).abs() <= interval + 0.05
                    && (rate == 0.0 || (cpu.next_negative - gpu.1).abs() <= 3.0 * negative_interval)
            }));
        }
        assert_eq!(gpu.epoch(), start_epoch + 3000);
    }

    /// A device with the WebGPU default limits and no features, as a browser
    /// grants without requests.
    fn default_limits_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()
    }

    #[test]
    fn layouts_run_within_webgpu_default_limits() {
        let Some(device) = default_limits_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let data = gaussian_blobs(700, 6, 4.0, 3);
        let graph = pruned_graph(&data, 10, 200);
        let init = random_init(data.nrows(), 2, 4);
        for kind in [ModernOptimizerKind::Adam, ModernOptimizerKind::Momentum] {
            for negative_selection_range in [200_000, 300] {
                let params = ModernLayoutParams {
                    kind,
                    negative_selection_range,
                    exclude_graph_neighbors: negative_selection_range < 1000,
                    n_epochs: 30,
                    ..adam()
                };
                let mut gpu = gpu_layout(device.clone(), &graph, &init, &params);
                for mode in [RateMode::Schedule, RateMode::Flat] {
                    for _ in 0..12 {
                        pollster::block_on(gpu.step_with(mode)).unwrap();
                    }
                }
                let embedding = gpu_embedding(&mut gpu);
                assert!(embedding.iter().all(|x| x.is_finite()));
                let moved = embedding
                    .iter()
                    .zip(init.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    moved > 0.1,
                    "{kind:?} range {negative_selection_range}: moved {moved}"
                );
            }
        }
    }

    #[test]
    fn bindings_fit_webgpu_defaults_for_large_graphs() {
        let webgpu_default = 128 << 20;
        assert!(largest_binding(1_000_000, 25_000_000, 2, 128) <= webgpu_default);
        assert!(largest_binding(1_000_000, 30_000_000, 3, 128) <= webgpu_default);
        assert!(largest_binding(2_000_000, 40_000_000, 2, 128) > webgpu_default);
    }

    /// `cargo test -p umap --release --features gpu million_point_layout -- --ignored`
    #[test]
    #[ignore]
    fn million_point_layout_runs() {
        let Some((device, queue)) = device() else {
            return;
        };
        let n = 1_000_000;
        let degree = 25;
        let mut rng = nndescent::rng::Xoshiro256StarStar::seed_from_u64(3);
        let indptr: Vec<usize> = (0..=n).map(|row| row * degree).collect();
        let indices: Vec<usize> = (0..n)
            .flat_map(|row| {
                let mut columns: Vec<usize> =
                    (1..=degree).map(|offset| (row + offset * 37) % n).collect();
                columns.sort_unstable();
                columns
            })
            .collect();
        let epochs_per_sample: Vec<f64> = (0..indices.len())
            .map(|_| 1.0 + (rng.random_f32() * 10.0) as f64)
            .collect();
        let init = random_init(n, 2, 4);
        for kind in [ModernOptimizerKind::Adam, ModernOptimizerKind::Momentum] {
            let params = ModernLayoutParams { kind, ..adam() };
            let mut gpu = GpuModernLayout::new(
                device.clone(),
                queue.clone(),
                indptr.clone(),
                indices.clone(),
                epochs_per_sample.clone(),
                init.as_slice().unwrap(),
                2,
                params,
                0,
            )
            .expect("25M-edge layout fits");
            let start = std::time::Instant::now();
            for _ in 0..5 {
                pollster::block_on(gpu.step()).unwrap();
            }
            let embedding = gpu_embedding(&mut gpu);
            eprintln!(
                "{kind:?} n {n} edges {}: {:.1} ms/epoch",
                indices.len(),
                start.elapsed().as_secs_f64() * 1e3 / 5.0
            );
            assert!(embedding.iter().all(|x| x.is_finite()));
            assert_ne!(embedding, init);
        }
    }

    /// Fraction of each point's `k` nearest data neighbors that are among its
    /// `k` nearest embedding neighbors.
    fn knn_recall(data: &Array2<f32>, embedding: &Array2<f32>, k: usize) -> f64 {
        let neighbors = |points: &Array2<f32>, i: usize| -> Vec<usize> {
            let mut others: Vec<(f32, usize)> = (0..points.nrows())
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = points
                        .row(i)
                        .iter()
                        .zip(points.row(j).iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (d, j)
                })
                .collect();
            others.sort_by(|a, b| a.0.total_cmp(&b.0));
            others[..k].iter().map(|&(_, j)| j).collect()
        };
        let hits: usize = (0..data.nrows())
            .map(|i| {
                let expected = neighbors(data, i);
                neighbors(embedding, i)
                    .iter()
                    .filter(|j| expected.contains(j))
                    .count()
            })
            .sum();
        hits as f64 / (data.nrows() * k) as f64
    }

    struct Quality {
        trustworthiness: f64,
        recall: f64,
    }

    fn quality(data: &Array2<f32>, embedding: &Array2<f32>) -> Quality {
        assert!(embedding.iter().all(|x| x.is_finite()));
        Quality {
            trustworthiness: trustworthiness(data, embedding, 10),
            recall: knn_recall(data, embedding, 10),
        }
    }

    /// Full runs on both backends reach comparable quality.
    fn assert_quality_matches_cpu(n_per_blob: usize, params: ModernLayoutParams) {
        let Some(device) = device() else { return };
        let data = gaussian_blobs(n_per_blob, 6, 4.0, 13);
        let graph = pruned_graph(&data, 15, params.n_epochs);
        let init = random_init(data.nrows(), 2, 17);
        let mut cpu = init.clone();
        ModernLayout::new(&graph, 2, params.clone()).run(&mut cpu);
        let mut gpu = gpu_layout(device, &graph, &init, &params);
        pollster::block_on(gpu.run()).unwrap();
        let cpu_quality = quality(&data, &cpu);
        let gpu_quality = quality(&data, &gpu_embedding(&mut gpu));
        let label = format!(
            "{:?} n {} range {}: trustworthiness cpu {:.4} gpu {:.4}, recall cpu {:.4} gpu {:.4}",
            params.kind,
            data.nrows(),
            params.negative_selection_range,
            cpu_quality.trustworthiness,
            gpu_quality.trustworthiness,
            cpu_quality.recall,
            gpu_quality.recall,
        );
        eprintln!("{label}");
        assert!(gpu_quality.trustworthiness > 0.85, "{label}");
        assert!(
            (gpu_quality.trustworthiness - cpu_quality.trustworthiness).abs() < 0.02,
            "{label}"
        );
        assert!(
            (gpu_quality.recall - cpu_quality.recall).abs() < 0.05,
            "{label}"
        );
    }

    #[test]
    fn small_layouts_match_cpu_quality() {
        assert_quality_matches_cpu(200, adam());
        assert_quality_matches_cpu(200, momentum());
    }

    #[test]
    fn large_layouts_match_cpu_quality() {
        assert_quality_matches_cpu(700, adam());
        assert_quality_matches_cpu(700, momentum());
    }

    #[test]
    fn windowed_layouts_match_cpu_quality() {
        let windowed = |params: ModernLayoutParams| ModernLayoutParams {
            negative_selection_range: 400,
            exclude_graph_neighbors: true,
            ..params
        };
        assert_quality_matches_cpu(700, windowed(adam()));
        assert_quality_matches_cpu(700, windowed(momentum()));
    }

    #[test]
    fn scale_adaptation_moves_scale_within_bounds() {
        let Some(device) = device() else { return };
        let data = gaussian_blobs(700, 6, 4.0, 19);
        let params = ModernLayoutParams {
            n_epochs: 60,
            negative_selection_range: 300,
            ..adam()
        };
        let graph = pruned_graph(&data, 10, params.n_epochs);
        let init = random_init(data.nrows(), 2, 23);
        let mut gpu = gpu_layout(device, &graph, &init, &params);
        let initial = pollster::block_on(gpu.negative_scale());
        assert!((initial as f64 - (300.0f64 / 1400.0).powf(0.75)).abs() < 1e-6);
        pollster::block_on(gpu.run()).unwrap();
        let adapted = pollster::block_on(gpu.negative_scale());
        assert!(adapted.is_finite() && adapted != initial);
        assert!(adapted >= 0.5 * initial && adapted <= 2.0 * initial);
    }
}

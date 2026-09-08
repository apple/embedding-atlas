/// GPU-accelerated distance computation using wgpu compute shaders.
///
/// Provides a `GpuContext` that uploads the data matrix once and supports
/// batch distance computation with threshold filtering on the GPU.
///
/// All public methods are async to support both native and WASM (WebGPU) targets.
/// On native, callers wrap with `pollster::block_on`. On WASM, they use `.await`.
use ndarray::Array2;

/// A filtered distance result from the GPU: (target_index, source_index, distance).
pub type FilteredResult = (u32, u32, f32);

/// Uniform parameters passed to the distance compute shader.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    n_features: u32,
    n_pairs: u32,
}

/// Uniform parameters passed to the projection compute shader.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ProjectionParams {
    n_features: u32,
    n_points: u32,
    offset: f32,
    /// Base index offset for chunked dispatch when n_points exceeds max workgroups.
    base_idx: u32,
}

/// Uniform parameters for the batch projection shader.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BatchProjectionParams {
    n_features: u32,
    n_total_points: u32,
    /// Base index offset for chunked dispatch when n_total_points exceeds max workgroups.
    base_idx: u32,
    _pad: u32,
}

/// Each output entry is 3 × u32 = 12 bytes. Max 2 outputs per pair.
const OUTPUT_ENTRY_BYTES: u64 = 3 * 4;

/// Holds wgpu device state and pre-allocated buffers for GPU distance computation.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    data_buffer: wgpu::Buffer,
    threshold_buffer: wgpu::Buffer,
    /// Pre-allocated pair/output/staging buffers, sized for max_pairs_per_chunk.
    pairs_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    output_staging: wgpu::Buffer,
    counter_staging: wgpu::Buffer,
    n_points: u32,
    n_features: u32,
    /// Max pairs per GPU dispatch, derived from device buffer size limits.
    max_pairs_per_chunk: usize,
    // Projection pipeline for RP tree construction
    proj_pipeline: wgpu::ComputePipeline,
    proj_bind_group_layout: wgpu::BindGroupLayout,
    proj_hyperplane_buffer: wgpu::Buffer,
    proj_indices_buffer: wgpu::Buffer,
    proj_margins_buffer: wgpu::Buffer,
    proj_margins_staging: wgpu::Buffer,
    proj_params_buffer: wgpu::Buffer,
    // Batch projection pipeline for level-wise RP tree construction
    batch_proj_pipeline: wgpu::ComputePipeline,
    batch_proj_bind_group_layout: wgpu::BindGroupLayout,
    batch_proj_hyperplanes_buffer: wgpu::Buffer,
    batch_proj_offsets_buffer: wgpu::Buffer,
    batch_proj_point_indices_buffer: wgpu::Buffer,
    batch_proj_node_ids_buffer: wgpu::Buffer,
    batch_proj_margins_buffer: wgpu::Buffer,
    batch_proj_margins_staging: wgpu::Buffer,
    batch_proj_params_buffer: wgpu::Buffer,
    /// Max number of nodes per batch projection dispatch.
    max_batch_nodes: usize,
    /// Max workgroups per dispatch dimension (from device limits).
    max_workgroups_per_dim: u32,
}

/// Async helper: map a buffer for reading and wait for it to be available.
///
/// On native, `device.poll(Wait)` blocks until the map completes synchronously,
/// so the callback fires before poll returns and no async yield is needed.
/// On WASM/WebGPU, `device.poll` is a no-op; we yield to the browser event loop
/// via the Pending→wake pattern so the internal JS mapAsync Promise can resolve.
async fn map_buffer_read(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    range: std::ops::Range<u64>,
) {
    let slice = buffer.slice(range);

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Native: poll(Wait) blocks until the map completes synchronously.
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::Maintain::Wait);
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM: callback fires when the browser resolves the internal JS Promise.
        // We use Rc<Cell> + Waker since everything is single-threaded.
        use std::cell::Cell;
        use std::rc::Rc;

        let state: Rc<(Cell<bool>, Cell<Option<std::task::Waker>>)> =
            Rc::new((Cell::new(false), Cell::new(None)));
        let state_cb = state.clone();

        slice.map_async(wgpu::MapMode::Read, move |_| {
            state_cb.0.set(true);
            if let Some(waker) = state_cb.1.take() {
                waker.wake();
            }
        });
        device.poll(wgpu::Maintain::Wait); // no-op on WebGPU

        // If already done (unlikely on WASM), skip the yield.
        if !state.0.get() {
            std::future::poll_fn(|cx| {
                if state.0.get() {
                    std::task::Poll::Ready(())
                } else {
                    state.1.set(Some(cx.waker().clone()));
                    std::task::Poll::Pending
                }
            })
            .await;
        }
    }
}

impl GpuContext {
    /// Create a new GPU context, uploading the data matrix and compiling the
    /// distance shader for the given metric.
    ///
    /// Returns `None` if no suitable GPU adapter is available.
    pub async fn new(data: &Array2<f32>, metric: &str) -> Option<Self> {
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        // Request the adapter's native limits, not the restrictive WebGPU defaults.
        // This gives us the full buffer sizes the hardware supports (e.g. multi-GB on
        // Metal/Vulkan) instead of the 128MB WebGPU minimum.
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("nndescent-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let n_points = data.nrows() as u32;
        let n_features = data.ncols() as u32;

        // Query device limits to size buffers correctly
        let limits = device.limits();
        let max_binding_size = limits.max_storage_buffer_binding_size as u64;
        let max_buffer_size = limits.max_buffer_size;
        let buffer_limit = max_binding_size.min(max_buffer_size);

        // Max pairs per chunk: output buffer = 2 * n_pairs * 12 bytes ≤ buffer_limit
        // Also pairs buffer = n_pairs * 8 bytes ≤ buffer_limit
        // Also limited by max dispatch workgroup count (65535 × 256 pairs)
        let max_pairs_from_output = buffer_limit / (2 * OUTPUT_ENTRY_BYTES);
        let max_pairs_from_pairs = buffer_limit / 8;
        let max_pairs_from_dispatch = 65535u64 * 256;
        let max_pairs_per_chunk = max_pairs_from_output
            .min(max_pairs_from_pairs)
            .min(max_pairs_from_dispatch) as usize;

        // Validate that the data matrix and per-point buffers fit within device limits.
        let data_bytes = (n_points as u64) * (n_features as u64) * 4;
        let per_point_bytes = (n_points as u64) * 4;
        if data_bytes > buffer_limit || per_point_bytes > buffer_limit {
            return None; // Dataset too large for this GPU
        }

        // Upload data matrix
        let data_slice = data
            .as_slice()
            .expect("data must be contiguous (standard layout)");
        let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("data"),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&data_buffer, 0, bytemuck::cast_slice(data_slice));

        // Pre-allocate threshold buffer (reused across dispatches)
        let threshold_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thresholds"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-allocate pair buffer (sized for max chunk)
        let max_pairs_bytes = (max_pairs_per_chunk as u64) * 2 * 4;
        let pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairs"),
            size: max_pairs_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-allocate output buffer
        let max_output_bytes = (max_pairs_per_chunk as u64) * 2 * OUTPUT_ENTRY_BYTES;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: max_output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Counter buffer (4 bytes)
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params buffer (8 bytes)
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-allocate staging buffers for readback
        let output_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_staging"),
            size: max_output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Select shader source
        let shader_src = match metric {
            "euclidean" | "l2" => include_str!("shaders/euclidean.wgsl"),
            "cosine" | "alternative_cosine" => include_str!("shaders/cosine.wgsl"),
            _ => return None,
        };

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distance_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance_bgl"),
            entries: &[
                bgl_entry(0, true),  // data
                bgl_entry(1, true),  // pairs
                bgl_entry(2, true),  // thresholds
                bgl_entry(3, false), // output
                bgl_entry(4, false), // counter
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("distance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Projection pipeline for RP tree construction ===
        let proj_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("projection_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/projection.wgsl").into()),
        });

        let proj_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("projection_bgl"),
                entries: &[
                    bgl_entry(0, true),  // data (reuse existing)
                    bgl_entry(1, true),  // hyperplane
                    bgl_entry(2, true),  // indices
                    bgl_entry(3, false), // margins (output)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let proj_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("projection_pipeline_layout"),
            bind_group_layouts: &[&proj_bind_group_layout],
            push_constant_ranges: &[],
        });

        let proj_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("projection_pipeline"),
            layout: Some(&proj_pipeline_layout),
            module: &proj_shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pre-allocate projection buffers (sized for n_points max)
        let proj_hyperplane_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_hyperplane"),
            size: (n_features as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_indices"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_margins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_margins"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let proj_margins_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_margins_staging"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_params"),
            size: std::mem::size_of::<ProjectionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Batch projection pipeline for level-wise tree construction ===
        let batch_proj_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch_projection_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/batch_projection.wgsl").into()),
        });

        let batch_proj_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("batch_projection_bgl"),
                entries: &[
                    bgl_entry(0, true),  // data
                    bgl_entry(1, true),  // hyperplanes
                    bgl_entry(2, true),  // offsets
                    bgl_entry(3, true),  // point_indices
                    bgl_entry(4, true),  // node_ids
                    bgl_entry(5, false), // margins
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let batch_proj_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("batch_projection_pipeline_layout"),
                bind_group_layouts: &[&batch_proj_bind_group_layout],
                push_constant_ranges: &[],
            });

        let batch_proj_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("batch_projection_pipeline"),
                layout: Some(&batch_proj_pipeline_layout),
                module: &batch_proj_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Batch projection buffers — sized for n_points (max total points per level)
        // Hyperplanes buffer: need enough for many nodes × n_features.
        // Derive max_batch_nodes from what actually fits in buffer_limit.
        let max_batch_nodes = (buffer_limit / (n_features as u64 * 4)).min(2048) as usize;
        let batch_proj_hyperplanes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_hyperplanes"),
            size: (max_batch_nodes as u64) * (n_features as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_proj_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_offsets"),
            size: (max_batch_nodes as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_proj_point_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_point_indices"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_proj_node_ids_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_node_ids"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_proj_margins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_margins"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let batch_proj_margins_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_margins_staging"),
            size: per_point_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_proj_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_proj_params"),
            size: std::mem::size_of::<BatchProjectionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ctx = GpuContext {
            device,
            queue,
            pipeline,
            bind_group_layout,
            data_buffer,
            threshold_buffer,
            pairs_buffer,
            output_buffer,
            counter_buffer,
            params_buffer,
            output_staging,
            counter_staging,
            n_points,
            n_features,
            max_pairs_per_chunk,
            proj_pipeline,
            proj_bind_group_layout,
            proj_hyperplane_buffer,
            proj_indices_buffer,
            proj_margins_buffer,
            proj_margins_staging,
            proj_params_buffer,
            batch_proj_pipeline,
            batch_proj_bind_group_layout,
            batch_proj_hyperplanes_buffer,
            batch_proj_offsets_buffer,
            batch_proj_point_indices_buffer,
            batch_proj_node_ids_buffer,
            batch_proj_margins_buffer,
            batch_proj_margins_staging,
            batch_proj_params_buffer,
            max_batch_nodes,
            max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
        };

        // Warm up the GPU pipeline with a tiny dummy dispatch to trigger shader
        // compilation and avoid a 200ms cold-start on the first real dispatch.
        let warmup_pairs = [0u32, 1.min(n_points - 1)];
        let warmup_thresholds = vec![f32::INFINITY; n_points as usize];
        ctx.compute_filtered_distances(&warmup_pairs, &warmup_thresholds)
            .await;

        Some(ctx)
    }

    /// Compute distances for the given pairs, filtering by thresholds.
    ///
    /// `pairs` is a flat slice of `[p0, q0, p1, q1, ...]` indices.
    /// `thresholds` has length `n_points` — the max distance in each point's heap.
    ///
    /// Automatically chunks into multiple dispatches if pairs exceed buffer limits.
    pub async fn compute_filtered_distances(
        &self,
        pairs: &[u32],
        thresholds: &[f32],
    ) -> Vec<FilteredResult> {
        let total_pairs = pairs.len() / 2;
        if total_pairs == 0 {
            return Vec::new();
        }

        // Upload thresholds once (shared across all chunks)
        self.queue
            .write_buffer(&self.threshold_buffer, 0, bytemuck::cast_slice(thresholds));

        // If fits in one chunk, dispatch directly
        if total_pairs <= self.max_pairs_per_chunk {
            return self.dispatch_chunk(pairs, total_pairs).await;
        }

        // Otherwise chunk
        let mut all_results = Vec::new();
        for chunk_start in (0..total_pairs).step_by(self.max_pairs_per_chunk) {
            let chunk_end = (chunk_start + self.max_pairs_per_chunk).min(total_pairs);
            let chunk_pairs = &pairs[chunk_start * 2..chunk_end * 2];
            let chunk_n = chunk_end - chunk_start;
            let mut results = self.dispatch_chunk(chunk_pairs, chunk_n).await;
            all_results.append(&mut results);
        }
        all_results
    }

    /// Dispatch a single chunk of pairs (must fit within buffer limits).
    async fn dispatch_chunk(&self, pairs: &[u32], n_pairs: usize) -> Vec<FilteredResult> {
        // Upload pairs, reset counter, upload params
        self.queue
            .write_buffer(&self.pairs_buffer, 0, bytemuck::cast_slice(pairs));
        self.queue
            .write_buffer(&self.counter_buffer, 0, bytemuck::cast_slice(&[0u32]));
        let params = Params {
            n_features: self.n_features,
            n_pairs: n_pairs as u32,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("distance_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.threshold_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Single submission: compute + copy counter + copy output
        let workgroup_count = ((n_pairs as u32) + 255) / 256;
        let max_output_bytes = (n_pairs as u64) * 2 * OUTPUT_ENTRY_BYTES;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("distance_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("distance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.counter_buffer, 0, &self.counter_staging, 0, 4);
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.output_staging,
            0,
            max_output_bytes,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read counter first
        map_buffer_read(&self.device, &self.counter_staging, 0..4).await;
        let count = {
            let data = self.counter_staging.slice(..4).get_mapped_range();
            let count = *bytemuck::from_bytes::<u32>(&data[..4]) as usize;
            drop(data);
            self.counter_staging.unmap();
            count
        };

        if count == 0 {
            return Vec::new();
        }

        // Read output — data is already in staging from the single submission
        let read_bytes = (count as u64) * OUTPUT_ENTRY_BYTES;
        map_buffer_read(&self.device, &self.output_staging, 0..read_bytes).await;
        let results = {
            let data = self.output_staging.slice(..read_bytes).get_mapped_range();
            let u32_data: &[u32] = bytemuck::cast_slice(&data);
            let mut results = Vec::with_capacity(count);
            for i in 0..count {
                let target = u32_data[i * 3];
                let source = u32_data[i * 3 + 1];
                let dist = f32::from_bits(u32_data[i * 3 + 2]);
                results.push((target, source, dist));
            }
            drop(data);
            self.output_staging.unmap();
            results
        };

        results
    }

    /// Compute distances without threshold filtering — all distances are returned.
    /// Used for `init_rp_tree` where the heap is initially empty.
    pub async fn compute_all_distances(&self, pairs: &[u32]) -> Vec<FilteredResult> {
        let thresholds = vec![f32::INFINITY; self.n_points as usize];
        self.compute_filtered_distances(pairs, &thresholds).await
    }

    /// Compute dot-product projections of points onto a hyperplane.
    ///
    /// Returns `margins[i] = dot(hyperplane, data[indices[i]]) + offset` for each index.
    /// Used by RP tree construction to determine which side of a split each point falls on.
    pub async fn compute_projections(
        &self,
        hyperplane: &[f32],
        offset: f32,
        indices: &[u32],
    ) -> Vec<f32> {
        let n_points = indices.len();
        if n_points == 0 {
            return Vec::new();
        }

        // Upload hyperplane and indices (shared across all dispatch chunks)
        self.queue.write_buffer(
            &self.proj_hyperplane_buffer,
            0,
            bytemuck::cast_slice(hyperplane),
        );
        self.queue
            .write_buffer(&self.proj_indices_buffer, 0, bytemuck::cast_slice(indices));

        let max_invocations_per_dispatch = self.max_workgroups_per_dim as usize * 256;

        // Create bind group (shared across chunks)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("projection_bg"),
            layout: &self.proj_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.proj_hyperplane_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.proj_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.proj_margins_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.proj_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch in chunks to respect max_compute_workgroups_per_dimension
        let mut base_idx = 0usize;
        while base_idx < n_points {
            let remaining = n_points - base_idx;
            let chunk_size = remaining.min(max_invocations_per_dispatch);
            let workgroup_count = ((chunk_size as u32) + 255) / 256;

            let params = ProjectionParams {
                n_features: self.n_features,
                n_points: n_points as u32,
                offset,
                base_idx: base_idx as u32,
            };
            self.queue
                .write_buffer(&self.proj_params_buffer, 0, bytemuck::cast_slice(&[params]));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("projection_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("projection_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.proj_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroup_count, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            base_idx += chunk_size;
        }

        // Copy and read back all margins at once
        let read_bytes = (n_points as u64) * 4;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("projection_copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.proj_margins_buffer,
            0,
            &self.proj_margins_staging,
            0,
            read_bytes,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        map_buffer_read(&self.device, &self.proj_margins_staging, 0..read_bytes).await;
        let data = self
            .proj_margins_staging
            .slice(..read_bytes)
            .get_mapped_range();
        let margins: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.proj_margins_staging.unmap();

        margins
    }

    /// Batch projection: compute margins for multiple nodes in a single GPU dispatch.
    ///
    /// Each entry in the batch is (hyperplane, offset, point_indices).
    /// Returns margins in the same order: first all margins for batch[0], then batch[1], etc.
    ///
    /// This amortizes GPU dispatch overhead across all nodes at a tree level.
    pub async fn compute_projections_batch(&self, batch: &[(&[f32], f32, &[u32])]) -> Vec<f32> {
        if batch.is_empty() {
            return Vec::new();
        }

        // If the batch fits in one dispatch, process it directly.
        // Otherwise, chunk by max_batch_nodes and concatenate results.
        if batch.len() <= self.max_batch_nodes {
            return self.compute_projections_batch_chunk(batch).await;
        }

        let mut all_margins = Vec::new();
        for chunk in batch.chunks(self.max_batch_nodes) {
            let chunk_margins = self.compute_projections_batch_chunk(chunk).await;
            all_margins.extend_from_slice(&chunk_margins);
        }
        all_margins
    }

    /// Dispatch a single chunk of nodes that fits within the pre-allocated
    /// hyperplane/offset buffers (at most `max_batch_nodes` nodes).
    async fn compute_projections_batch_chunk(&self, batch: &[(&[f32], f32, &[u32])]) -> Vec<f32> {
        debug_assert!(batch.len() <= self.max_batch_nodes);

        // Flatten batch into GPU buffers
        let n_features = self.n_features as usize;
        let mut all_hyperplanes: Vec<f32> = Vec::new();
        let mut all_offsets: Vec<f32> = Vec::new();
        let mut all_point_indices: Vec<u32> = Vec::new();
        let mut all_node_ids: Vec<u32> = Vec::new();

        for (node_id, (hp, offset, indices)) in batch.iter().enumerate() {
            all_hyperplanes.extend_from_slice(hp);
            // Pad if hyperplane is shorter than n_features
            if hp.len() < n_features {
                all_hyperplanes.resize(all_hyperplanes.len() + n_features - hp.len(), 0.0);
            }
            all_offsets.push(*offset);
            for &idx in *indices {
                all_point_indices.push(idx);
                all_node_ids.push(node_id as u32);
            }
        }

        let n_total = all_point_indices.len();
        if n_total == 0 {
            return Vec::new();
        }

        // Upload data
        self.queue.write_buffer(
            &self.batch_proj_hyperplanes_buffer,
            0,
            bytemuck::cast_slice(&all_hyperplanes),
        );
        self.queue.write_buffer(
            &self.batch_proj_offsets_buffer,
            0,
            bytemuck::cast_slice(&all_offsets),
        );
        self.queue.write_buffer(
            &self.batch_proj_point_indices_buffer,
            0,
            bytemuck::cast_slice(&all_point_indices),
        );
        self.queue.write_buffer(
            &self.batch_proj_node_ids_buffer,
            0,
            bytemuck::cast_slice(&all_node_ids),
        );

        let params = BatchProjectionParams {
            n_features: self.n_features,
            n_total_points: n_total as u32,
            base_idx: 0,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.batch_proj_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        // Create bind group (shared across dispatch chunks)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_projection_bg"),
            layout: &self.batch_proj_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.batch_proj_hyperplanes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.batch_proj_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.batch_proj_point_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.batch_proj_node_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.batch_proj_margins_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.batch_proj_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch in chunks to respect max_compute_workgroups_per_dimension
        let max_invocations_per_dispatch = self.max_workgroups_per_dim as usize * 256;
        let mut base_idx = 0usize;
        while base_idx < n_total {
            let remaining = n_total - base_idx;
            let chunk_size = remaining.min(max_invocations_per_dispatch);
            let workgroup_count = ((chunk_size as u32) + 255) / 256;

            let params = BatchProjectionParams {
                n_features: self.n_features,
                n_total_points: n_total as u32,
                base_idx: base_idx as u32,
                _pad: 0,
            };
            self.queue.write_buffer(
                &self.batch_proj_params_buffer,
                0,
                bytemuck::cast_slice(&[params]),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_projection_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("batch_projection_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.batch_proj_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroup_count, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            base_idx += chunk_size;
        }

        // Copy and read back all margins at once
        let read_bytes = (n_total as u64) * 4;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_projection_copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.batch_proj_margins_buffer,
            0,
            &self.batch_proj_margins_staging,
            0,
            read_bytes,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back
        map_buffer_read(
            &self.device,
            &self.batch_proj_margins_staging,
            0..read_bytes,
        )
        .await;
        let data = self
            .batch_proj_margins_staging
            .slice(..read_bytes)
            .get_mapped_range();
        let margins: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.batch_proj_margins_staging.unmap();

        margins
    }
}

/// Helper to create a storage buffer bind group layout entry.
fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

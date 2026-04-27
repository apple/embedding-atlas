/// GPU-accelerated UMAP SGD optimization using wgpu compute shaders.
///
/// Uses a two-pass approach per sub-batch:
///   1. accumulate_grads: computes per-edge gradients and atomicAdds them
///      to a fixed-point i32 buffer (no lost updates from GPU write conflicts).
///   2. apply_grads: converts accumulated gradients from fixed-point, applies
///      them to the embedding with the current learning rate, and clears the
///      accumulator.
///
/// Edges are split into sub-batches dispatched sequentially to approximate
/// CPU Hogwild behavior where later edges see partially-updated positions.
use ndarray::Array2;
use nndescent::Logger;

/// Uniform parameters passed to the optimize compute shader.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
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
    /// Base element offset for chunked apply_grads dispatch.
    apply_offset: u32,
    _pad2: f32,
    _pad3: f32,
}

/// Holds wgpu state for the GPU optimization pipeline.
struct OptimizeGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    accumulate_pipeline: wgpu::ComputePipeline,
    apply_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    embedding_buffer: wgpu::Buffer,
    edges_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    grad_accum_buffer: wgpu::Buffer,
    embedding_staging: wgpu::Buffer,
    n_vertices: u32,
    dim: u32,
    max_edges_per_dispatch: usize,
    max_workgroups_per_dim: u32,
}

/// Async helper: map a buffer for reading and wait for it to be available.
/// Duplicated from nndescent::gpu (internal to that crate).
async fn map_buffer_read(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    range: std::ops::Range<u64>,
) {
    let slice = buffer.slice(range);

    #[cfg(not(target_arch = "wasm32"))]
    {
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::Maintain::Wait);
    }

    #[cfg(target_arch = "wasm32")]
    {
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
        device.poll(wgpu::Maintain::Wait);

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

impl OptimizeGpuContext {
    /// Create a new GPU context for UMAP optimization.
    ///
    /// Uploads the initial embedding and compiles the optimization shaders.
    /// Returns `None` if no suitable GPU adapter is available.
    async fn new(embedding: &[f32], n_vertices: u32, dim: u32) -> Option<Self> {
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

        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("umap-optimize-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let limits = device.limits();
        let max_binding_size = limits.max_storage_buffer_binding_size as u64;
        let max_buffer_size = limits.max_buffer_size;
        let buffer_limit = max_binding_size.min(max_buffer_size);

        // Each edge is 4 × u32 = 16 bytes
        let max_edges_from_buffer = buffer_limit / 16;
        let max_edges_from_dispatch = 65535u64 * 256;
        let max_edges_per_dispatch = max_edges_from_buffer.min(max_edges_from_dispatch) as usize;

        // Validate that embedding and grad_accum fit within device limits
        let embedding_bytes = (embedding.len() * 4) as u64;
        let grad_accum_bytes = (n_vertices as u64) * (dim as u64) * 4;
        if embedding_bytes > buffer_limit || grad_accum_bytes > buffer_limit {
            return None; // Embedding too large for this GPU
        }

        // Upload embedding
        let embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding"),
            size: embedding_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&embedding_buffer, 0, bytemuck::cast_slice(embedding));

        // Pre-allocate edges buffer (sized for max chunk)
        let edges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edges"),
            size: (max_edges_per_dispatch as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("optimize_params"),
            size: std::mem::size_of::<OptimizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Gradient accumulation buffer (atomic<i32>, same layout as embedding)
        let grad_accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_accum"),
            size: grad_accum_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-initialize
        queue.write_buffer(&grad_accum_buffer, 0, &vec![0u8; grad_accum_bytes as usize]);

        // Staging buffer for final readback
        let embedding_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_staging"),
            size: embedding_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compile shader (contains both entry points)
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("optimize_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/optimize.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("optimize_bgl"),
            entries: &[
                bgl_entry(0, false), // embedding (read_write)
                bgl_entry(1, true),  // edges (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                bgl_entry(3, false), // grad_accum (read_write)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("optimize_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let accumulate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("accumulate_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("accumulate_grads"),
                compilation_options: Default::default(),
                cache: None,
            });

        let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apply_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("apply_grads"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Warm up with a tiny dispatch to trigger shader compilation
        let warmup_params = OptimizeParams {
            n_vertices,
            dim,
            n_edges: 0,
            _pad0: 0,
            a: 1.0,
            b: 1.0,
            gamma: 1.0,
            alpha: 0.0,
            grad_clamp: 20.0,
            apply_offset: 0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[warmup_params]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warmup_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: embedding_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_accum_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warmup_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&accumulate_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_apply_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&apply_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        Some(OptimizeGpuContext {
            device,
            queue,
            accumulate_pipeline,
            apply_pipeline,
            bind_group_layout,
            embedding_buffer,
            edges_buffer,
            params_buffer,
            grad_accum_buffer,
            embedding_staging,
            n_vertices,
            dim,
            max_edges_per_dispatch,
            max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
        })
    }

    /// Run one epoch of SGD on the GPU.
    ///
    /// `packed_edges` is a flat [head, tail, n_neg, rng_seed, ...] array (4 u32 per edge).
    /// All edges accumulate gradients atomically in one pass, then a second pass
    /// applies the accumulated gradients to the embedding.
    fn run_epoch(
        &self,
        packed_edges: &[u32],
        n_edges: usize,
        alpha: f32,
        a: f32,
        b: f32,
        gamma: f32,
        grad_clamp: f32,
    ) {
        if n_edges == 0 {
            return;
        }

        // Accumulate gradients (chunk if edges exceed buffer capacity)
        if n_edges > self.max_edges_per_dispatch {
            for chunk_start in (0..n_edges).step_by(self.max_edges_per_dispatch) {
                let chunk_end = (chunk_start + self.max_edges_per_dispatch).min(n_edges);
                let chunk_n = chunk_end - chunk_start;
                let chunk_data = &packed_edges[chunk_start * 4..chunk_end * 4];
                self.dispatch_accumulate(chunk_data, chunk_n, alpha, a, b, gamma, grad_clamp);
            }
        } else {
            self.dispatch_accumulate(packed_edges, n_edges, alpha, a, b, gamma, grad_clamp);
        }

        // Apply accumulated gradients and clear accumulator
        // Chunk dispatch to respect max_compute_workgroups_per_dimension
        let total_elements = self.n_vertices * self.dim;
        let max_invocations = self.max_workgroups_per_dim * 256;
        let mut base = 0u32;
        while base < total_elements {
            let remaining = total_elements - base;
            let chunk = remaining.min(max_invocations);
            let workgroups = (chunk + 255) / 256;
            self.dispatch_apply(workgroups, base);
            base += chunk;
        }
    }

    /// Upload edges and dispatch the accumulate_grads shader.
    fn dispatch_accumulate(
        &self,
        packed_edges: &[u32],
        n_edges: usize,
        alpha: f32,
        a: f32,
        b: f32,
        gamma: f32,
        grad_clamp: f32,
    ) {
        let params = OptimizeParams {
            n_vertices: self.n_vertices,
            dim: self.dim,
            n_edges: n_edges as u32,
            _pad0: 0,
            a,
            b,
            gamma,
            alpha,
            grad_clamp,
            apply_offset: 0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&self.edges_buffer, 0, bytemuck::cast_slice(packed_edges));

        let bind_group = self.create_bind_group("accumulate_bg");

        let workgroup_count = ((n_edges as u32) + 255) / 256;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("accumulate_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accumulate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch the apply_grads shader to update the embedding and clear grad_accum.
    ///
    /// `apply_offset` is the base element index for this chunk of the apply pass.
    fn dispatch_apply(&self, apply_workgroups: u32, apply_offset: u32) {
        // Update apply_offset in params. The other fields (alpha, n_vertices, dim, etc.)
        // were already uploaded in dispatch_accumulate and don't change between chunks.
        // We re-upload the full params to set apply_offset.
        self.queue.write_buffer(
            &self.params_buffer,
            std::mem::offset_of!(OptimizeParams, apply_offset) as u64,
            bytemuck::cast_slice(&[apply_offset]),
        );

        let bind_group = self.create_bind_group("apply_bg");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apply_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apply_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(apply_workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Create a bind group with all 4 bindings.
    fn create_bind_group(&self, label: &str) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.embedding_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.edges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.grad_accum_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Download the final embedding from the GPU.
    async fn download_embedding(&self) -> Vec<f32> {
        let size = (self.n_vertices as u64) * (self.dim as u64) * 4;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.embedding_buffer, 0, &self.embedding_staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        map_buffer_read(&self.device, &self.embedding_staging, 0..size).await;
        let data = self.embedding_staging.slice(..size).get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.embedding_staging.unmap();
        result
    }
}

/// GPU-accelerated version of `optimize_layout_euclidean`.
///
/// Returns `Some(())` on success, `None` if GPU is unavailable.
/// The epoch scheduling logic is identical to the CPU version.
#[allow(clippy::too_many_arguments)]
pub async fn optimize_layout_gpu(
    embedding: &mut Array2<f32>,
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f32],
    n_epochs: usize,
    n_vertices: usize,
    n_neighbors: usize,
    a: f64,
    b: f64,
    gamma: f32,
    initial_alpha: f32,
    negative_sample_rate: f32,
    rng_state: [i64; 3],
    logger: &mut Logger,
) -> Option<()> {
    let dim = embedding.ncols();
    let n_edges = head.len();

    let emb_slice = embedding.as_slice().expect("embedding not contiguous");

    let ctx = OptimizeGpuContext::new(emb_slice, n_vertices as u32, dim as u32).await?;

    logger.log("Using GPU for layout optimization");

    let a_f32 = a as f32;
    let b_f32 = b as f32;

    // Accumulated gradient clamp, derived from n_neighbors.
    // Each vertex receives ~7K gradient contributions per epoch (K as head,
    // K as tail for attractive, ~5K for repulsive with negative_sample_rate=5).
    // Since contributions point in different directions, the expected magnitude
    // per component scales as sqrt(N) (random walk). We clamp at
    // EDGE_CLAMP * sqrt(7 * K) to allow normal accumulation while preventing
    // extreme outliers from overshooting.
    let edge_clamp = 4.0f32;
    let grad_clamp = edge_clamp * (7.0 * n_neighbors as f32).sqrt();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    let mut epoch_of_next_sample = epochs_per_sample.to_vec();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    // Base seed for per-edge RNG on GPU
    let base_seed = (rng_state[0] as u32)
        .wrapping_add(rng_state[1] as u32)
        .wrapping_add(rng_state[2] as u32);

    let mut packed_edges: Vec<u32> = Vec::new();

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        // Phase 1: Collect active edges (same logic as CPU)
        packed_edges.clear();
        let mut n_active = 0usize;
        for i in 0..n_edges {
            if epoch_of_next_sample[i] > epoch as f32 {
                continue;
            }

            let n_neg = ((epoch as f32 - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]) as u32;

            // Per-edge RNG seed: deterministic from (base_seed, edge_idx, epoch)
            let rng_seed = base_seed
                .wrapping_mul(epoch as u32 + 1)
                .wrapping_add(i as u32 * 2654435761);

            packed_edges.push(head[i] as u32);
            packed_edges.push(tail[i] as u32);
            packed_edges.push(n_neg);
            packed_edges.push(rng_seed);
            n_active += 1;

            epoch_of_next_sample[i] += epochs_per_sample[i];
            epoch_of_next_negative_sample[i] += n_neg as f32 * epochs_per_negative_sample[i];
        }

        // Phase 2: Dispatch to GPU (atomic gradient accumulation)
        ctx.run_epoch(
            &packed_edges,
            n_active,
            alpha,
            a_f32,
            b_f32,
            gamma,
            grad_clamp,
        );

        if n_epochs >= 10 && epoch % (n_epochs / 10) == 0 {
            logger.log(&format!("completed {} / {} epochs", epoch, n_epochs));
        }
        logger.stage_progress((epoch + 1) as f64 / n_epochs as f64, None);
    }

    // Download result
    let result = ctx.download_embedding().await;
    let emb_mut = embedding
        .as_slice_memory_order_mut()
        .expect("embedding not contiguous");
    emb_mut.copy_from_slice(&result);

    Some(())
}

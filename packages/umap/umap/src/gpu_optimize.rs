/// Resident GPU context for UMAP velocity-SGD optimization (wgpu compute).
///
/// All buffers (embedding, edge list, per-edge schedule counters, gradient
/// accumulator, velocity) are uploaded once and live for the whole run. Each
/// `run_epoch` dispatches the self-scheduling `accumulate_grads` shader (in
/// sub-batches for SGD, a single round for Momentum) followed by `apply_grads`.
/// The CPU mirror is refreshed by `download_embedding` at the end of a step.
///
/// Two-pass-per-sub-batch design: `accumulate_grads` atomicAdds per-edge
/// gradients into a fixed-point i32 buffer (no lost updates from write
/// conflicts); `apply_grads` reads/clears the accumulator and applies the step.
use crate::optimize::{LiveParams, Optimizer};

/// Uniform parameters passed to the optimize compute shaders. std140-compatible:
/// 16 four-byte scalars = 64 bytes, every member naturally aligned.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
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
    /// Base element offset for chunked apply_grads dispatch.
    apply_offset: u32,
    /// Sub-batch edge range [edge_offset, edge_end) for accumulate_grads.
    edge_offset: u32,
    edge_end: u32,
    /// 0 = Sgd, 1 = Momentum.
    optimizer: u32,
    beta: f32,
}

/// Holds resident wgpu state for the GPU optimization pipeline.
pub struct OptimizeGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    accumulate_pipeline: wgpu::ComputePipeline,
    apply_pipeline: wgpu::ComputePipeline,
    rebase_pipeline: wgpu::ComputePipeline,
    accumulate_bind_group: wgpu::BindGroup,
    apply_bind_group: wgpu::BindGroup,
    rebase_bind_group: wgpu::BindGroup,
    embedding_buffer: wgpu::Buffer,
    // head/tail/eps are uploaded once and only referenced by the accumulate bind
    // group; retained here so the buffers outlive that bind group.
    #[allow(dead_code)]
    head_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    tail_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    eps_buffer: wgpu::Buffer,
    eons_buffer: wgpu::Buffer,
    eonns_buffer: wgpu::Buffer,
    grad_accum_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    embedding_staging: wgpu::Buffer,
    /// Initial epochs_per_sample (CPU copy), used to reset the schedule counters.
    eps: Vec<f32>,
    n_vertices: u32,
    dim: u32,
    n_edges: usize,
    n_neighbors: usize,
    grad_clamp: f32,
    base_seed: u32,
    max_workgroups_per_dim: u32,
}

/// Async helper: map a buffer for reading and wait for it to be available.
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

/// Storage-buffer bind group layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

/// Uniform-buffer bind group layout entry.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl OptimizeGpuContext {
    /// Create a resident GPU context. Returns `None` if no suitable adapter is
    /// available, the buffers exceed device limits, or there are no edges (the
    /// caller then uses the CPU path, which is a no-op for an empty edge list).
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        embedding: &[f32],
        head: &[u32],
        tail: &[u32],
        epochs_per_sample: &[f32],
        n_vertices: u32,
        dim: u32,
        n_neighbors: usize,
        negative_sample_rate: f32,
        rng_state: [i64; 3],
    ) -> Option<Self> {
        let n_edges = head.len();
        if n_edges == 0 || n_vertices == 0 || dim == 0 {
            return None;
        }

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
        let buffer_limit =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);

        let embedding_bytes = (embedding.len() * 4) as u64;
        let grad_accum_bytes = (n_vertices as u64) * (dim as u64) * 4;
        let edge_bytes = (n_edges as u64) * 4;
        if embedding_bytes > buffer_limit
            || grad_accum_bytes > buffer_limit
            || edge_bytes > buffer_limit
        {
            return None;
        }

        // --- Buffers ---
        let embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding"),
            size: embedding_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&embedding_buffer, 0, bytemuck::cast_slice(embedding));

        let head_buffer = storage_buffer(&device, "head", edge_bytes);
        queue.write_buffer(&head_buffer, 0, bytemuck::cast_slice(head));
        let tail_buffer = storage_buffer(&device, "tail", edge_bytes);
        queue.write_buffer(&tail_buffer, 0, bytemuck::cast_slice(tail));
        let eps_buffer = storage_buffer(&device, "eps", edge_bytes);
        queue.write_buffer(&eps_buffer, 0, bytemuck::cast_slice(epochs_per_sample));

        let eons_buffer = storage_buffer(&device, "eons", edge_bytes);
        let eonns_buffer = storage_buffer(&device, "eonns", edge_bytes);

        let grad_accum_buffer = storage_buffer(&device, "grad_accum", grad_accum_bytes);
        queue.write_buffer(&grad_accum_buffer, 0, &vec![0u8; grad_accum_bytes as usize]);

        let velocity_buffer = storage_buffer(&device, "velocity", grad_accum_bytes);
        queue.write_buffer(&velocity_buffer, 0, &vec![0u8; grad_accum_bytes as usize]);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("optimize_params"),
            size: std::mem::size_of::<OptimizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let embedding_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_staging"),
            size: embedding_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Shader + pipelines (separate bind group layouts to stay within the
        //     8-storage-buffers-per-stage limit; accumulate peaks at 7). ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("optimize_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/optimize.wgsl").into()),
        });

        let accumulate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accumulate_bgl"),
            entries: &[
                storage_entry(0, false), // embedding rw
                storage_entry(1, true),  // head ro
                storage_entry(2, true),  // tail ro
                storage_entry(3, true),  // eps ro
                storage_entry(4, false), // eons rw
                storage_entry(5, false), // eonns rw
                storage_entry(6, false), // grad_accum rw
                uniform_entry(8),        // params
            ],
        });
        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apply_bgl"),
            entries: &[
                storage_entry(0, false), // embedding rw
                storage_entry(6, false), // grad_accum rw
                storage_entry(7, false), // velocity rw
                uniform_entry(8),        // params
            ],
        });
        let rebase_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rebase_bgl"),
            entries: &[
                storage_entry(4, false), // eons rw
                storage_entry(5, false), // eonns rw
                uniform_entry(8),        // params
            ],
        });

        let make_pipeline = |bgl: &wgpu::BindGroupLayout, entry: &str, label: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let accumulate_pipeline = make_pipeline(&accumulate_bgl, "accumulate_grads", "accumulate");
        let apply_pipeline = make_pipeline(&apply_bgl, "apply_grads", "apply");
        let rebase_pipeline = make_pipeline(&rebase_bgl, "rebase_schedule", "rebase");

        // --- Bind groups (static: buffers don't change identity). ---
        let accumulate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate_bg"),
            layout: &accumulate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: embedding_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: head_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tail_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: eps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: eons_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: eonns_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apply_bg"),
            layout: &apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: embedding_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let rebase_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rebase_bg"),
            layout: &rebase_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: eons_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: eonns_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let edge_clamp = 4.0f32;
        let grad_clamp = edge_clamp * (7.0 * n_neighbors as f32).sqrt();
        let base_seed = (rng_state[0] as u32)
            .wrapping_add(rng_state[1] as u32)
            .wrapping_add(rng_state[2] as u32);

        let ctx = OptimizeGpuContext {
            device,
            queue,
            accumulate_pipeline,
            apply_pipeline,
            rebase_pipeline,
            accumulate_bind_group,
            apply_bind_group,
            rebase_bind_group,
            embedding_buffer,
            head_buffer,
            tail_buffer,
            eps_buffer,
            eons_buffer,
            eonns_buffer,
            grad_accum_buffer,
            velocity_buffer,
            params_buffer,
            embedding_staging,
            eps: epochs_per_sample.to_vec(),
            n_vertices,
            dim,
            n_edges,
            n_neighbors,
            grad_clamp,
            base_seed,
            max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
        };

        // Initialize the schedule counters.
        ctx.reset_schedule(negative_sample_rate);

        Some(ctx)
    }

    /// Reset the per-edge schedule counters (eons = eps, eonns = eps/rate) and
    /// clear the gradient accumulator.
    pub fn reset_schedule(&self, negative_sample_rate: f32) {
        self.queue
            .write_buffer(&self.eons_buffer, 0, bytemuck::cast_slice(&self.eps));
        let eonns: Vec<f32> = if negative_sample_rate > 0.0 {
            self.eps.iter().map(|&e| e / negative_sample_rate).collect()
        } else {
            self.eps.clone()
        };
        self.queue
            .write_buffer(&self.eonns_buffer, 0, bytemuck::cast_slice(&eonns));
        self.zero_grad_accum();
    }

    /// Re-seed only the negative-sample counters (`eonns`) for a new rate,
    /// relative to `from_epoch` (the current `gpu_epoch`). Unlike
    /// [`reset_schedule`](Self::reset_schedule) this leaves the positive-sample
    /// counters and the gradient accumulator untouched, so it is safe to call
    /// mid-run when the live `negative_sample_rate` changes.
    pub fn reseed_negative_schedule(&self, negative_sample_rate: f32, from_epoch: u32) {
        let from = from_epoch as f32;
        let eonns: Vec<f32> = if negative_sample_rate > 0.0 {
            self.eps
                .iter()
                .map(|&e| from + e / negative_sample_rate)
                .collect()
        } else {
            // Parked: the shader's `rate > 0` guard skips reading these.
            vec![f32::INFINITY; self.eps.len()]
        };
        self.queue
            .write_buffer(&self.eonns_buffer, 0, bytemuck::cast_slice(&eonns));
    }

    /// Re-upload the embedding (used by `reinit_embedding`).
    pub fn reupload_embedding(&self, embedding: &[f32]) {
        self.queue
            .write_buffer(&self.embedding_buffer, 0, bytemuck::cast_slice(embedding));
    }

    /// Zero the velocity buffer (on optimizer switch / reinit).
    pub fn zero_velocity(&self) {
        let bytes = (self.n_vertices as usize) * (self.dim as usize) * 4;
        self.queue
            .write_buffer(&self.velocity_buffer, 0, &vec![0u8; bytes]);
    }

    fn zero_grad_accum(&self) {
        let bytes = (self.n_vertices as usize) * (self.dim as usize) * 4;
        self.queue
            .write_buffer(&self.grad_accum_buffer, 0, &vec![0u8; bytes]);
    }

    #[allow(clippy::too_many_arguments)]
    fn params_for(
        &self,
        epoch: u32,
        params: LiveParams,
        beta: f32,
        opt_flag: u32,
        apply_offset: u32,
        edge_offset: u32,
        edge_end: u32,
    ) -> OptimizeParams {
        OptimizeParams {
            n_vertices: self.n_vertices,
            dim: self.dim,
            n_edges: self.n_edges as u32,
            epoch,
            a: params.a,
            b: params.b,
            gamma: params.gamma,
            alpha: params.alpha,
            grad_clamp: self.grad_clamp,
            negative_sample_rate: params.negative_sample_rate,
            base_seed: self.base_seed,
            apply_offset,
            edge_offset,
            edge_end,
            optimizer: opt_flag,
            beta,
        }
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run one epoch on the GPU at the given (already-decayed) parameters.
    ///
    /// SGD splits the epoch into `n_sub` accumulate→apply rounds so later
    /// sub-batches see partially-updated positions (emulating CPU Hogwild and
    /// preventing full-batch overshoot). Momentum runs a single coherent round.
    pub fn run_epoch(&self, epoch: u32, params: LiveParams, beta: f32, optimizer: Optimizer) {
        if self.n_edges == 0 {
            return;
        }

        let (n_sub, opt_flag) = match optimizer {
            Optimizer::Sgd => {
                let per_vertex = (2.0 + params.negative_sample_rate) * self.n_neighbors as f32;
                ((per_vertex.sqrt().ceil() as usize).clamp(4, 32), 0u32)
            }
            Optimizer::Momentum => (1, 1u32),
        };

        let max_invocations = self.max_workgroups_per_dim * 256;
        let sub_size = self.n_edges.div_ceil(n_sub);

        for sub_start in (0..self.n_edges).step_by(sub_size) {
            let sub_end = (sub_start + sub_size).min(self.n_edges);

            // Accumulate [sub_start, sub_end), chunked to respect dispatch limits.
            let mut off = sub_start as u32;
            let end = sub_end as u32;
            while off < end {
                let count = (end - off).min(max_invocations);
                let p = self.params_for(epoch, params, beta, opt_flag, 0, off, off + count);
                self.queue
                    .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[p]));
                self.dispatch(
                    &self.accumulate_pipeline,
                    &self.accumulate_bind_group,
                    count.div_ceil(256),
                    "accumulate",
                );
                off += count;
            }

            // Apply all elements (chunked) and clear the accumulator.
            let total = self.n_vertices * self.dim;
            let mut base = 0u32;
            while base < total {
                let chunk = (total - base).min(max_invocations);
                let p = self.params_for(epoch, params, beta, opt_flag, base, 0, 0);
                self.queue
                    .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[p]));
                self.dispatch(
                    &self.apply_pipeline,
                    &self.apply_bind_group,
                    chunk.div_ceil(256),
                    "apply",
                );
                base += chunk;
            }
        }
    }

    /// Subtract `base` from both schedule-counter buffers (keeps the f32 epoch
    /// coordinate precise over indefinitely long runs).
    pub fn rebase(&self, base: u32) {
        if self.n_edges == 0 {
            return;
        }
        let max_invocations = self.max_workgroups_per_dim * 256;
        let total = self.n_edges as u32;
        let mut off = 0u32;
        while off < total {
            let count = (total - off).min(max_invocations);
            let mut p = self.params_for(
                base,
                LiveParams {
                    alpha: 0.0,
                    gamma: 1.0,
                    negative_sample_rate: 0.0,
                    a: 1.0,
                    b: 1.0,
                },
                0.0,
                0,
                0,
                off,
                total,
            );
            p.edge_offset = off;
            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[p]));
            self.dispatch(
                &self.rebase_pipeline,
                &self.rebase_bind_group,
                count.div_ceil(256),
                "rebase",
            );
            off += count;
        }
    }

    /// Download the embedding from the GPU into a fresh Vec.
    pub async fn download_embedding(&self) -> Vec<f32> {
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

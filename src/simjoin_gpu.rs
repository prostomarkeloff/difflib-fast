//! GPU (Metal) batched sparse-cosine — the offload experiment for [`crate::simjoin`].
//!
//! The CPU L2AP join is **memory-bandwidth-bound** on its exact-verify step: ~10⁸ candidate pairs,
//! each an `O(nnz)` sorted-merge dot that gathers two random CSR rows. This module tests whether the
//! Apple GPU — which sustains far more in-flight memory requests against the same unified-memory pool
//! — clears those dot-products faster. One GPU thread per pair walks the two rows and writes `cos`.
//!
//! **Parity caveat (important):** Metal has no `f64` (Apple GPUs are 32-bit float only), so the GPU
//! dot is computed in `f32` and is *not* bit-identical to the CPU `f64` `cos_full`. The GPU result is
//! therefore only usable as an **approximate, conservative filter** (CPU re-verifies survivors
//! exactly to preserve the byte-for-byte gate) — never as the emitted score. This module exists to
//! measure the throughput question; wiring it as a filter is gated on it actually being faster.
#![allow(clippy::cast_possible_truncation, clippy::doc_markdown, clippy::similar_names)]

use std::ffi::c_void;

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
    NSUInteger,
};

/// MSL: one thread per pair, sorted-merge dot of two CSR rows (f32). Rows are dim-ascending, so the
/// merge mirrors the CPU `cos_full`.
const KERNEL_SRC: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void batch_cosine(
    device const uint*  indptr [[buffer(0)]],
    device const uint*  dims   [[buffer(1)]],
    device const float* wts    [[buffer(2)]],
    device const uint*  pa     [[buffer(3)]],
    device const uint*  pb     [[buffer(4)]],
    device float*       out    [[buffer(5)]],
    constant uint&      npairs [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= npairs) return;
    uint a = pa[tid], b = pb[tid];
    uint ia = indptr[a], ea = indptr[a + 1];
    uint ib = indptr[b], eb = indptr[b + 1];
    float s = 0.0f;
    while (ia < ea && ib < eb) {
        uint da = dims[ia], db = dims[ib];
        if (da == db) { s += wts[ia] * wts[ib]; ia++; ib++; }
        else if (da < db) { ia++; }
        else { ib++; }
    }
    out[tid] = s;
}
";

/// A CSR corpus resident in unified memory + the compiled `batch_cosine` pipeline.
pub struct BatchCosineGpu {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    pipeline: ComputePipelineState,
    indptr: Buffer,
    dims: Buffer,
    wts: Buffer,
    n: usize,
}

// SAFETY: Metal device/queue/library/pipeline/buffers are documented thread-safe (see `gpu.rs`).
unsafe impl Send for BatchCosineGpu {}
unsafe impl Sync for BatchCosineGpu {}

impl BatchCosineGpu {
    /// Acquire the default Metal device, compile the kernel, and upload the CSR corpus (`indptr`
    /// length `n+1`, `dims`/`wts` length `nnz`) into UMA once. Returns `None` if no Metal device or
    /// the kernel fails to compile.
    #[must_use]
    pub fn new(indptr: &[u32], dims: &[u32], wts: &[f32]) -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(true);
        let library = match device.new_library_with_source(KERNEL_SRC, &options) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("simjoin_gpu: kernel compile failed: {e}");
                return None;
            }
        };
        let func = library.get_function("batch_cosine", None).ok()?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func).ok()?;
        let indptr_buf = upload(&device, indptr);
        let dims_buf = upload(&device, dims);
        let wts_buf = upload(&device, wts);
        Some(BatchCosineGpu {
            device,
            queue,
            _library: library,
            pipeline,
            indptr: indptr_buf,
            dims: dims_buf,
            wts: wts_buf,
            n: indptr.len().saturating_sub(1),
        })
    }

    /// Device name (e.g. "Apple M3 Pro").
    #[must_use]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Number of vectors in the resident corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if the resident corpus is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Compute `cos(row pa[t], row pb[t])` (f32 sparse dot) for every pair `t`, on the GPU.
    /// `pa` and `pb` must be equal length with all ids `< len()`.
    ///
    /// # Panics
    ///
    /// Panics if `pa.len() != pb.len()`.
    #[must_use]
    pub fn cosine_batch(&self, pa: &[u32], pb: &[u32]) -> Vec<f32> {
        assert_eq!(pa.len(), pb.len(), "pair arrays must match length");
        let np = pa.len();
        if np == 0 {
            return Vec::new();
        }
        let buf_pa = upload(&self.device, pa);
        let buf_pb = upload(&self.device, pb);
        let out_bytes = (np * std::mem::size_of::<f32>()) as NSUInteger;
        let buf_out = self.device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let np_u32 = np as u32;
        let buf_np = self.device.new_buffer_with_data(
            (&raw const np_u32).cast::<c_void>(),
            std::mem::size_of::<u32>() as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        for (i, b) in [&self.indptr, &self.dims, &self.wts, &buf_pa, &buf_pb, &buf_out, &buf_np]
            .into_iter()
            .enumerate()
        {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        let max_t = self.pipeline.max_total_threads_per_threadgroup() as usize;
        let tg = max_t.min(np).max(1);
        enc.dispatch_threads(MTLSize::new(np as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = buf_out.contents().cast::<f32>();
        // SAFETY: `buf_out` holds `np` f32s the kernel filled; it outlives this read.
        unsafe { std::slice::from_raw_parts(ptr, np).to_vec() }
    }
}

/// Copy `data` into a fresh shared (UMA) buffer — one memcpy into unified memory (zero device copy).
fn upload<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = (std::mem::size_of_val(data) as NSUInteger).max(1);
    device.new_buffer_with_data(data.as_ptr().cast::<c_void>(), bytes, MTLResourceOptions::StorageModeShared)
}

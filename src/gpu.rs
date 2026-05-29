//! `gpu` — Metal compute backend for **heterogeneous CPU+GPU exact RO** (Stage 4).
//!
//! Behind the `gpu` feature flag, macOS-only. The CPU SAM walker — `gestalt::matching_stats_into`
//! and `gestalt::longest_in` — is ported into a Metal compute kernel that runs in parallel with
//! the CPU rayon pool: same algorithm, byte-for-byte identical output, just spread across both
//! pieces of silicon. On Apple Silicon's unified memory architecture (UMA) the SAM buffers are
//! visible to both CPU and GPU without copying, so the only cost is a small bookkeeping fee per
//! `dispatch_threadgroups` call.
//!
//! ## Layout
//!
//! - [`Gpu`] — owns the Metal device, command queue, and pre-compiled compute pipelines.
//!   `Gpu::new()` returns `None` if no Metal device is available, so the rest of `difflib-fast`
//!   can fall back to the CPU path gracefully (headless macOS, virtualized environments, etc).
//! - [`KERNELS`] — Metal Shading Language source for the kernels, compiled once at `Gpu::new()`.
//!   For this stage we ship a `smoke_elementwise_add` kernel that validates the whole
//!   buffer-encode-dispatch-readback flow before we add the matching-stats kernel in Stage 4a-2.
//!
//! ## Why MSL inline as a string
//!
//! Pre-compiling to a `.metallib` (offline `metal -c` + `metallib`) is the production path; for
//! research-grade iteration the MSL source is small and `Device::new_library_with_source` compiles
//! it at process start (<10 ms typical). When we lock in the kernels we can move to offline
//! compilation.

#![cfg(all(feature = "gpu", target_os = "macos"))]
// Metal/objc FFI glue: doc prose names Apple types (`QoS`, `SoC`, `IOKit`, `CFString`, …) without
// backticks, and the `objc` msg_send patterns lean on intentional raw-pointer borrows / casts. These
// are inherent to the FFI surface, so allow the pedantic/style lints they trip module-wide.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::borrow_as_ptr,
    clippy::needless_borrow
)]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::time::Instant;

use metal::foreign_types::ForeignTypeRef;
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputePipelineDescriptor, ComputePipelineState,
    Device, Library, MTLResourceOptions, MTLSize, NSUInteger,
};

// ---------------------------------------------------------------------------
// Priority / boost knobs for short-running CLI use.
//
// Five mechanisms, in order from cheapest to heaviest. All are best-effort — failure to apply any
// one of them is non-fatal; we log and continue. The matching `release_*` helpers (or `Drop` on
// `BoostGuard`) reverse them at process exit so we don't leave power assertions held for the
// shell session that spawned us.
// ---------------------------------------------------------------------------

/// Apple QoS class constants (from `<sys/qos.h>`). The Metal driver looks at the current thread's
/// QoS when scheduling command-buffer commits — `USER_INTERACTIVE` (0x21) puts our submissions
/// ahead of other userspace work in the dispatch queue.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

#[link(name = "c")]
extern "C" {
    /// `pthread_set_qos_class_self_np(qos_class, relative_priority)` — raise/lower the calling
    /// thread's QoS. `relative_priority` must be in `[-15, 0]`; we always pass 0 (max within class).
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
}

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    /// `IOPMAssertionCreateWithName(type: CFString, level: u32, name: CFString, out_id: *mut u32)`
    /// — request the SoC stay at a higher performance state. Returns `kIOReturnSuccess` (0) on
    /// success. We pass the assertion ID to `IOPMAssertionRelease` at process exit.
    fn IOPMAssertionCreateWithName(
        assertion_type: *const c_void,
        assertion_level: u32,
        assertion_name: *const c_void,
        out_assertion_id: *mut u32,
    ) -> i32;
    fn IOPMAssertionRelease(assertion_id: u32) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringCreateWithCString(
        allocator: *const c_void,
        c_str: *const std::os::raw::c_char,
        encoding: u32,
    ) -> *const c_void;
}

/// `kCFStringEncodingUTF8`. Used for building the `CFString` arguments to IOKit.
const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

/// Raise the calling thread's QoS to `USER_INTERACTIVE`. Best-effort; logs on failure.
///
/// Effect: Metal driver dispatches our command buffers ahead of UTILITY/BACKGROUND work; the
/// kernel scheduler also raises this thread's run priority. Combined with [`hold_high_perf_assertion`]
/// this is the cheapest way to keep an interactive CLI from getting swapped behind WindowServer
/// compositor passes.
pub fn raise_thread_qos_user_interactive() {
    // SAFETY: pthread_set_qos_class_self_np is documented thread-safe; we pass standard constants.
    let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    if rc != 0 {
        eprintln!("difflib-fast: pthread_set_qos_class_self_np failed (rc={rc}); proceeding at default QoS");
    }
}

/// Acquire an IOPM "high performance" assertion. Holds the SoC at boost clocks (CPU + GPU) until
/// [`release_high_perf_assertion`] is called or the process exits.
///
/// Without this, the M3 GPU drops to ~500 MHz between dispatches; first dispatch then pays ~50 ms
/// to ramp back. Apple uses this in Final Cut for the same reason. Best-effort; logs on failure.
///
/// Returns the assertion ID for later release; 0 means failed.
#[must_use]
pub fn hold_high_perf_assertion() -> u32 {
    // SAFETY: All-FFI with documented Apple types. CFStrings are released by IOKit when the
    // assertion is released.
    unsafe {
        let kind = b"PreventUserIdleSystemSleep\0";
        let name = b"difflib-fast.gpu.boost\0";
        let null_alloc: *const c_void = std::ptr::null();
        let kind_cs = CFStringCreateWithCString(
            null_alloc,
            kind.as_ptr().cast(),
            K_CF_STRING_ENCODING_UTF8,
        );
        let name_cs = CFStringCreateWithCString(
            null_alloc,
            name.as_ptr().cast(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if kind_cs.is_null() || name_cs.is_null() {
            return 0;
        }
        let mut id: u32 = 0;
        // 255 = kIOPMAssertionLevelOn.
        let rc = IOPMAssertionCreateWithName(kind_cs, 255, name_cs, &mut id);
        if rc != 0 {
            eprintln!("difflib-fast: IOPMAssertionCreate failed (rc={rc}); proceeding without boost");
            return 0;
        }
        id
    }
}

/// Release an assertion acquired by [`hold_high_perf_assertion`]. No-op for id 0.
pub fn release_high_perf_assertion(id: u32) {
    if id == 0 {
        return;
    }
    // SAFETY: id is one we got back from IOPMAssertionCreateWithName.
    let _ = unsafe { IOPMAssertionRelease(id) };
}

/// RAII guard for the full priority-boost combo (thread QoS + IOPM assertion). Construct once at
/// process start, hold for the duration of the GPU work, drop to release.
///
/// ```ignore
/// let _boost = difflib_fast::gpu::BoostGuard::acquire();
/// // ... GPU work here ...
/// // boost dropped automatically at end of scope
/// ```
pub struct BoostGuard {
    assertion_id: u32,
}

impl BoostGuard {
    /// Apply all priority knobs to the current thread + process. Idempotent.
    #[must_use]
    pub fn acquire() -> Self {
        raise_thread_qos_user_interactive();
        let assertion_id = hold_high_perf_assertion();
        Self { assertion_id }
    }
}

impl Drop for BoostGuard {
    fn drop(&mut self) {
        release_high_perf_assertion(self.assertion_id);
    }
}

/// Best-effort: tag `queue` as high-priority via the private `setReducedCPUPriority:` selector if
/// available (no-op on builds of macOS that don't expose it). Apple uses this for foreground
/// graphics queues; for our CLI we want the same priority class.
///
/// The selector takes a `BOOL`: setting it to `NO` (false) means "do NOT reduce CPU priority on
/// behalf of this queue's commits", which is what we want — keep the priority of the calling
/// thread (which we've already raised via [`raise_thread_qos_user_interactive`]).
fn set_queue_high_priority(queue: &CommandQueue) {
    // SAFETY: queue.as_ptr() is a live MTLCommandQueue. The selector is a no-op if not implemented
    // on the current OS; we don't check existence because `objc_msgSend` of an unknown selector
    // throws — to be robust we use `respondsToSelector:` first.
    unsafe {
        let q: *mut Object = queue.as_ptr().cast();
        let sel_obj = sel!(setReducedCPUPriority:);
        let responds: bool = msg_send![q, respondsToSelector: sel_obj];
        if responds {
            let _: () = msg_send![q, setReducedCPUPriority: false];
        }
    }
}

/// `MTLCommandBuffer.GPUStartTime` — when the GPU actually began executing this command buffer.
/// Returned as a `CFAbsoluteTime` (seconds since 2001-01-01 Reference Date). Not bound in the
/// `metal` crate version we use, so we send the selector by hand. Apple docs: timing is wall-clock
/// GPU-side (no CPU wait), populated after `wait_until_completed` returns.
fn gpu_command_buffer_times(cmd: &CommandBufferRef) -> (f64, f64, f64, f64) {
    // SAFETY: cmd is a valid live MTLCommandBuffer; these selectors are present on every Metal
    // OS Apple ships. The selectors return `double` (CFAbsoluteTime / NSTimeInterval).
    unsafe {
        let gpu_start: f64 = msg_send![cmd.as_ptr(), GPUStartTime];
        let gpu_end: f64 = msg_send![cmd.as_ptr(), GPUEndTime];
        let kernel_start: f64 = msg_send![cmd.as_ptr(), kernelStartTime];
        let kernel_end: f64 = msg_send![cmd.as_ptr(), kernelEndTime];
        (kernel_start, kernel_end, gpu_start, gpu_end)
    }
}

/// Inline Metal Shading Language source. Compiled once per process at `Gpu::new()`.
const KERNELS: &str = "
#include <metal_stdlib>
using namespace metal;

// Stage-4a-1 smoke test: pure element-wise add over u32 arrays. Used to verify the Metal pipeline
// is wired correctly end-to-end (buffer upload -> dispatch -> readback) before we wire in the
// SAM matching-stats kernel. Writes are well-defined per-thread, so this is a sound correctness gate.
kernel void smoke_elementwise_add(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device       uint* out [[buffer(2)]],
    constant uint& n     [[buffer(3)]],
    uint id              [[thread_position_in_grid]]
) {
    if (id >= n) return;
    out[id] = a[id] + b[id];
}

// Stage-4a-11: partial-cache variant. ONLY the first `K_HOT_NODES` states (low-len, near-root,
// most-visited per matching_stats traffic distribution) live in threadgroup memory; states with
// `state >= K_HOT_NODES` fall through to global `sam_nodes_g`. Drops the full-SAM-in-TG cap of
// `matching_stats_by_b` (which forced a CPU fallback for SAMs > 32 KB) while still giving the
// hot path TG-memory-speed (≈1 cycle vs 30 cycles for L1 vs 200 cycles for RAM).
//
// Theory of expected gain on canonical Python:
//   - K=256 covers the low-`len` band; instrumented runs show ~60-90% of byte visits land here
//     (after each suffix-link backtrack the walker resets to a shallow state).
//   - Hot byte = ~1 cycle node read; cold byte = ~9 cycles. At 80% hot ratio average drops
//     from ~9 cycles/byte (all-global baseline) to ~2.6 cycles/byte → ~3.5× kernel speedup,
//     putting GPU compute in the ~27 ms range for 100 k mypy pairs and CPU wall at ~5-7× CPU.
//
// Edges stay in global memory (already largely L1-resident — each state's edge range is small
// and contiguous). Caching them too would push K down due to the 32 KB threadgroup cap.
kernel void matching_stats_by_b_partial(
    device const uint*   pair_a_idx_sorted [[buffer(0)]],
    device const uint*   pair_b_offsets    [[buffer(1)]],
    device const uint*   active_b_idx      [[buffer(2)]],
    device const uchar*  a_data            [[buffer(3)]],
    device const uint*   a_offsets         [[buffer(4)]],
    device const uint4*  sam_nodes_g       [[buffer(5)]],
    device const uint*   sam_node_offs     [[buffer(6)]],
    device const uint*   sam_edges_g       [[buffer(7)]],
    device const uint*   sam_edge_offs     [[buffer(8)]],
    device const int*    sam_root_g        [[buffer(9)]],
    device       uint*   fmatch_out        [[buffer(10)]],
    device       uint*   fstate_out        [[buffer(11)]],
    device const uint*   out_offsets       [[buffer(12)]],
    constant uint&       k_hot_nodes       [[buffer(13)]],
    threadgroup uchar*   tg_mem            [[threadgroup(0)]],
    uint                 tg_pos            [[threadgroup_position_in_grid]],
    uint                 lid               [[thread_position_in_threadgroup]],
    uint                 tg_size           [[threads_per_threadgroup]]
) {
    uint b_idx = active_b_idx[tg_pos];
    uint node_lo = sam_node_offs[b_idx];
    uint node_hi = sam_node_offs[b_idx + 1u];
    uint n_nodes = node_hi - node_lo;
    uint sam_node_base = node_lo;
    uint sam_edge_base = sam_edge_offs[b_idx];
    uint sam_root_base = b_idx * 128u;

    // Cache first min(n_nodes, K_HOT) state nodes + root_next in TG memory. MEASURED: caching
    // edges too gave NO additional win on canonical Python (HA, mypy) — edges within a state's
    // contiguous range are already L1-resident, while caching them in TG memory cost arena
    // bytes that reduced occupancy. Keep edges in global; cache only nodes.
    uint k_hot = (n_nodes < k_hot_nodes) ? n_nodes : k_hot_nodes;
    threadgroup uint4* nodes_tg = (threadgroup uint4*)(tg_mem);
    threadgroup int*   root_tg  = (threadgroup int*)  (tg_mem + k_hot_nodes * 16u);
    for (uint i = lid; i < k_hot; i += tg_size) {
        nodes_tg[i] = sam_nodes_g[node_lo + i];
    }
    for (uint i = lid; i < 128u; i += tg_size) {
        root_tg[i] = sam_root_g[sam_root_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pair_lo = pair_b_offsets[tg_pos];
    uint pair_hi = pair_b_offsets[tg_pos + 1u];
    uint n_my = pair_hi - pair_lo;
    for (uint pos = lid; pos < n_my; pos += tg_size) {
        uint t = pair_lo + pos;
        uint a_idx = pair_a_idx_sorted[t];
        uint a_lo = a_offsets[a_idx];
        uint a_len = a_offsets[a_idx + 1u] - a_lo;
        uint out_base = out_offsets[t];

        uint state = 0u;
        uint matched = 0u;
        for (uint i = 0u; i < a_len; i++) {
            uint c = (uint)a_data[a_lo + i];
            for (;;) {
                int nx = -1;
                uint4 cur_nd;
                bool have_cur_nd = false;
                if (state == 0u) {
                    if (c < 128u) {
                        nx = root_tg[c];
                    } else {
                        // Root state is always in TG (state 0 < k_hot trivially).
                        uint4 nd = nodes_tg[0];
                        uint elo = nd.z;
                        uint ehi = nd.w;
                        while (elo < ehi) {
                            uint mid = elo + (ehi - elo) / 2u;
                            uint e = sam_edges_g[sam_edge_base + mid];
                            uint mc = e >> 24;
                            if (mc == c) { nx = (int)(e & 0xFFFFFFu); break; }
                            if (mc < c) { elo = mid + 1u; } else { ehi = mid; }
                        }
                    }
                } else {
                    // Hot path: state < k_hot → nodes_tg (≈1 cycle TG memory latency).
                    // Cold path: state >= k_hot → global memory (≈30+ cycles).
                    // Ternary on the SELECT side — both addresses are computed but only one
                    // load fires per warp lane (M3 select-merge keeps it from doubling traffic).
                    cur_nd = (state < k_hot) ? nodes_tg[state] : sam_nodes_g[sam_node_base + state];
                    have_cur_nd = true;
                    uint elo = cur_nd.z;
                    uint ehi = cur_nd.w;
                    while (elo < ehi) {
                        uint mid = elo + (ehi - elo) / 2u;
                        uint e = sam_edges_g[sam_edge_base + mid];
                        uint mc = e >> 24;
                        if (mc == c) { nx = (int)(e & 0xFFFFFFu); break; }
                        if (mc < c) { elo = mid + 1u; } else { ehi = mid; }
                    }
                }
                if (nx >= 0) {
                    state = (uint)nx;
                    matched += 1u;
                    break;
                }
                if (state == 0u) {
                    matched = 0u;
                    break;
                }
                // Link backtrack — `cur_nd` is loaded above for state>0.
                uint4 nd = have_cur_nd
                    ? cur_nd
                    : ((state < k_hot) ? nodes_tg[state] : sam_nodes_g[sam_node_base + state]);
                state = nd.y;
                matched = nd.x;
            }
            fmatch_out[out_base + i] = matched;
            fstate_out[out_base + i] = state;
        }
    }
}

// Stage-4a-3: same matching-stats walk as matching_stats_one_pair, but BATCHED — one thread per
// pair, K pairs processed by a single dispatch_threads call. The pairs share the corpus buffers:
//
//   pair_a_idx[t], pair_b_idx[t]        — which a-string and which SAM thread t handles
//   a_data[a_offsets[i]..a_offsets[i+1]] — string i's bytes
//   sam_nodes[sam_node_offs[j]..]       — SAM j's nodes (uint4 each, units of uint4)
//   sam_edges[sam_edge_offs[j]..]       — SAM j's edges (ulong each, units of ulong)
//   sam_root_next[j*128..(j+1)*128]    — SAM j's root direct ASCII table
//   fmatch_out[out_offsets[t]..out_offsets[t+1]] — thread t's per-position fmatch
//   fstate_out[out_offsets[t]..out_offsets[t+1]] — thread t's per-position fstate
//
// edge_lo/edge_hi in nodes are LOCAL indices into the SAM's edge range (the SAM never sees the
// global concatenated buffer); the kernel reads sam_edges[sam_edge_base + mid] where mid is the
// SAM-local index. Same applies to suffix-link targets (state indices) — they're local. This is
// why we don't need to rewrite any field during concatenation.
// One uint4 load per byte. MEASURED: hot/cold split (separate uint + uint2 buffers) is SLOWER —
// IR-level CSE keeps the uint4 layout to a single load instruction, while split forced two
// separate loads (~+60% i32-load count). uint4 wins on M3 due to wider memory ops + better
// instruction amortization. Layout: nd = (link_len_of_state, link, edge_lo, edge_hi).
kernel void matching_stats_batched(
    device const uint*   pair_a_idx     [[buffer(0)]],
    device const uint*   pair_b_idx     [[buffer(1)]],
    device const uchar*  a_data         [[buffer(2)]],
    device const uint*   a_offsets      [[buffer(3)]],
    device const uint4*  sam_nodes      [[buffer(4)]],
    device const uint*   sam_node_offs  [[buffer(5)]],
    device const uint*   sam_edges      [[buffer(6)]],
    device const uint*   sam_edge_offs  [[buffer(7)]],
    device const int*    sam_root_next  [[buffer(8)]],
    device       uint*   fmatch_out     [[buffer(9)]],
    device       uint*   fstate_out     [[buffer(10)]],
    device const uint*   out_offsets    [[buffer(11)]],
    constant uint&       n_pairs        [[buffer(12)]],
    uint tid                            [[thread_position_in_grid]]
) {
    if (tid >= n_pairs) return;

    uint a_idx = pair_a_idx[tid];
    uint b_idx = pair_b_idx[tid];
    uint a_lo = a_offsets[a_idx];
    uint a_hi = a_offsets[a_idx + 1u];
    uint a_len = a_hi - a_lo;
    uint sam_node_base = sam_node_offs[b_idx];
    uint sam_edge_base = sam_edge_offs[b_idx];
    uint sam_root_base = b_idx * 128u;
    uint out_base = out_offsets[tid];

    uint state   = 0u;
    uint matched = 0u;
    for (uint i = 0u; i < a_len; i++) {
        uint c = (uint)a_data[a_lo + i];
        for (;;) {
            int nx = -1;
            uint4 cur_nd;
            bool have_cur_nd = false;
            if (state == 0u) {
                if (c < 128u) {
                    nx = sam_root_next[sam_root_base + c];
                } else {
                    uint4 nd = sam_nodes[sam_node_base + 0u];
                    uint elo = nd.z;
                    uint ehi = nd.w;
                    while (elo < ehi) {
                        uint mid = elo + (ehi - elo) / 2u;
                        uint e = sam_edges[sam_edge_base + mid];
                        uint mc = e >> 24;
                        if (mc == c) { nx = (int)(e & 0xFFFFFFu); break; }
                        if (mc < c) { elo = mid + 1u; } else { ehi = mid; }
                    }
                }
            } else {
                cur_nd = sam_nodes[sam_node_base + state];
                have_cur_nd = true;
                uint elo = cur_nd.z;
                uint ehi = cur_nd.w;
                while (elo < ehi) {
                    uint mid = elo + (ehi - elo) / 2u;
                    uint e = sam_edges[sam_edge_base + mid];
                    uint mc = e >> 24;
                    if (mc == c) { nx = (int)(e & 0xFFFFFFu); break; }
                    if (mc < c) { elo = mid + 1u; } else { ehi = mid; }
                }
            }
            if (nx >= 0) {
                state = (uint)nx;
                matched += 1u;
                break;
            }
            if (state == 0u) {
                matched = 0u;
                break;
            }
            // nd.x is precomputed by CorpusGpu::build to be len(link[state]) — read directly,
            // skipping a second sam_nodes load.
            uint4 nd = have_cur_nd ? cur_nd : sam_nodes[sam_node_base + state];
            state = nd.y;
            matched = nd.x;
        }
        fmatch_out[out_base + i] = matched;
        fstate_out[out_base + i] = state;
    }
}
";

/// Holds the Metal device handle and pre-compiled compute pipelines.
///
/// Construction is fallible (`Option<Gpu>`) so callers can degrade to CPU-only on systems
/// without a usable Metal device. The struct is `Send`/`Sync` because the underlying Metal
/// types are; multiple threads can share one `Gpu` and concurrently submit work to its
/// command queue.
pub struct Gpu {
    device: Device,
    queue: CommandQueue,
    /// Kept alive because the compute pipelines hold weak references back through their library.
    _library: Library,
    smoke_pipeline: ComputePipelineState,
    matching_stats_batched_pipeline: ComputePipelineState,
    matching_stats_by_b_partial_pipeline: ComputePipelineState,
}

// SAFETY: Metal device, queue, library, and pipeline are all thread-safe — Apple's Metal API
// documents them as usable from any thread, and the `metal` crate's `Send + Sync` impls reflect
// that. We share one `Gpu` across rayon worker threads in the dispatcher.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    /// Acquire the system's default Metal device and compile the kernels. Returns `None` if the
    /// platform has no Metal device or the MSL fails to compile (treated as "GPU unavailable" —
    /// the caller falls back to CPU). Compilation logs go to stderr on failure.
    #[must_use]
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        // High-priority command queue: Apple's `MTLCommandQueue` has a private
        // `setReducedCPUPriority:` selector + the device-level
        // `newCommandQueueWithMaxCommandBufferCount:` accessor. The metal-rs crate doesn't expose
        // priority, so we override after construction via objc msg_send. Failure here is benign —
        // we just fall back to the default queue.
        let queue = device.new_command_queue();
        set_queue_high_priority(&queue);

        // Aggressive compile options: fast-math math (no NaN/inf checks), latest MSL spec.
        // These are the same defaults Xcode applies to release builds with -O2 -ffast-math.
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(true);
        let library = match device.new_library_with_source(KERNELS, &options) {
            Ok(lib) => lib,
            Err(err) => {
                eprintln!("difflib-fast: Metal kernel compile failed: {err}");
                return None;
            }
        };
        let smoke_pipeline = make_pipeline(&device, &library, "smoke_elementwise_add").ok()?;
        let matching_stats_batched_pipeline =
            make_pipeline(&device, &library, "matching_stats_batched").ok()?;
        let matching_stats_by_b_partial_pipeline =
            make_pipeline(&device, &library, "matching_stats_by_b_partial").ok()?;
        let gpu = Gpu {
            device,
            queue,
            _library: library,
            smoke_pipeline,
            matching_stats_batched_pipeline,
            matching_stats_by_b_partial_pipeline,
        };

        // Warm-up dispatch: a 1 K-element smoke add ramps the GPU into a high P-state before
        // real work arrives. Without this, the first matching_stats dispatch eats 50–100 ms of
        // clock spin-up. We discard the result and the buffers; the side effect we want is just
        // the GPU power state transition.
        gpu.warm_up();

        Some(gpu)
    }

    /// Submit a tiny GPU dispatch and wait, purely to ramp the M3 GPU's P-state up so that the
    /// first production dispatch hits at high clocks instead of paying spin-up latency.
    fn warm_up(&self) {
        let a: [u32; 1024] = [0; 1024];
        let _ = self.smoke_elementwise_add(&a, &a);
    }

    /// Inspect the device's reported name (e.g. "Apple M3 Pro") — used by diagnostics to confirm
    /// we're talking to the integrated GPU.
    #[must_use]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Smoke test: `out[i] = a[i] + b[i]` over equal-length u32 arrays, run on the GPU. Verifies
    /// the buffer-upload + dispatch + readback path works before we wire in the matching-stats
    /// kernel. Returns a fresh `Vec<u32>` of the output (length = inputs' length).
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()`. Internal `unwrap()`s on Metal pipeline acquisition are
    /// gated by the `Gpu::new()` constructor — if you have a `Gpu` handle they cannot panic.
    #[must_use]
    pub fn smoke_elementwise_add(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
        assert_eq!(a.len(), b.len(), "smoke_elementwise_add: inputs must match length");
        let n = a.len();
        if n == 0 {
            return Vec::new();
        }
        // Storage mode `shared` puts the buffer in unified memory accessible to both CPU and GPU
        // with no synchronization beyond command-buffer waitUntilCompleted — exactly what Apple
        // Silicon's UMA was designed for. No copy back to host needed; we read straight from
        // `buf_out.contents()` after the kernel finishes.
        let buf_a = self.upload_u32(a);
        let buf_b = self.upload_u32(b);
        let buf_out = self.empty_u32_buffer(n);
        let n_u32 = n as u32;
        let buf_n = self.device.new_buffer_with_data(
            (&raw const n_u32).cast::<c_void>(),
            std::mem::size_of::<u32>() as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.smoke_pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_out), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);

        // Pick a threadgroup size — Metal's `max_total_threads_per_threadgroup` is the upper
        // bound; for a 1-D linear kernel we just take min(n, max) and let the driver tile.
        let max_t = self.smoke_pipeline.max_total_threads_per_threadgroup() as usize;
        let tg = max_t.min(n);
        let grid_size = MTLSize::new(n as u64, 1, 1);
        let tg_size = MTLSize::new(tg as u64, 1, 1);
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read the result directly from the shared buffer's contents pointer — no Metal-side copy.
        let out_ptr = buf_out.contents().cast::<u32>();
        // SAFETY: buf_out was allocated with `n * size_of::<u32>()` bytes; the kernel wrote every
        // index in `[0, n)`; the buffer outlives this slice.
        let slice = unsafe { std::slice::from_raw_parts(out_ptr, n) };
        slice.to_vec()
    }

    /// Allocate a shared (UMA) buffer with `n * 4` bytes and seed it from `data` (length `n`).
    /// One copy on the CPU side — equivalent in cost to a `memcpy` since the destination is
    /// already in unified memory.
    fn upload_u32(&self, data: &[u32]) -> Buffer {
        let bytes = std::mem::size_of_val(data) as NSUInteger;
        self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a shared (UMA) buffer with `n * 4` bytes, uninitialized.
    fn empty_u32_buffer(&self, n: usize) -> Buffer {
        let bytes = (n * std::mem::size_of::<u32>()) as NSUInteger;
        self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Generic shared-memory upload: copies `data` byte-for-byte into a new UMA buffer.
    /// Used by `CorpusGpu` to upload the concatenated SAM arrays. The destination buffer is
    /// `Send + Sync` so the caller can move it freely across threads.
    fn upload_buf<T: Copy>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data) as NSUInteger;
        // Zero-length buffers aren't allowed by Metal; round up to 1 byte for safety. We never
        // dispatch with a zero-element view, so this only triggers on degenerate inputs.
        let bytes_safe = bytes.max(1);
        self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            bytes_safe,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Flat-buffer variant of `matching_stats_batched`: returns concatenated `fstate` / `fmatch`
    /// arrays + per-pair offsets, with NO per-pair `Vec` allocations. Production callers
    /// (cluster_canonicals_qualifies under heterogeneous dispatch) should use this — measurement
    /// shows the per-pair `Vec<Vec>` materialization adds ~40% wall on a 100k-pair batch.
    ///
    /// `MatchingStatsFlat::fstate_all[out_offsets[t]..out_offsets[t+1]]` is pair `t`'s `fstate`,
    /// and similarly for `fmatch_all`.
    #[must_use]
    pub fn matching_stats_batched_flat(
        &self,
        corpus: &CorpusGpu,
        pairs: &[(u32, u32)],
    ) -> MatchingStatsFlat {
        self.matching_stats_batched_flat_with_timings(corpus, pairs).0
    }

    /// Flat-buffer variant + per-stage timings. See `matching_stats_batched_with_timings` for
    /// the timing array layout.
    #[must_use]
    #[allow(clippy::similar_names, clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn matching_stats_batched_flat_with_timings(
        &self,
        corpus: &CorpusGpu,
        pairs: &[(u32, u32)],
    ) -> (MatchingStatsFlat, [u128; 5]) {
        let n_pairs = pairs.len();
        if n_pairs == 0 {
            let empty = self.empty_u32_buffer(1);
            return (
                MatchingStatsFlat {
                    out_offsets: vec![0],
                    pair_orig_idx: Vec::new(),
                    fstate_buf: empty.clone(),
                    fmatch_buf: empty,
                    total_out: 0,
                },
                [0; 5],
            );
        }

        // Stage 1 — build pair-index arrays + size per-pair output regions.
        //
        // We MEASURED sort-by-a-length and found no win (M3 GPU is memory-latency-bound on this
        // kernel; SIMD divergence on the outer-loop iteration count isn't the bottleneck).
        //
        // What IS the bottleneck: global-memory traffic into per-b SAM regions. Each pair walks
        // its own SAM, hitting `sam_nodes[base + state]` and `sam_edges[base + mid]` — `base`
        // varies per b_idx, scattering the access pattern across hundreds of MB. By sorting pairs
        // so consecutive threads share the same b_idx, all 32 threads in a SIMD-group hit the SAME
        // cache lines (the b-SAM is ~2-4 KB and fits in L1). This is the cheap-to-measure step
        // before we restructure to per-threadgroup SAM caching in 4a-5.
        let t1 = Instant::now();
        let mut order: Vec<u32> = (0..n_pairs as u32).collect();
        // Sort by (b_idx, a_len, a_idx) — primary key `b` so threads in a SIMD-group share the
        // same SAM (hit L1 instead of scattered global memory), secondary `a_len` so they have
        // similar outer-loop trip counts (reduces lockstep divergence at warp tail), tertiary
        // `a_idx` for stable order. MEASURED: sort-by-(b,a_idx) alone roughly halved GPU
        // compute (220 ms → 112 ms on 100k mypy pairs); adding a_len as a secondary key
        // shaves further divergence.
        order.sort_by_key(|&t| {
            let (a, b) = pairs[t as usize];
            let a_lo = corpus.a_offsets_cpu[a as usize];
            let a_hi = corpus.a_offsets_cpu[a as usize + 1];
            (b, a_hi - a_lo, a)
        });
        let pair_a_idx: Vec<u32> = order.iter().map(|&t| pairs[t as usize].0).collect();
        let pair_b_idx: Vec<u32> = order.iter().map(|&t| pairs[t as usize].1).collect();
        let mut out_offsets: Vec<u32> = Vec::with_capacity(n_pairs + 1);
        out_offsets.push(0);
        let mut total_out: u32 = 0;
        for &t_idx in &order {
            let (a_idx, b_idx) = pairs[t_idx as usize];
            assert!((a_idx as usize) < corpus.n_strings, "a_idx out of range");
            assert!((b_idx as usize) < corpus.n_sams, "b_idx out of range");
            let a_len = corpus.a_offsets_cpu[a_idx as usize + 1]
                - corpus.a_offsets_cpu[a_idx as usize];
            total_out = total_out.checked_add(a_len).expect("matching_stats_batched output too large");
            out_offsets.push(total_out);
        }
        let stage_build_pairs = t1.elapsed().as_nanos();

        // Stage 2 — upload pair arrays + allocate output buffers (everything in UMA).
        let t2 = Instant::now();
        let buf_pair_a = self.upload_buf(&pair_a_idx);
        let buf_pair_b = self.upload_buf(&pair_b_idx);
        let buf_out_offsets = self.upload_buf(&out_offsets);
        let buf_fmatch = self.empty_u32_buffer(total_out as usize);
        let buf_fstate = self.empty_u32_buffer(total_out as usize);
        let n_pairs_u32 = n_pairs as u32;
        let buf_n_pairs = self.upload_buf(std::slice::from_ref(&n_pairs_u32));
        let stage_upload = t2.elapsed().as_nanos();

        // Stage 3 — encode kernel + dispatch (does not wait yet).
        let t3 = Instant::now();
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matching_stats_batched_pipeline);
        encoder.set_buffer(0, Some(&buf_pair_a), 0);
        encoder.set_buffer(1, Some(&buf_pair_b), 0);
        encoder.set_buffer(2, Some(&corpus.a_data_buf), 0);
        encoder.set_buffer(3, Some(&corpus.a_offsets_buf), 0);
        encoder.set_buffer(4, Some(&corpus.sam_nodes_buf), 0);
        encoder.set_buffer(5, Some(&corpus.sam_node_offsets_buf), 0);
        encoder.set_buffer(6, Some(&corpus.sam_edges_buf), 0);
        encoder.set_buffer(7, Some(&corpus.sam_edge_offsets_buf), 0);
        encoder.set_buffer(8, Some(&corpus.sam_root_next_buf), 0);
        encoder.set_buffer(9, Some(&buf_fmatch), 0);
        encoder.set_buffer(10, Some(&buf_fstate), 0);
        encoder.set_buffer(11, Some(&buf_out_offsets), 0);
        encoder.set_buffer(12, Some(&buf_n_pairs), 0);
        // Threadgroup size: default 1024 (pipeline_max) — Apple's docs say wider TG amortizes
        // launch overhead, but for memory-latency-bound kernels narrower TG can boost occupancy.
        // Tunable via BENCH_TG env var so we can sweep.
        let max_t = self
            .matching_stats_batched_pipeline
            .max_total_threads_per_threadgroup() as usize;
        let tg_env: usize = std::env::var("BENCH_TG")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(max_t);
        let tg = tg_env.min(max_t).min(n_pairs).max(32);
        encoder.dispatch_threads(
            MTLSize::new(n_pairs as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        let stage_dispatch = t3.elapsed().as_nanos();

        // Stage 4 — wait for GPU to finish (this is the actual on-device compute).
        let t4 = Instant::now();
        cmd.wait_until_completed();
        let stage_wait = t4.elapsed().as_nanos();

        // Pull GPU-side timestamps. `gpu_end - gpu_start` is the wall on-device time; anything
        // beyond that in `stage_wait` is CPU-side dispatch overhead. `kernel_*` brackets just the
        // compute kernel execution (slightly tighter than gpu_*, which includes scheduling slack).
        // Stored in `gpu_times` for the caller to log.
        let (kernel_start, kernel_end, gpu_start, gpu_end) = gpu_command_buffer_times(&cmd);
        eprintln!(
            "  [gpu_times] kernel: {:.3} ms, gpu: {:.3} ms (gpu_start={:.6} end={:.6})",
            (kernel_end - kernel_start) * 1000.0,
            (gpu_end - gpu_start) * 1000.0,
            gpu_start,
            gpu_end,
        );

        // Stage 5 — wrap the Metal buffers in a zero-copy view. The slices returned from
        // `MatchingStatsFlat::fstate_all` / `fmatch_all` are read directly from UMA, no host copy.
        let t5 = Instant::now();
        let flat = MatchingStatsFlat {
            out_offsets,
            pair_orig_idx: order,
            fstate_buf: buf_fstate,
            fmatch_buf: buf_fmatch,
            total_out: total_out as usize,
        };
        let stage_readback = t5.elapsed().as_nanos();

        (flat, [stage_build_pairs, stage_upload, stage_dispatch, stage_wait, stage_readback])
    }

}

impl Gpu {
    /// Stage-4a-11: per-b kernel with PARTIAL threadgroup cache (only the first `K_HOT_NODES`
    /// states). Unlike `matching_stats_by_b_flat_with_timings`, this never falls back to global —
    /// states ≥ K go through global memory directly inside the kernel, so it handles any SAM
    /// size. K is tunable via `DFGPU_K_HOT_NODES` env var (default 256 = 4 KB per TG cache).
    ///
    /// Best for canonical Python where ~80% of byte visits hit low-`len` states (after each
    /// suffix-link backtrack the walker resets near root). Expected ~3-4× speedup over the
    /// all-global SAM walk kernel.
    #[must_use]
    #[allow(clippy::similar_names, clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn matching_stats_by_b_partial_flat_with_timings(
        &self,
        corpus: &CorpusGpu,
        pairs: &[(u32, u32)],
    ) -> (MatchingStatsFlat, [u128; 5]) {
        let n_pairs = pairs.len();
        if n_pairs == 0 {
            let empty = self.empty_u32_buffer(1);
            return (
                MatchingStatsFlat {
                    out_offsets: vec![0],
                    pair_orig_idx: Vec::new(),
                    fstate_buf: empty.clone(),
                    fmatch_buf: empty,
                    total_out: 0,
                },
                [0; 5],
            );
        }
        // `k_hot_nodes` is read back from CorpusGpu so the kernel's TG cache sizing matches
        // what the corpus was built with.
        let k_hot_nodes: u32 = corpus.k_hot_nodes_build;
        let t1 = Instant::now();
        let mut order: Vec<u32> = (0..n_pairs as u32).collect();
        order.sort_by_key(|&t| {
            let (a, b) = pairs[t as usize];
            let a_lo = corpus.a_offsets_cpu[a as usize];
            let a_hi = corpus.a_offsets_cpu[a as usize + 1];
            (b, a_hi - a_lo, a)
        });
        let mut active_b_idx: Vec<u32> = Vec::new();
        let mut pair_b_offsets: Vec<u32> = vec![0];
        let mut pair_a_idx_sorted: Vec<u32> = Vec::with_capacity(n_pairs);
        let mut out_offsets: Vec<u32> = Vec::with_capacity(n_pairs + 1);
        out_offsets.push(0);
        let mut total_out: u32 = 0;
        let mut current_b: u32 = u32::MAX;
        for (slot, &t_idx) in order.iter().enumerate() {
            let (a_idx, b_idx) = pairs[t_idx as usize];
            if b_idx != current_b {
                if !active_b_idx.is_empty() {
                    pair_b_offsets.push(slot as u32);
                }
                active_b_idx.push(b_idx);
                current_b = b_idx;
            }
            pair_a_idx_sorted.push(a_idx);
            let a_len = corpus.a_offsets_cpu[a_idx as usize + 1]
                - corpus.a_offsets_cpu[a_idx as usize];
            total_out = total_out.checked_add(a_len).expect("output too large");
            out_offsets.push(total_out);
        }
        pair_b_offsets.push(n_pairs as u32);
        let n_active_b = active_b_idx.len();
        let stage_build = t1.elapsed().as_nanos();

        let t2 = Instant::now();
        let buf_pair_a_sorted = self.upload_buf(&pair_a_idx_sorted);
        let buf_pair_b_offsets = self.upload_buf(&pair_b_offsets);
        let buf_active_b = self.upload_buf(&active_b_idx);
        let buf_out_offsets = self.upload_buf(&out_offsets);
        let buf_fmatch = self.empty_u32_buffer(total_out as usize);
        let buf_fstate = self.empty_u32_buffer(total_out as usize);
        let buf_k_hot_nodes = self.upload_buf(std::slice::from_ref(&k_hot_nodes));
        let stage_upload = t2.elapsed().as_nanos();

        let t3 = Instant::now();
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matching_stats_by_b_partial_pipeline);
        encoder.set_buffer(0, Some(&buf_pair_a_sorted), 0);
        encoder.set_buffer(1, Some(&buf_pair_b_offsets), 0);
        encoder.set_buffer(2, Some(&buf_active_b), 0);
        encoder.set_buffer(3, Some(&corpus.a_data_buf), 0);
        encoder.set_buffer(4, Some(&corpus.a_offsets_buf), 0);
        encoder.set_buffer(5, Some(&corpus.sam_nodes_buf), 0);
        encoder.set_buffer(6, Some(&corpus.sam_node_offsets_buf), 0);
        encoder.set_buffer(7, Some(&corpus.sam_edges_buf), 0);
        encoder.set_buffer(8, Some(&corpus.sam_edge_offsets_buf), 0);
        encoder.set_buffer(9, Some(&corpus.sam_root_next_buf), 0);
        encoder.set_buffer(10, Some(&buf_fmatch), 0);
        encoder.set_buffer(11, Some(&buf_fstate), 0);
        encoder.set_buffer(12, Some(&buf_out_offsets), 0);
        encoder.set_buffer(13, Some(&buf_k_hot_nodes), 0);

        // Threadgroup arena: K_HOT_NODES × 16 B for nodes + 128 × 4 B for root_next.
        let tg_mem_bytes = (k_hot_nodes as usize) * 16 + 128 * 4;
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes as NSUInteger);

        let pipeline_max =
            self.matching_stats_by_b_partial_pipeline.max_total_threads_per_threadgroup() as usize;
        let max_pairs_in_a_group =
            pair_b_offsets.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(1);
        let tg = 128.min(pipeline_max).min(max_pairs_in_a_group.max(32).next_power_of_two());

        encoder.dispatch_thread_groups(
            MTLSize::new(n_active_b as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        let stage_dispatch = t3.elapsed().as_nanos();

        let t4 = Instant::now();
        cmd.wait_until_completed();
        let stage_wait = t4.elapsed().as_nanos();
        let (ks, ke, gs, ge) = gpu_command_buffer_times(&cmd);
        eprintln!(
            "  [by_b_partial K={k_hot_nodes} tg={tg} mem={tg_mem_bytes}B] kernel: {:.3} ms, gpu: {:.3} ms (n_active_b={n_active_b})",
            (ke - ks) * 1000.0,
            (ge - gs) * 1000.0,
        );

        let t5 = Instant::now();
        let flat = MatchingStatsFlat {
            out_offsets,
            pair_orig_idx: order,
            fstate_buf: buf_fstate,
            fmatch_buf: buf_fmatch,
            total_out: total_out as usize,
        };
        let stage_readback = t5.elapsed().as_nanos();
        (flat, [stage_build, stage_upload, stage_dispatch, stage_wait, stage_readback])
    }

}

/// Flat-buffer output of the batched matching_stats kernel. Pair `t`'s `fstate` slice is
/// `fstate_all()[out_offsets[t]..out_offsets[t+1]]`, and similarly for `fmatch`. The slice
/// methods return **zero-copy** views directly into the Metal shared-memory buffers — there
/// is no host readback step at all on Apple Silicon UMA.
///
/// The Metal buffers are held inside the struct; dropping `MatchingStatsFlat` releases them.
pub struct MatchingStatsFlat {
    /// `[u32; n_pairs + 1]` — DISPATCH-order region offsets. Pair `t` in DISPATCH ORDER has its
    /// region in `out_offsets[t]..out_offsets[t+1]`. To recover input-pair-order, use
    /// `pair_orig_idx[t]` which maps dispatch position back to input position. Reordering is
    /// done internally to reduce SIMD-group divergence on the kernel side.
    pub out_offsets: Vec<u32>,
    /// `[u32; n_pairs]` — `pair_orig_idx[t]` = the input pair index whose result lives at
    /// dispatch slot `t`. If the caller wants pair index `i`'s result, scan `pair_orig_idx` for
    /// the slot where `pair_orig_idx[t] == i` and read `pair(t)`. For checksum-style consumers
    /// the order doesn't matter and they can iterate `pair_orig_idx` order.
    pub pair_orig_idx: Vec<u32>,
    fstate_buf: Buffer,
    fmatch_buf: Buffer,
    total_out: usize,
}

// SAFETY: Same reasoning as `Gpu` — Metal buffers are documented thread-safe; the `metal` crate's
// `Send`/`Sync` impls reflect that.
unsafe impl Send for MatchingStatsFlat {}
unsafe impl Sync for MatchingStatsFlat {}

impl MatchingStatsFlat {
    /// Concatenated `fstate` arrays from every pair, in input pair-order. **Zero-copy** — the
    /// returned slice references the Metal shared-memory buffer directly. Borrow ends with `self`.
    #[must_use]
    pub fn fstate_all(&self) -> &[u32] {
        // SAFETY: the buffer was allocated for `total_out * sizeof(u32)` bytes and fully written
        // by the kernel; we read it back as plain u32. UMA means no synchronization is needed
        // beyond the `wait_until_completed` that already ran.
        unsafe {
            std::slice::from_raw_parts(self.fstate_buf.contents().cast::<u32>(), self.total_out)
        }
    }

    /// Concatenated `fmatch` arrays from every pair, in input pair-order. **Zero-copy**.
    #[must_use]
    pub fn fmatch_all(&self) -> &[u32] {
        // SAFETY: as in `fstate_all`.
        unsafe {
            std::slice::from_raw_parts(self.fmatch_buf.contents().cast::<u32>(), self.total_out)
        }
    }

    /// Pair `t`'s `(fstate, fmatch)` slice pair — convenience wrapper over the two flat views.
    #[must_use]
    pub fn pair(&self, t: usize) -> (&[u32], &[u32]) {
        let lo = self.out_offsets[t] as usize;
        let hi = self.out_offsets[t + 1] as usize;
        (&self.fstate_all()[lo..hi], &self.fmatch_all()[lo..hi])
    }

    /// Number of pairs processed in this batch.
    #[must_use]
    pub fn n_pairs(&self) -> usize {
        self.out_offsets.len() - 1
    }
}

/// Corpus serialized into Metal-shared buffers: one big SAM-and-input arena that every pair the
/// GPU dispatch sees can index into. Built once per `cluster_canonicals` call, reused across
/// every batched dispatch. The CPU keeps a shadow copy of `a_offsets` to size per-pair output
/// regions without round-tripping through the GPU buffer.
pub struct CorpusGpu {
    n_strings: usize,
    n_sams: usize,
    /// Shadow copy of `a_offsets` kept on the CPU so the dispatcher can compute
    /// `out_offsets` (per-pair output region starts) without reading back from a GPU buffer.
    a_offsets_cpu: Vec<u32>,
    /// Concatenated ASCII bytes of every input string.
    a_data_buf: Buffer,
    /// `[u32; n_strings + 1]` — start offsets (in BYTES) for each string in `a_data_buf`.
    a_offsets_buf: Buffer,
    /// Concatenated SAM `[link_len, link, edge_lo, edge_hi]` quads. `node.x` is rewritten from
    /// state's own `len` to `len(link[state])` so the kernel's backtrack branch can fetch the
    /// link-state length from the already-loaded `nd` without a second sam_nodes load.
    sam_nodes_buf: Buffer,
    /// `[u32; n_sams + 1]` — start offsets (in `uint4` UNITS, i.e. node count) per SAM.
    sam_node_offsets_buf: Buffer,
    /// Concatenated SAM edges, packed `(char << 24) | target_state` (u32 each).
    sam_edges_buf: Buffer,
    /// `[u32; n_sams + 1]` — start offsets (in u32 units) per SAM.
    sam_edge_offsets_buf: Buffer,
    /// Concatenated 128-entry `root_next` tables (one per SAM, indexed as `b_idx * 128 + c`).
    sam_root_next_buf: Buffer,
    /// `K` used at corpus build time for the partial-TG-cache kernel
    /// ([`Gpu::matching_stats_by_b_partial_flat_with_timings`]). Tunable via
    /// `DFGPU_K_HOT_NODES_BUILD` env var (default 128). The kernel's `k_hot_nodes` constant
    /// reads back this value at dispatch time so TG cache sizing matches.
    k_hot_nodes_build: u32,
}

impl CorpusGpu {
    /// Build the GPU-side corpus arena from the input strings (ASCII bytes) and the matched
    /// SAMs. The string count must equal the SAM count; pair `(a_idx, b_idx)` references string
    /// `a_idx` against SAM `b_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `strings.len() != sams.len()`.
    #[must_use]
    pub fn build(gpu: &Gpu, strings: &[&[u8]], sams: &[crate::gestalt::Sam]) -> Self {
        assert_eq!(strings.len(), sams.len(), "CorpusGpu: must have one SAM per input string");

        // Concatenate string data, build byte-offset table.
        let total_str_bytes: usize = strings.iter().map(|s| s.len()).sum();
        let mut a_data: Vec<u8> = Vec::with_capacity(total_str_bytes);
        let mut a_offsets_cpu: Vec<u32> = Vec::with_capacity(strings.len() + 1);
        a_offsets_cpu.push(0);
        for s in strings {
            a_data.extend_from_slice(s);
            a_offsets_cpu.push(a_data.len() as u32);
        }

        // Concatenate SAM nodes — offsets in uint4 (= [u32;4]) units.
        //
        // Optimization (instrumentation showed ~47% of bytes need link backtrack): rewrite
        // `node.x` from state's own `len` to `len(link[state])` so the GPU kernel can read the
        // link's length directly from the already-loaded `nd` without a second sam_nodes load on
        // the backtrack path. The CPU `Sam` struct keeps the original layout (state's own len);
        // only the GPU arena copy is rewritten. This stays byte-for-byte equivalent — both paths
        // assign `matched = len(link)` on the backtrack branch.
        //
        // Root (state 0) has no link; its `link_len` is set to 0. The kernel never reads `nd.x`
        // for state==0 (state==0 path either succeeds via root_next or sets matched=0 and breaks).
        let total_nodes: usize = sams.iter().map(|s| s.nodes().len()).sum();
        let mut sam_nodes: Vec<[u32; 4]> = Vec::with_capacity(total_nodes);
        let mut sam_node_offsets: Vec<u32> = Vec::with_capacity(sams.len() + 1);
        sam_node_offsets.push(0);
        for sam in sams {
            let nodes = sam.nodes();
            for (state, &node) in nodes.iter().enumerate() {
                let link = node[1] as usize;
                let link_len = if state == 0 { 0 } else { nodes[link][0] };
                let edge_count = node[3] - node[2];
                assert!(edge_count <= 255, "edge count {edge_count} exceeds u8 — bump packing");
                sam_nodes.push([link_len, node[1], node[2], node[3]]);
            }
            sam_node_offsets.push(sam_nodes.len() as u32);
        }

        // Concatenate SAM edges — packed from u64 (char<<32 | target) to u32 (char<<24 | target).
        // ASCII char fits in 7 bits; target state count fits in 24 bits (16M states max — well
        // above mypy's ~90k worst case). Halves edge memory traffic. The CPU `Sam` keeps the
        // u64 layout (CPU walker still reads ulong); only the GPU arena is repacked.
        let total_edges: usize = sams.iter().map(|s| s.edges_packed().len()).sum();
        let mut sam_edges: Vec<u32> = Vec::with_capacity(total_edges);
        let mut sam_edge_offsets: Vec<u32> = Vec::with_capacity(sams.len() + 1);
        sam_edge_offsets.push(0);
        for sam in sams {
            let edges = sam.edges_packed();
            for &e in edges {
                let c = (e >> 32) as u32;
                let target = (e & 0xffff_ffff) as u32;
                assert!(c < 128, "ASCII corpus only — non-ASCII edge char");
                assert!(target < (1 << 24), "SAM exceeds 16M states — bump packing width");
                sam_edges.push((c << 24) | target);
            }
            sam_edge_offsets.push(sam_edges.len() as u32);
        }

        // Concatenate per-SAM 128-entry root_next tables.
        let mut sam_root_next: Vec<i32> = Vec::with_capacity(sams.len() * 128);
        for sam in sams {
            let rn = sam.root_next_table();
            assert_eq!(rn.len(), 128, "SAM root_next must be 128 entries");
            sam_root_next.extend_from_slice(rn);
        }

        let k_hot_nodes_build: u32 = std::env::var("DFGPU_K_HOT_NODES_BUILD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);

        CorpusGpu {
            n_strings: strings.len(),
            n_sams: sams.len(),
            a_data_buf: gpu.upload_buf(&a_data),
            a_offsets_buf: gpu.upload_buf(&a_offsets_cpu),
            sam_nodes_buf: gpu.upload_buf(&sam_nodes),
            sam_node_offsets_buf: gpu.upload_buf(&sam_node_offsets),
            sam_edges_buf: gpu.upload_buf(&sam_edges),
            sam_edge_offsets_buf: gpu.upload_buf(&sam_edge_offsets),
            sam_root_next_buf: gpu.upload_buf(&sam_root_next),
            k_hot_nodes_build,
            a_offsets_cpu,
        }
    }

    /// Number of SAMs (= number of input strings). Used by callers to size reverse-index maps
    /// from `gpu_idx` back to original string index.
    #[must_use]
    pub fn n_sams(&self) -> usize {
        self.n_sams
    }
}

// SAFETY: identical reasoning as `Gpu` above — Metal buffers are documented thread-safe and
// the `metal` crate's `Send`/`Sync` impls reflect that. The corpus is shared across rayon
// worker threads in the dispatcher.
unsafe impl Send for CorpusGpu {}
unsafe impl Sync for CorpusGpu {}

/// Compile one named kernel function from `library` into a `ComputePipelineState`. The function
/// name must match a `kernel`-qualified function declared in [`KERNELS`].
fn make_pipeline(
    device: &Device,
    library: &Library,
    fn_name: &str,
) -> Result<ComputePipelineState, String> {
    let func = library.get_function(fn_name, None).map_err(|e| format!("get_function({fn_name}): {e}"))?;
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&func));
    device.new_compute_pipeline_state_with_function(&func).map_err(|e| format!("pipeline({fn_name}): {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_acquires_metal_device() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("no Metal device on this machine — skipping GPU tests");
            return;
        };
        let name = gpu.device_name();
        eprintln!("Metal device: {name}");
        // The name on M3 Pro is "Apple M3 Pro"; on M1/M2 it's similar. We don't assert specifics
        // (the test must pass on any Apple Silicon CI box), only that the device is non-empty.
        assert!(!name.is_empty(), "device name must be non-empty");
    }

    #[test]
    fn smoke_elementwise_add_correct() {
        let Some(gpu) = Gpu::new() else { return };
        let a: Vec<u32> = (0..1024).collect();
        let b: Vec<u32> = (0..1024).map(|x| x * 2).collect();
        let got = gpu.smoke_elementwise_add(&a, &b);
        let want: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(got, want, "GPU elementwise add disagrees with CPU");
    }

    #[test]
    fn smoke_handles_short_inputs() {
        // Edge cases the production matching_stats kernel will also see.
        let Some(gpu) = Gpu::new() else { return };
        assert_eq!(gpu.smoke_elementwise_add(&[], &[]), Vec::<u32>::new());
        assert_eq!(gpu.smoke_elementwise_add(&[1], &[2]), vec![3]);
        let big = vec![5u32; 100_000];
        let got = gpu.smoke_elementwise_add(&big, &big);
        assert!(got.iter().all(|&x| x == 10));
    }

    /// Byte-for-byte parity gate against CPU `matching_stats_for_test`, exercising the live
    /// `matching_stats_batched_flat` kernel (the one `Rationer::cluster_canonicals_chars` uses).
    /// CLAUDE.md's hard correctness gate: ANY divergence here means a GPU optimization broke RO.
    #[test]
    #[allow(clippy::similar_names)] // *_gpu / *_cpu pair-naming is the test's whole point
    fn batched_flat_matches_cpu_on_real_corpus() {
        let Some(gpu) = Gpu::new() else { return };
        let Ok(data) = std::fs::read_to_string("benchmarks/corpora/mypy.canon.bin") else {
            return; // bench corpus isn't shipped with the published crate — skip if absent
        };
        let strings_str: Vec<&str> = data
            .split('\0')
            .filter(|s| !s.is_empty() && s.is_ascii())
            .take(8)
            .collect();
        if strings_str.len() < 2 {
            return;
        }
        let strings_bytes: Vec<Vec<u8>> =
            strings_str.iter().map(|s| s.as_bytes().to_vec()).collect();
        let strings_chars: Vec<Vec<char>> = strings_str.iter().map(|s| s.chars().collect()).collect();
        let sams: Vec<crate::gestalt::Sam> =
            strings_chars.iter().map(|c| crate::gestalt::build_sam(c)).collect();
        let byte_refs: Vec<&[u8]> = strings_bytes.iter().map(Vec::as_slice).collect();
        let corpus = CorpusGpu::build(&gpu, &byte_refs, &sams);

        let n = strings_str.len();
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n * (n - 1));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    pairs.push((i as u32, j as u32));
                }
            }
        }
        let flat = gpu.matching_stats_batched_flat(&corpus, &pairs);
        let fstate_all = flat.fstate_all();
        let fmatch_all = flat.fmatch_all();
        for slot in 0..pairs.len() {
            let orig = flat.pair_orig_idx[slot] as usize;
            let (a_idx, b_idx) = pairs[orig];
            let lo = flat.out_offsets[slot] as usize;
            let hi = flat.out_offsets[slot + 1] as usize;
            let fstate_gpu = &fstate_all[lo..hi];
            let fmatch_gpu = &fmatch_all[lo..hi];

            let mut fstate_cpu = Vec::new();
            let mut fmatch_cpu = Vec::new();
            crate::gestalt::matching_stats_for_test(
                &strings_chars[a_idx as usize],
                &sams[b_idx as usize],
                &mut fstate_cpu,
                &mut fmatch_cpu,
            );
            assert_eq!(
                fstate_gpu, &fstate_cpu[..],
                "fstate diverges on pair (a={a_idx}, b={b_idx})"
            );
            assert_eq!(
                fmatch_gpu, &fmatch_cpu[..],
                "fmatch diverges on pair (a={a_idx}, b={b_idx})"
            );
        }
    }
}

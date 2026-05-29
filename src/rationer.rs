//! `Rationer` — stateful similarity / clustering handle with explicit backend control.
//!
//! The free-function API (`ratio`, `cluster_canonicals`, …) is stateless and rebuilds every per-call
//! resource. That's fine for one-shot calls, but for callers that issue many clustering jobs in a
//! tight loop (e.g. `find-dup-defs` runs `cluster_canonicals` once per same-name group, often a few
//! thousand times per repo) the per-call setup — Metal device handshake, IOPM assertion, thread QoS,
//! GPU warmup dispatch — adds up to seconds. A `Rationer` constructs that state once at `new()` and
//! reuses it across every `.cluster_canonicals(…)` call.
//!
//! Whether any given call routes through the GPU or stays on CPU is an internal decision; callers
//! don't pass GPU flags, don't import the `gpu` module, and don't change with feature gates. On
//! systems without Metal (`cfg(not(target_os = "macos"))` or the `gpu` feature off), the GPU side is
//! simply absent and every call goes to CPU — same behaviour, same output.
//!
//! ## Usage
//!
//! ```ignore
//! let r = difflib_fast::Rationer::new();
//! for group in groups {
//!     let clusters = r.cluster_canonicals(&group.canonicals, 0.5);
//!     // ...
//! }
//! ```
//!
//! Pass `&Rationer` across rayon threads — it's `Send + Sync`, designed for shared use from
//! `par_iter()` chains.

// GPU-glue module: string/pair counts are bounded well within u32 (the same invariant the `gpu`
// module relies on), index arithmetic uses intentional `as` casts (incl. the `-1` non-ASCII
// sentinel round-tripped through i32), local `use`/`const` sit next to the code they serve, and the
// doc prose names internal types (`CorpusGpu`, `QoS`, …) without backticks — mirror `gpu.rs`'s
// module-level allows.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::iter_cloned_collect,
    clippy::redundant_closure_for_method_calls,
    clippy::type_complexity
)]

use crate::cluster_canonicals_chars;

#[cfg(all(feature = "gpu", target_os = "macos"))]
use crate::gpu::{BoostGuard, CorpusGpu, Gpu};

/// Backend selection for `Rationer` similarity work.
///
/// Picked at construction; can't change for a given handle (the resource set is wired up at
/// `build()` time). Re-create the handle if you need a different concurrency mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concurrency {
    /// Pure CPU — single-threaded if `threads(1)`, multi-threaded rayon otherwise. Works on
    /// every platform; the only mode available with `feature = "gpu"` disabled.
    Cpu,
    /// Allow the Metal GPU where it measured a net win. In practice that is **only**
    /// `cluster_canonicals` on a single group above `DFGPU_CLUSTER_THRESHOLD` (~1.1–1.4× CPU,
    /// peaking on mid-size canonical-Python corpora). `ratio_many` and `cluster_canonicals_multi`
    /// stay on CPU under this mode too — the GPU `matching_stats` offload lost on both at every
    /// size benchmarked (their GPU paths are env-opt-in only). On non-Metal platforms: `Cpu`.
    Gpu,
    /// Default. Same routing as `Gpu` (GPU only for big single-group `cluster_canonicals`,
    /// everything else CPU), plus rayon-parallel CPU for the per-pair `longest_in` recursion on the
    /// GPU output. Behaves identically to `Gpu` for the currently GPU-enabled path; kept as a
    /// distinct mode so heterogeneous overlap can be re-enabled per-path via the env thresholds
    /// without an API change.
    GpuPlusCpu,
}

impl Default for Concurrency {
    // Not `#[derive(Default)]`: the default is `GpuPlusCpu`, not the first variant.
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self::GpuPlusCpu
    }
}

/// Builder for [`Rationer`]. Configure once, then `.build()`.
pub struct RationerBuilder {
    concurrency: Concurrency,
    threads: Option<usize>,
    delta: f64,
}

impl Default for RationerBuilder {
    fn default() -> Self {
        Self { concurrency: Concurrency::default(), threads: None, delta: 0.0 }
    }
}

impl RationerBuilder {
    /// Pick a backend. See [`Concurrency`].
    #[must_use]
    pub fn concurrency(mut self, c: Concurrency) -> Self {
        self.concurrency = c;
        self
    }

    /// Set the rayon worker count for CPU-side work (filtering, per-pair recursion, assemble).
    /// `None` (default) uses the global rayon pool. `Some(1)` forces single-threaded execution.
    #[must_use]
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }

    /// Approximate-RO knob (Phase 3). `delta = 0.0` (default) means exact Ratcliff-Obershelp
    /// — bit-identical to Python difflib. `delta ∈ (0, 1]` trades a bounded amount of accuracy
    /// for fewer chain walks inside `longest_in`: the suffix-link chain is capped at roughly
    /// `1/√delta` ascensions, which empirically only fires on the long-tail (p99+) cases.
    /// Property-tested in `tests/approx_ro.rs`: actual worst-case absolute RO deviation
    /// stays below `delta` on canonical-Python corpora (typically several times tighter).
    #[must_use]
    pub fn delta(mut self, d: f64) -> Self {
        self.delta = d.clamp(0.0, 1.0);
        self
    }

    /// Materialize the handle. Acquires Metal device + IOPM boost assertion (when GPU is in
    /// the requested concurrency mode); allocates a dedicated rayon pool if `threads` is set.
    #[must_use]
    pub fn build(self) -> Rationer {
        Rationer::new_with(self.concurrency, self.threads, self.delta)
    }
}

/// Handle that owns long-lived clustering resources (Metal device, IOPM power assertion, thread QoS
/// boost). Construct once per process; share `&Rationer` across rayon workers.
///
/// On macOS with `feature = "gpu"`, `Rationer::new()` lazily acquires a Metal device + power
/// assertion. On platforms without Metal (or with the `gpu` feature disabled at compile time) the
/// struct is essentially empty — methods still work and produce the same answers, just always on CPU.
pub struct Rationer {
    concurrency: Concurrency,
    /// Local rayon pool when the builder pinned a thread count; `None` ⇒ use rayon's global pool.
    pool: Option<rayon::ThreadPool>,
    /// Phase 3 approximate-RO knob (see `RationerBuilder::delta`). 0.0 = exact (default).
    delta: f64,
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    gpu: Option<Gpu>,
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    _boost: Option<BoostGuard>,
}

impl Rationer {
    /// Start with sensible defaults: GPU+CPU concurrency, rayon's global thread pool.
    #[must_use]
    pub fn builder() -> RationerBuilder {
        RationerBuilder::default()
    }

    /// Create a `Rationer` with default settings (`Concurrency::GpuPlusCpu`, global rayon pool,
    /// `delta = 0.0` = exact RO). Never fails — if Metal is unavailable, the handle quietly
    /// degrades to CPU-only.
    ///
    /// Equivalent to `Rationer::builder().build()`.
    #[must_use]
    pub fn new() -> Self {
        Self::new_with(Concurrency::default(), None, 0.0)
    }

    /// Active `delta` for the approximate-RO knob — 0.0 means exact.
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    fn new_with(concurrency: Concurrency, threads: Option<usize>, delta: f64) -> Self {
        let pool = threads.and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok());

        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            let want_gpu = matches!(concurrency, Concurrency::Gpu | Concurrency::GpuPlusCpu);
            let (gpu, boost) = if want_gpu {
                // Acquire the boost guard FIRST so the GPU warmup dispatch inside Gpu::new()
                // runs at USER_INTERACTIVE QoS and with IOPM holding boost clocks — that gets
                // first-call latency out of the way at construction time, not on first
                // `.cluster_*()`.
                let b = BoostGuard::acquire();
                let g = Gpu::new();
                (g, Some(b))
            } else {
                (None, None)
            };
            Self { concurrency, pool, delta, gpu, _boost: boost }
        }
        #[cfg(not(all(feature = "gpu", target_os = "macos")))]
        {
            // On non-macOS (or with `feature = "gpu"` off) GPU is impossible — degrade to CPU.
            let _ = concurrency;
            Self { concurrency: Concurrency::Cpu, pool, delta }
        }
    }

    /// The active backend (after construction-time fallback). A `Rationer` built with
    /// `Concurrency::Gpu` on a non-Metal platform reports `Cpu` here.
    #[must_use]
    pub fn concurrency(&self) -> Concurrency {
        self.concurrency
    }

    /// Run a closure inside the configured rayon pool (the local pool when `threads(n)` was
    /// set; rayon's global pool otherwise). All public methods that do rayon work funnel
    /// through this so the `threads` setting actually takes effect.
    fn with_pool<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        if let Some(pool) = &self.pool {
            pool.install(f)
        } else {
            f()
        }
    }

    /// Single-pair Ratcliff–Obershelp ratio. Always CPU (one pair offers no GPU win). Same
    /// output as the free function [`crate::gestalt_ratio`].
    #[must_use]
    pub fn ratio(&self, a: &str, b: &str) -> f64 {
        crate::gestalt_ratio(a, b)
    }

    /// Batched ratio over a list of `(a, b)` string pairs, in parallel across cores.
    ///
    /// **CPU by default.** Benchmarking (mypy/django/sympy/ha/transformers, 61k–404k pairs) showed
    /// the GPU `matching_stats` offload *loses* here — 0.82–0.93× CPU across every size measured —
    /// because the CPU rayon path (intern uniques, prebuild each SAM once, `gestalt_ratio_prebuilt`
    /// per pair) is already efficient and the GPU's corpus-build + dispatch overhead isn't amortized
    /// by the relatively light per-pair `matching_stats` walk. So `ratio_many` stays on CPU even
    /// under `Concurrency::Gpu`. The GPU path is retained behind `DFGPU_RATIO_MANY_THRESHOLD=<n>`
    /// (default off) for experimentation / other hardware: set it to engage GPU at `pairs.len() >= n`.
    ///
    /// Output is the bit-identical Ratcliff–Obershelp ratio for each pair, in input order. On
    /// non-ASCII pairs the GPU path routes the affected pairs to the CPU per-pair fallback.
    #[must_use]
    pub fn ratio_many<S1, S2>(&self, pairs: &[(S1, S2)]) -> Vec<f64>
    where
        S1: AsRef<str> + Sync,
        S2: AsRef<str> + Sync,
    {
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            let want_gpu =
                matches!(self.concurrency, Concurrency::Gpu | Concurrency::GpuPlusCpu);
            // Default off (usize::MAX): GPU ratio_many measured slower than CPU at every tested
            // size. Opt in via env for experimentation.
            let gpu_threshold: usize = std::env::var("DFGPU_RATIO_MANY_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            if want_gpu && pairs.len() >= gpu_threshold {
                if let Some(gpu) = &self.gpu {
                    let delta = self.delta;
                    return self.with_pool(|| ratio_many_via_gpu(gpu, pairs, delta));
                }
            }
        }
        self.with_pool(|| ratio_many_cpu(pairs))
    }

    /// Build a reusable corpus over `strings`: parses chars, builds SAMs, and (when GPU is
    /// active) uploads the [`CorpusGpu`] arena. The returned [`PreparedRationer`] borrows the
    /// `Rationer` and lets you issue multiple `ratio_many_idx(pairs)` calls that amortize the
    /// SAM-build + GPU-upload cost over an arbitrary number of pair queries.
    ///
    /// Use this when the same string set is queried repeatedly (e.g. iterative refinement,
    /// dedup pipelines, find-dup-defs over a fixed file list). For one-shot queries the regular
    /// [`Rationer::ratio_many`] does the same work internally and is fine.
    ///
    /// All strings should be ASCII for the GPU path to engage; non-ASCII strings are kept in the
    /// SAM pool but their pair queries automatically fall back to a CPU per-pair compute on the
    /// host (same semantics as `ratio_many`).
    #[must_use]
    pub fn prepare<S: AsRef<str>>(&self, strings: &[S]) -> PreparedRationer<'_> {
        use rayon::prelude::*;
        let owned: Vec<String> = strings.iter().map(|s| s.as_ref().to_owned()).collect();
        let chars_pool: Vec<Vec<char>> =
            self.with_pool(|| owned.par_iter().map(|s| s.chars().collect()).collect());
        let sams: Vec<crate::gestalt::Sam> =
            self.with_pool(|| chars_pool.par_iter().map(|c| crate::gestalt::build_sam(c)).collect());

        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            // Identify ASCII-only strings; only those are uploaded to the GPU arena. Non-ASCII
            // strings stay in the SAM pool for the CPU fallback path.
            let mut gpu_idx_for: Vec<i32> = vec![-1; owned.len()];
            let mut ascii_strings: Vec<&str> = Vec::new();
            let mut ascii_sams: Vec<crate::gestalt::Sam> = Vec::new();
            // SAFETY: ascii_sams takes ownership of a CLONE of ascii SAMs — borrowing
            // would tangle lifetimes with `sams` (which we keep around for CPU fallback).
            // Sam is cheap to clone (Vec<u32> data); only done once at prepare time.
            for (i, s) in owned.iter().enumerate() {
                if s.bytes().all(|b| b < 128) {
                    gpu_idx_for[i] = ascii_strings.len() as i32;
                    ascii_strings.push(s.as_str());
                    ascii_sams.push(sams[i].clone());
                }
            }

            let corpus = if let Some(ref gpu) = self.gpu {
                if ascii_strings.is_empty() {
                    None
                } else {
                    let byte_refs: Vec<&[u8]> = ascii_strings.iter().map(|s| s.as_bytes()).collect();
                    Some(CorpusGpu::build(gpu, &byte_refs, &ascii_sams))
                }
            } else {
                None
            };

            PreparedRationer {
                rationer: self,
                strings: owned,
                chars_pool,
                sams,
                corpus,
                gpu_idx_for,
            }
        }
        #[cfg(not(all(feature = "gpu", target_os = "macos")))]
        {
            PreparedRationer { rationer: self, strings: owned, chars_pool, sams }
        }
    }

    /// Exact single-linkage clustering at `threshold`, identical to the free-standing
    /// [`crate::cluster_canonicals_chars`] in behaviour and output. Routes through the GPU on
    /// macOS when the group is large enough to amortize dispatch overhead; small groups stay on
    /// CPU because the GPU dispatch fixed cost (~5–50 ms) exceeds CPU rayon's full run for them.
    ///
    /// Routing: if the handle's [`Concurrency`] includes GPU and the group is big + all-ASCII,
    /// dispatch through Metal; else stay on the CPU. The size cutoff is set so the GPU path's
    /// `CorpusGpu` build + dispatch (~10–30 ms) is amortized over enough verified pairs to win.
    /// Override with `DFGPU_CLUSTER_THRESHOLD=<n>` env var; default 300.
    #[must_use]
    pub fn cluster_canonicals_chars(
        &self,
        chars: &[Vec<char>],
        threshold: f64,
    ) -> Vec<(Vec<usize>, f64)> {
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            let want_gpu =
                matches!(self.concurrency, Concurrency::Gpu | Concurrency::GpuPlusCpu);
            if want_gpu {
                if let Some(gpu) = &self.gpu {
                    let gpu_threshold: usize = std::env::var("DFGPU_CLUSTER_THRESHOLD")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(300);
                    if chars.len() >= gpu_threshold
                        && chars.iter().all(|c| c.iter().all(|&ch| (ch as u32) < 128))
                    {
                        return self.with_pool(|| {
                            cluster_canonicals_chars_via_gpu(gpu, chars, threshold, self.delta)
                        });
                    }
                }
            }
        }
        self.with_pool(|| cluster_canonicals_chars(chars, threshold))
    }

    /// String-input convenience, mirrors [`crate::cluster_canonicals`].
    #[must_use]
    pub fn cluster_canonicals(
        &self,
        canonicals: &[String],
        threshold: f64,
    ) -> Vec<(Vec<usize>, f64)> {
        let chars: Vec<Vec<char>> = canonicals.iter().map(|s| s.chars().collect()).collect();
        self.cluster_canonicals_chars(&chars, threshold)
    }

    /// Batched-across-groups cluster_canonicals.
    ///
    /// Run K independent clustering jobs as ONE GPU dispatch — concatenate every group's strings
    /// into a single Metal corpus arena, run filters per-group on the CPU, submit ALL surviving
    /// candidate pairs (from every group) in one batched `matching_stats` kernel call, then split
    /// results back to per-group `gestalt_edge_with_ms` + `assemble` on the CPU.
    ///
    /// The single-dispatch idea was meant to be find-dup-defs's win condition (thousands of
    /// same-name groups, each too small for per-call GPU overhead, batched into one dispatch).
    /// **CPU by default**, though: benchmarking showed the batched GPU path loses on that very shape
    /// — 0.70× CPU at 44 groups of 50, only reaching break-even (~0.99×) at a handful of large
    /// groups — the corpus-build + dispatch overhead isn't amortized when each group's surviving-pair
    /// count is small. So `cluster_canonicals_multi` runs per-group on CPU (rayon across groups)
    /// unless `DFGPU_MULTI_THRESHOLD=<total strings>` is set to opt the batched GPU path back in.
    ///
    /// Returns one cluster list per input group in the input order. Each list has identical
    /// semantics to [`cluster_canonicals`] called on that group alone.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn cluster_canonicals_multi(
        &self,
        groups: &[Vec<String>],
        threshold: f64,
    ) -> Vec<Vec<(Vec<usize>, f64)>> {
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            let want_gpu =
                matches!(self.concurrency, Concurrency::Gpu | Concurrency::GpuPlusCpu);
            // Default off (usize::MAX): the batched GPU path measured at best break-even vs the
            // per-group CPU path. Opt in via env for experimentation / other hardware.
            let gpu_threshold: usize = std::env::var("DFGPU_MULTI_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            let total: usize = groups.iter().map(Vec::len).sum();
            if want_gpu && total >= gpu_threshold {
                if let Some(gpu) = &self.gpu {
                    return self
                        .with_pool(|| cluster_canonicals_multi_via_gpu(gpu, groups, threshold, self.delta));
                }
            }
        }
        // CPU path (default): iterate per group across rayon.
        use rayon::prelude::*;
        self.with_pool(|| {
            groups
                .par_iter()
                .map(|g| {
                    let chars: Vec<Vec<char>> =
                        g.iter().map(|s| s.chars().collect()).collect();
                    cluster_canonicals_chars(&chars, threshold)
                })
                .collect()
        })
    }
}

impl Default for Rationer {
    fn default() -> Self {
        Self::new()
    }
}

/// Reusable corpus + SAMs built by [`Rationer::prepare`]. Hold this across many
/// `ratio_many_idx` calls to amortize SAM building and the GPU upload over the lifetime of the
/// string set. Borrowing the `Rationer` keeps the Metal device and thread pool wiring alive.
pub struct PreparedRationer<'r> {
    rationer: &'r Rationer,
    strings: Vec<String>,
    chars_pool: Vec<Vec<char>>,
    sams: Vec<crate::gestalt::Sam>,
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    corpus: Option<CorpusGpu>,
    /// Maps original string index → index into `corpus` (ascii-only). `-1` for non-ASCII strings.
    /// Used to translate pair indices the user gives in `ratio_many_idx` into GPU-corpus indices.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    gpu_idx_for: Vec<i32>,
}

impl PreparedRationer<'_> {
    /// Number of strings in the prepared corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// True iff `len() == 0`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Batched Ratcliff-Obershelp ratio over a pair list referencing the prepared strings by
    /// index (`(i, j)` ⇒ `ratio(strings[i], strings[j])`). All SAM / GPU-corpus state is reused
    /// from `prepare()`, so a long-lived `PreparedRationer` paying its build cost once can serve
    /// many queries at the kernel-pure GPU throughput.
    ///
    /// # Panics
    /// Panics if any pair index is out of bounds.
    #[must_use]
    pub fn ratio_many_idx(&self, pairs: &[(u32, u32)]) -> Vec<f64> {
        let n = pairs.len();
        if n == 0 {
            return Vec::new();
        }
        // Bounds-check eagerly so the rest of the code can use unchecked indexing in the kernel.
        let n_strings = self.strings.len() as u32;
        for &(i, j) in pairs {
            assert!(i < n_strings && j < n_strings, "ratio_many_idx: index out of bounds");
        }

        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            if let (Some(corpus), Some(gpu)) = (self.corpus.as_ref(), self.rationer.gpu.as_ref()) {
                let delta = self.rationer.delta;
                return self.rationer.with_pool(|| {
                    ratio_many_via_prepared_gpu(gpu, corpus, &self.sams, &self.chars_pool, &self.gpu_idx_for, pairs, delta)
                });
            }
        }
        // CPU fallback — reuse prebuilt SAMs.
        self.rationer.with_pool(|| ratio_many_via_prepared_cpu(&self.chars_pool, &self.sams, pairs))
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn ratio_many_via_prepared_gpu(
    gpu: &Gpu,
    corpus: &CorpusGpu,
    sams: &[crate::gestalt::Sam],
    chars_pool: &[Vec<char>],
    gpu_idx_for: &[i32],
    pairs: &[(u32, u32)],
    delta: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    let n = pairs.len();
    let mut out = vec![0.0f64; n];

    // Split into GPU-eligible (both strings ASCII) and CPU-fallback lanes.
    let mut gpu_pairs: Vec<(u32, u32)> = Vec::with_capacity(n);
    let mut gpu_slot_for: Vec<usize> = Vec::with_capacity(n);
    let mut cpu_slot_for: Vec<usize> = Vec::new();
    for (slot, &(a, b)) in pairs.iter().enumerate() {
        let ga = gpu_idx_for[a as usize];
        let gb = gpu_idx_for[b as usize];
        if ga >= 0 && gb >= 0 {
            gpu_pairs.push((ga as u32, gb as u32));
            gpu_slot_for.push(slot);
        } else {
            cpu_slot_for.push(slot);
        }
    }

    // GPU lane: matching_stats kernel + CPU `gestalt_edge_with_ms` per pair. This is the only
    // GPU path that wins on real workloads; Stage 4d/4g full_ratio kernels lose 3× because
    // longest_in's seg-tree queries are bandwidth-bound on the 50-200 MB seg_data buffer with
    // no GPU cache locality. See `src/new/PERF_MAP.md` for measurements.
    if !gpu_pairs.is_empty() {
        // gpu_idx → original_idx reverse map for fetching chars/sams by ORIGINAL string index.
        let mut orig_for: Vec<u32> = vec![u32::MAX; corpus.n_sams()];
        for (orig, &gi) in gpu_idx_for.iter().enumerate() {
            if gi >= 0 {
                orig_for[gi as usize] = orig as u32;
            }
        }
        let flat = gpu.matching_stats_by_b_partial_flat_with_timings(corpus, &gpu_pairs).0;
        let fstate_all = flat.fstate_all();
        let fmatch_all = flat.fmatch_all();
        let results: Vec<(usize, f64)> = (0..gpu_pairs.len())
            .into_par_iter()
            .map(|slot| {
                let orig_pair_idx = flat.pair_orig_idx[slot] as usize;
                let (ga, gb) = gpu_pairs[orig_pair_idx];
                let lo = flat.out_offsets[slot] as usize;
                let hi = flat.out_offsets[slot + 1] as usize;
                let fstate = &fstate_all[lo..hi];
                let fmatch = &fmatch_all[lo..hi];
                let oa = orig_for[ga as usize] as usize;
                let ob = orig_for[gb as usize] as usize;
                let r = crate::gestalt::gestalt_edge_with_ms_delta(
                    &chars_pool[oa],
                    &chars_pool[ob],
                    &sams[ob],
                    fstate,
                    fmatch,
                    0.0,
                    delta,
                )
                .unwrap_or(0.0);
                (gpu_slot_for[orig_pair_idx], r)
            })
            .collect();
        for (slot, r) in results {
            out[slot] = r;
        }
    }

    // CPU lane (non-ASCII pairs).
    if !cpu_slot_for.is_empty() {
        let cpu_results: Vec<(usize, f64)> = cpu_slot_for
            .par_iter()
            .map(|&slot| {
                let (i, j) = pairs[slot];
                let a = &chars_pool[i as usize];
                let b = &chars_pool[j as usize];
                let r = crate::gestalt::gestalt_ratio_prebuilt(a, b, &sams[j as usize]);
                (slot, r)
            })
            .collect();
        for (slot, r) in cpu_results {
            out[slot] = r;
        }
    }

    out
}

fn ratio_many_via_prepared_cpu(
    chars_pool: &[Vec<char>],
    sams: &[crate::gestalt::Sam],
    pairs: &[(u32, u32)],
) -> Vec<f64> {
    use rayon::prelude::*;
    let n = pairs.len();
    // OPTIMIZATION C: sort pair indices by b (then a) so adjacent rayon-chunk work hits the
    // same SAM → b's node + edges + epmeta + epos stay in L2 across calls. Without this each
    // worker thread bounces between SAMs in input order (random) and pays L2 eviction per pair.
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_unstable_by_key(|&i| {
        let (a, b) = pairs[i as usize];
        (b, a)
    });
    let mut out = vec![0.0f64; n];
    let results: Vec<(usize, f64)> = perm
        .par_iter()
        .map(|&pi| {
            let pi_us = pi as usize;
            let (i, j) = pairs[pi_us];
            let a = &chars_pool[i as usize];
            let b = &chars_pool[j as usize];
            let r = crate::gestalt::gestalt_ratio_prebuilt(a, b, &sams[j as usize]);
            (pi_us, r)
        })
        .collect();
    for (pi, r) in results {
        out[pi] = r;
    }
    out
}

/// CPU `ratio_many`: intern unique strings, prebuild each one's SAM once, then call
/// `gestalt_ratio_prebuilt` per pair. Without this we'd pay a SAM build per pair — for
/// 50 k pairs over 1 k unique strings that's 50 k rebuilds vs 1 k. Matches what the GPU
/// path's CorpusGpu does, so the cpu-vs-gpu speedup measurement is apples to apples.
fn ratio_many_cpu<S1, S2>(pairs: &[(S1, S2)]) -> Vec<f64>
where
    S1: AsRef<str> + Sync,
    S2: AsRef<str> + Sync,
{
    use std::collections::HashMap;

    use rayon::prelude::*;

    if pairs.is_empty() {
        return Vec::new();
    }

    let mut pool: Vec<String> = Vec::new();
    let mut by_str: HashMap<String, u32> = HashMap::new();
    let mut pair_idx: Vec<(u32, u32)> = Vec::with_capacity(pairs.len());
    for p in pairs {
        let a: &str = p.0.as_ref();
        let b: &str = p.1.as_ref();
        let mut intern = |s: &str| -> u32 {
            if let Some(&id) = by_str.get(s) {
                return id;
            }
            let idx = pool.len() as u32;
            pool.push(s.to_owned());
            by_str.insert(s.to_owned(), idx);
            idx
        };
        let ai = intern(a);
        let bi = intern(b);
        pair_idx.push((ai, bi));
    }
    drop(by_str);

    let chars_pool: Vec<Vec<char>> =
        pool.par_iter().map(|s| s.chars().collect()).collect();
    let sams: Vec<crate::gestalt::Sam> =
        chars_pool.par_iter().map(|c| crate::gestalt::build_sam(c)).collect();

    // OPTIMIZATION C: sort indices by b (then a) so each rayon worker gets a chunk of consecutive
    // pairs sharing the same SAM-b → b's node + edges + epmeta + epos stay in L2 across calls.
    let n = pair_idx.len();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_unstable_by_key(|&i| {
        let (a, b) = pair_idx[i as usize];
        (b, a)
    });
    let mut out = vec![0.0f64; n];
    let results: Vec<(usize, f64)> = perm
        .par_iter()
        .map(|&pi| {
            let pi_us = pi as usize;
            let (ai, bi) = pair_idx[pi_us];
            let a = &chars_pool[ai as usize];
            let b = &chars_pool[bi as usize];
            let sam_b = &sams[bi as usize];
            let r = crate::gestalt::gestalt_ratio_prebuilt(a, b, sam_b);
            (pi_us, r)
        })
        .collect();
    for (pi, r) in results {
        out[pi] = r;
    }
    out
}

/// Stage-4c: GPU port of `ratio_many`. Pipeline is:
///
/// 1. **Intern** every unique `&str` from the input pairs into a flat pool — most workloads
///    have lots of repeated strings (find-dup-defs's same-name groups, cluster_canonicals
///    inside the same group, etc.), so deduplication shrinks the corpus we build SAMs over.
/// 2. **Split** the pair list into ASCII-routed and non-ASCII routed lanes. Non-ASCII pairs
///    fall through to CPU `gestalt_ratio` (the GPU kernel reads `u8`).
/// 3. **Build SAMs + CorpusGpu** over the ASCII pool. Rayon-parallel.
/// 4. **GPU dispatch**: matching_stats kernel batched over the surviving ASCII pairs.
/// 5. **Per-pair recursion on CPU** using the GPU-filled `(fstate, fmatch)` — same code path
///    as `gestalt_edge_with_ms` with threshold 0, so the return value is the unconditional
///    ratio (no early-exit drop).
/// 6. **Merge** ASCII and non-ASCII results back in input order.
///
/// Same byte-for-byte correctness guarantee as `gestalt_ratio`. Wins on big batches because
/// the heavy SAM walk runs on the bandwidth-bound GPU at ~3× CPU rayon throughput, and the
/// CorpusGpu build cost (~10–50 ms) is amortized across all pairs.
#[cfg(all(feature = "gpu", target_os = "macos"))]
#[allow(clippy::too_many_lines)]
fn ratio_many_via_gpu<S1, S2>(gpu: &Gpu, pairs: &[(S1, S2)], delta: f64) -> Vec<f64>
where
    S1: AsRef<str> + Sync,
    S2: AsRef<str> + Sync,
{
    use std::collections::HashMap;

    use rayon::prelude::*;

    use crate::gpu::CorpusGpu;

    let n_pairs = pairs.len();
    if n_pairs == 0 {
        return Vec::new();
    }

    // Step 1 — intern unique strings into a pool. Routing: ASCII strings get an index in the
    // GPU corpus; non-ASCII strings get the `NON_ASCII` sentinel and the pair is handled by
    // the CPU fallback below. We use `String`-keyed dedup so the pool owns its data — the
    // input slice's lifetime isn't carried into the HashMap.
    const NON_ASCII: u32 = u32::MAX;
    let mut pool: Vec<String> = Vec::new();
    let mut by_str: HashMap<String, u32> = HashMap::new();
    let mut pair_idx: Vec<(u32, u32)> = Vec::with_capacity(n_pairs);
    for p in pairs {
        let a: &str = p.0.as_ref();
        let b: &str = p.1.as_ref();
        let mut intern = |s: &str| -> u32 {
            if !s.chars().all(|c| (c as u32) < 128) {
                return NON_ASCII;
            }
            if let Some(&id) = by_str.get(s) {
                return id;
            }
            let idx = pool.len() as u32;
            pool.push(s.to_owned());
            by_str.insert(s.to_owned(), idx);
            idx
        };
        let ai = intern(a);
        let bi = intern(b);
        pair_idx.push((ai, bi));
    }
    let n_unique = pool.len();
    drop(by_str);

    // Step 2 — split pairs into GPU and CPU lanes. The GPU lane has all `(a, b)` where both
    // strings are ASCII; the CPU lane has the rest.
    let mut gpu_pairs: Vec<(u32, u32)> = Vec::with_capacity(n_pairs);
    let mut gpu_slot_for: Vec<usize> = Vec::with_capacity(n_pairs);
    let mut cpu_slot_for: Vec<usize> = Vec::new();
    for (i, &(a, b)) in pair_idx.iter().enumerate() {
        if a == NON_ASCII || b == NON_ASCII {
            cpu_slot_for.push(i);
        } else {
            gpu_pairs.push((a, b));
            gpu_slot_for.push(i);
        }
    }

    // Step 3 — build SAMs + char view in parallel.
    let chars_pool: Vec<Vec<char>> =
        pool.par_iter().map(|s| s.chars().collect()).collect();
    let sams: Vec<crate::gestalt::Sam> =
        chars_pool.par_iter().map(|c| crate::gestalt::build_sam(c)).collect();
    let byte_refs: Vec<&[u8]> = pool.iter().map(|s| s.as_bytes()).collect();
    let corpus = CorpusGpu::build(gpu, &byte_refs, &sams);
    let _ = n_unique;

    // Allocate output, fill from both lanes.
    let mut out = vec![0.0f64; n_pairs];

    // Step 4–5 — GPU lane. `matching_stats_by_b_partial_flat` does the SAM walk on the GPU;
    // CPU runs `gestalt_edge_with_ms` (longest_in + RO recursion) per pair on rayon. This is
    // the only GPU path that wins on real workloads (see `src/new/PERF_MAP.md`).
    if !gpu_pairs.is_empty() {
        let flat = gpu.matching_stats_by_b_partial_flat_with_timings(&corpus, &gpu_pairs).0;
        let fstate_all = flat.fstate_all();
        let fmatch_all = flat.fmatch_all();
        let results: Vec<(usize, f64)> = (0..gpu_pairs.len())
            .into_par_iter()
            .map(|slot| {
                let orig = flat.pair_orig_idx[slot] as usize;
                let (a_idx, b_idx) = gpu_pairs[orig];
                let lo = flat.out_offsets[slot] as usize;
                let hi = flat.out_offsets[slot + 1] as usize;
                let fstate = &fstate_all[lo..hi];
                let fmatch = &fmatch_all[lo..hi];
                let r = crate::gestalt::gestalt_edge_with_ms_delta(
                    &chars_pool[a_idx as usize],
                    &chars_pool[b_idx as usize],
                    &sams[b_idx as usize],
                    fstate,
                    fmatch,
                    0.0,
                    delta,
                )
                .unwrap_or(0.0);
                (gpu_slot_for[orig], r)
            })
            .collect();
        for (slot, r) in results {
            out[slot] = r;
        }
    }

    // Step 6 — CPU lane for non-ASCII (rare) pairs.
    let cpu_results: Vec<(usize, f64)> = cpu_slot_for
        .par_iter()
        .map(|&i| {
            let (a, b) = &pairs[i];
            (i, crate::gestalt_ratio(a.as_ref(), b.as_ref()))
        })
        .collect();
    for (slot, r) in cpu_results {
        out[slot] = r;
    }
    out
}

/// Cross-group batched GPU path — one CorpusGpu, one dispatch, K group results.
/// See [`Rationer::cluster_canonicals_multi`] for the contract.
#[cfg(all(feature = "gpu", target_os = "macos"))]
#[allow(clippy::too_many_lines)]
fn cluster_canonicals_multi_via_gpu(
    gpu: &Gpu,
    groups: &[Vec<String>],
    threshold: f64,
    delta: f64,
) -> Vec<Vec<(Vec<usize>, f64)>> {
    use rayon::prelude::*;

    use crate::gpu::CorpusGpu;
    use crate::{assemble, char_counts, quick_ratio_counts, real_quick_ratio};

    if groups.is_empty() {
        return Vec::new();
    }

    // Flatten: every (group_idx, in_group_idx) → global_idx. Per-group offset for unflattening.
    let mut group_offsets: Vec<usize> = Vec::with_capacity(groups.len() + 1);
    group_offsets.push(0);
    for g in groups {
        let end = group_offsets[group_offsets.len() - 1] + g.len();
        group_offsets.push(end);
    }

    // OPTIMIZATION E: dedup strings across groups before building SAMs. find-dup-defs's actual
    // shape is "the same function name appears in many groups" — the synthetic `ratoner-groups`
    // bench partitions strings by `len % 50` so each string is in exactly one group (no dedup
    // possible there), but real workloads dedupe heavily. We build SAMs once per UNIQUE string
    // and remap flat (input-order) indices via `flat_to_unique` whenever we touch SAM-side data.
    // For pair (a_flat, b_flat): both sides are mapped to unique indices for the GPU corpus and
    // for SAM lookups; the FLAT indices are kept for per-group demux (group_offsets[] uses flat
    // space).
    let total: usize = groups.iter().map(|g| g.len()).sum();
    let mut unique_idx: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut unique_strings: Vec<&str> = Vec::new();
    let mut flat_to_unique: Vec<u32> = Vec::with_capacity(total);
    for g in groups {
        for s in g {
            let s_ref = s.as_str();
            let u = if let Some(&u) = unique_idx.get(s_ref) {
                u
            } else {
                let u = unique_strings.len() as u32;
                unique_strings.push(s_ref);
                unique_idx.insert(s_ref, u);
                u
            };
            flat_to_unique.push(u);
        }
    }
    let unique_chars: Vec<Vec<char>> =
        unique_strings.par_iter().map(|s| s.chars().collect()).collect();
    let unique_sams: Vec<crate::gestalt::Sam> =
        unique_chars.par_iter().map(|c| crate::gestalt::build_sam(c)).collect();
    let unique_ascii: Vec<bool> =
        unique_chars.iter().map(|c| c.iter().all(|&ch| (ch as u32) < 128)).collect();

    // group_ascii: all strings in a group are ASCII → ok for GPU. Resolve via unique_ascii.
    let group_ascii: Vec<bool> = (0..groups.len())
        .map(|gi| {
            let lo = group_offsets[gi];
            let hi = group_offsets[gi + 1];
            (lo..hi).all(|i| unique_ascii[flat_to_unique[i] as usize])
        })
        .collect();

    // CorpusGpu is built over UNIQUE strings; the GPU kernel indexes by unique_idx. Pair (a, b)
    // fed to GPU must be the unique pair, not the flat one.
    let unique_bytes: Vec<Vec<u8>> = unique_chars
        .iter()
        .map(|c| {
            if c.iter().all(|&ch| (ch as u32) < 128) {
                c.iter().map(|&ch| ch as u8).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    let byte_refs: Vec<&[u8]> = unique_bytes.iter().map(Vec::as_slice).collect();
    let corpus = CorpusGpu::build(gpu, &byte_refs, &unique_sams);

    // Below, indexing flat → unique uses `flat_to_unique[flat_i] as usize` whenever we need
    // SAM/chars data. Pair tuples carry BOTH flat (for demux) and unique (for GPU) indices.
    let flat_chars = |flat_i: u32| -> &Vec<char> { &unique_chars[flat_to_unique[flat_i as usize] as usize] };

    // Per-group filtering on CPU (length + quick_ratio). Survivors are concatenated into one
    // global candidate list. Each candidate carries:
    //   (a_unique, b_unique)  → for the GPU CorpusGpu lookup (unique-string indexed)
    //   (a_flat, b_flat)      → for per-group demux: local = flat - group_offsets[gi]
    //   group_idx             → so we can route the edge back to the right group
    let unique_counts: Vec<Vec<(char, u32)>> =
        unique_chars.par_iter().map(|c| char_counts(c)).collect();
    let per_group_candidates: Vec<Vec<(u32, u32, u32, u32, u32)>> = (0..groups.len())
        .into_par_iter()
        .map(|gi| {
            if !group_ascii[gi] {
                return Vec::new();
            }
            let lo = group_offsets[gi];
            let hi = group_offsets[gi + 1];
            let n = hi - lo;
            let mut order: Vec<usize> = (lo..hi).collect();
            order.sort_by_key(|&i| flat_chars(i as u32).len());
            let mut out: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
            #[allow(clippy::cast_possible_truncation)]
            for p in 0..n {
                let i = order[p];
                let i_u = flat_to_unique[i] as usize;
                for &j in &order[p + 1..] {
                    let j_u = flat_to_unique[j] as usize;
                    let ci = &unique_chars[i_u];
                    let cj = &unique_chars[j_u];
                    if real_quick_ratio(ci, cj) < threshold {
                        break;
                    }
                    if quick_ratio_counts(&unique_counts[i_u], &unique_counts[j_u], ci.len() + cj.len()) < threshold {
                        continue;
                    }
                    let (loi_flat, hii_flat) = if i < j { (i, j) } else { (j, i) };
                    let loi_u = flat_to_unique[loi_flat];
                    let hii_u = flat_to_unique[hii_flat];
                    out.push((loi_u, hii_u, loi_flat as u32, hii_flat as u32, gi as u32));
                }
            }
            out
        })
        .collect();

    // Concatenate all candidates into one global list. `pairs_for_gpu` is in UNIQUE space (fed to
    // GPU); `pair_flat` parallel-stores flat indices for per-group demux + gestalt_edge_with_ms.
    let mut pairs_for_gpu: Vec<(u32, u32)> = Vec::new();
    let mut pair_flat: Vec<(u32, u32)> = Vec::new();
    let mut pair_group: Vec<u32> = Vec::new();
    for group_pairs in &per_group_candidates {
        for &(au, bu, af, bf, gi) in group_pairs {
            pairs_for_gpu.push((au, bu));
            pair_flat.push((af, bf));
            pair_group.push(gi);
        }
    }

    // Per-group `pairs` arrays (for `assemble` later) — populate from GPU results.
    let mut per_group_edges: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); groups.len()];

    if !pairs_for_gpu.is_empty() {
        // Chunk the GPU dispatch to keep the output buffer (fmatch + fstate) under ~1 GB. Each
        // pair contributes `a_len * 8 B`; for canonical Python the avg is ~1 KB/pair, so we cap
        // at 250 k pairs per dispatch. Without this cap, multi-million-pair runs OOM Metal's
        // single-buffer limit and the kernel silently returns zero output.
        let max_pairs_per_dispatch: usize = std::env::var("DFGPU_MAX_PAIRS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(250_000);
        let mut edges_acc: Vec<(u32, u32, u32, f64)> = Vec::new(); // (a_flat, b_flat, gi, ratio)
        for chunk_start in (0..pairs_for_gpu.len()).step_by(max_pairs_per_dispatch) {
            let chunk_end = (chunk_start + max_pairs_per_dispatch).min(pairs_for_gpu.len());
            let chunk_pairs = &pairs_for_gpu[chunk_start..chunk_end]; // unique-indexed
            let chunk_pair_flat = &pair_flat[chunk_start..chunk_end];
            let chunk_pair_group = &pair_group[chunk_start..chunk_end];
            let flat = gpu.matching_stats_batched_flat(&corpus, chunk_pairs);
            let fstate_all = flat.fstate_all();
            let fmatch_all = flat.fmatch_all();
            let chunk_edges: Vec<(u32, u32, u32, f64)> = (0..chunk_pairs.len())
                .into_par_iter()
                .filter_map(|slot| {
                    let orig = flat.pair_orig_idx[slot] as usize;
                    let (au, bu) = chunk_pairs[orig]; // unique indices for SAM/chars
                    let (af, bf) = chunk_pair_flat[orig]; // flat indices for demux
                    let gi = chunk_pair_group[orig];
                    let lo_st = flat.out_offsets[slot] as usize;
                    let hi_st = flat.out_offsets[slot + 1] as usize;
                    let fstate = &fstate_all[lo_st..hi_st];
                    let fmatch = &fmatch_all[lo_st..hi_st];
                    let ratio = crate::gestalt::gestalt_edge_with_ms_delta(
                        &unique_chars[au as usize],
                        &unique_chars[bu as usize],
                        &unique_sams[bu as usize],
                        fstate,
                        fmatch,
                        threshold,
                        delta,
                    )?;
                    Some((af, bf, gi, ratio))
                })
                .collect();
            edges_acc.extend(chunk_edges);
        }

        for (af, bf, gi, ratio) in edges_acc {
            let base = group_offsets[gi as usize];
            per_group_edges[gi as usize].push((
                af as usize - base,
                bf as usize - base,
                ratio,
            ));
        }
    }

    // Non-ASCII groups: handle with the pure CPU path (rare).
    for (gi, ascii) in group_ascii.iter().enumerate() {
        if !ascii {
            let lo = group_offsets[gi];
            let hi = group_offsets[gi + 1];
            // Materialise the group's per-flat-position SAMs + chars via flat_to_unique (cloning
            // — see comment on assemble below: SAM-clone is much cheaper than SAM-build, so on
            // dedupe-heavy real workloads the E saving still dominates).
            let local_chars: Vec<Vec<char>> = (lo..hi)
                .map(|i| unique_chars[flat_to_unique[i] as usize].clone())
                .collect();
            let clusters = cluster_canonicals_chars(&local_chars, threshold);
            per_group_edges[gi].clear();
            let mut out: Vec<Vec<(Vec<usize>, f64)>> = Vec::with_capacity(groups.len());
            for (k, edges) in per_group_edges.into_iter().enumerate() {
                if k == gi {
                    out.push(clusters.clone());
                } else {
                    let n_k = group_offsets[k + 1] - group_offsets[k];
                    let lo_k = group_offsets[k];
                    let chars_k: Vec<Vec<char>> = (lo_k..lo_k + n_k)
                        .map(|i| unique_chars[flat_to_unique[i] as usize].clone())
                        .collect();
                    let sams_k: Vec<crate::gestalt::Sam> = (lo_k..lo_k + n_k)
                        .map(|i| unique_sams[flat_to_unique[i] as usize].clone())
                        .collect();
                    out.push(assemble(n_k, edges, &chars_k, &sams_k));
                }
            }
            return out;
        }
    }

    // Assemble per-group on CPU. Each group is independent; parallelize across groups. We
    // materialise per-group chars/sams from unique storage; SAM-clone is a Vec memcpy of the
    // already-built tables — much cheaper than re-building SAMs from scratch.
    (0..groups.len())
        .into_par_iter()
        .map(|gi| {
            let lo = group_offsets[gi];
            let hi = group_offsets[gi + 1];
            let n = hi - lo;
            let chars_k: Vec<Vec<char>> = (lo..hi)
                .map(|i| unique_chars[flat_to_unique[i] as usize].clone())
                .collect();
            let sams_k: Vec<crate::gestalt::Sam> = (lo..hi)
                .map(|i| unique_sams[flat_to_unique[i] as usize].clone())
                .collect();
            assemble(n, std::mem::take(&mut per_group_edges[gi].clone()), &chars_k, &sams_k)
        })
        .collect()
}

/// Stage-4b: do the per-pair `matching_stats` walk in ONE GPU dispatch instead of CPU rayon's
/// per-pair scan. The CPU side keeps doing the cheap stuff — length blocking, `quick_ratio`
/// filter (these kill 70–90% of pairs without ever computing `matching_stats`), and the small
/// `longest_in` stack walk per surviving pair (data-dependent depth, poor GPU fit).
///
/// Bit-identical to [`crate::cluster_canonicals_chars`] under any threshold ≥ 0; the kernel
/// computes the same `(fstate, fmatch)` byte-for-byte, and `gestalt::gestalt_edge_with_ms`
/// runs the same recursion + early-exit on the host with those arrays.
///
/// Cost model: per-call GPU setup ≈ 5 ms (sort+offsets+upload) + dispatch ≈ proportional to
/// the number of surviving-pair bytes. For groups smaller than `DFGPU_CLUSTER_THRESHOLD` the
/// caller stays on the pure-CPU path so we don't pay that fixed cost for nothing.
#[cfg(all(feature = "gpu", target_os = "macos"))]
#[allow(clippy::too_many_lines)]
fn cluster_canonicals_chars_via_gpu(
    gpu: &Gpu,
    chars: &[Vec<char>],
    threshold: f64,
    delta: f64,
) -> Vec<(Vec<usize>, f64)> {
    use rayon::prelude::*;

    use crate::gpu::CorpusGpu;
    use crate::{assemble, char_counts, quick_ratio_counts, real_quick_ratio};

    let n = chars.len();
    if n < 2 {
        return Vec::new();
    }

    // Prebuild every string's SAM (rayon-parallel).
    let sams: Vec<crate::gestalt::Sam> =
        chars.par_iter().map(|c| crate::gestalt::build_sam(c)).collect();

    // ASCII byte view of each string for the GPU corpus arena. We already enforced ASCII at
    // the call site (`Rationer::cluster_canonicals_chars`), so this cast is exact.
    let bytes: Vec<Vec<u8>> = chars.iter().map(|c| c.iter().map(|&ch| ch as u8).collect()).collect();
    let byte_refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let corpus = CorpusGpu::build(gpu, &byte_refs, &sams);

    // CPU-side filter: length blocking + quick_ratio. Produces candidate (i, j) pairs with i < j.
    // Length-sorted outer loop + break-on-length-bound is the same shape as the canonical
    // qualifying_pairs path, so we don't visit any pair we'd skip there.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| chars[i].len());
    let counts: Vec<Vec<(char, u32)>> = chars.par_iter().map(|c| char_counts(c)).collect();
    let candidates: Vec<(u32, u32)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|p| {
            let i = order[p];
            let mut local: Vec<(u32, u32)> = Vec::new();
            for &j in &order[p + 1..] {
                if real_quick_ratio(&chars[i], &chars[j]) < threshold {
                    break;
                }
                if quick_ratio_counts(&counts[i], &counts[j], chars[i].len() + chars[j].len())
                    < threshold
                {
                    continue;
                }
                // Production gestalt_edge(lo, hi, sam_b=sam_hi) — replicate ordering.
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                #[allow(clippy::cast_possible_truncation)]
                local.push((lo as u32, hi as u32));
            }
            local
        })
        .collect();

    if candidates.is_empty() {
        return assemble(n, Vec::new(), chars, &sams);
    }

    // GPU pass: matching_stats kernel + CPU `gestalt_edge_with_ms` per pair. (Stage 4d/4f
    // full_ratio kernel removed — lost 3× against this path on every canonical-Python corpus;
    // see `src/new/PERF_MAP.md`.)
    let pairs_for_gpu: Vec<(u32, u32)> = candidates.iter().copied().collect();
    let flat = gpu.matching_stats_batched_flat(&corpus, &pairs_for_gpu);
    let fstate_all = flat.fstate_all();
    let fmatch_all = flat.fmatch_all();
    let edges: Vec<(usize, usize, f64)> = (0..pairs_for_gpu.len())
        .into_par_iter()
        .filter_map(|slot| {
            let orig = flat.pair_orig_idx[slot] as usize;
            let (a_idx, b_idx) = candidates[orig];
            let lo_state = flat.out_offsets[slot] as usize;
            let hi_state = flat.out_offsets[slot + 1] as usize;
            let fstate = &fstate_all[lo_state..hi_state];
            let fmatch = &fmatch_all[lo_state..hi_state];
            let ratio = crate::gestalt::gestalt_edge_with_ms_delta(
                &chars[a_idx as usize],
                &chars[b_idx as usize],
                &sams[b_idx as usize],
                fstate,
                fmatch,
                threshold,
                delta,
            )?;
            Some((a_idx as usize, b_idx as usize, ratio))
        })
        .collect();

    assemble(n, edges, chars, &sams)
}

// SAFETY: All owned fields (Gpu, BoostGuard) are Send+Sync (Gpu wraps Metal handles documented
// thread-safe; BoostGuard owns a kernel object referenced by id). No interior mutability.
#[cfg(all(feature = "gpu", target_os = "macos"))]
unsafe impl Send for Rationer {}
#[cfg(all(feature = "gpu", target_os = "macos"))]
unsafe impl Sync for Rationer {}

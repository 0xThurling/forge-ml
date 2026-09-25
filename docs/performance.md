# performance — aiming at PyTorch-class throughput

"PyTorch-level performance" is a target, not a slogan. This document defines
it precisely, states the levers that get there, and specifies how every claim
is measured. The rule from the rest of the stack holds: **a change that is not
measured is not an optimization**, and a regression is a failing gate.

Depends on: everything. This is a cross-cutting program, not a layer.

Files: `bench/` (one file per module), `perf/` (counters and timers),
`bench/compare/` (PyTorch reference scripts).

**ForgeFP usage at a glance**

| Need | ForgeFP |
|---|---|
| Timers | `fp::Stopwatch`, `fp::now_seconds` |
| Threads | `fp::par_for`, `fp::par_map`, `fp::ThreadPool` |
| SIMD | `fp::simd` (`map_inplace`, `map_to`, `dot`, `reduce`, `axpy_inplace`) |
| Blocked kernels | `fp::linalg` (`matmul`, `batched_matmul`) |
| Memory | `fp::Buffer`, `fp::Arena`, `with_buffer` |
| Benchmarks | Forge's `forge bench` (Google Benchmark, `--save`/`--compare`) |

---

## What "PyTorch-level" means here

Three workload classes, three honest targets, all on the reference machine and
all *end-to-end* (a training step, not a bare GEMM):

| Class | Reference | Target |
|---|---|---|
| Classical models (linear/logistic/tree/ensemble on ≤ 100k × 100) | PyTorch CPU (float32) | **≤ 2× slower** |
| Deep models (MLP/CNN/LSTM/attention forward+backward) | PyTorch CPU (float32) | **≤ 3× slower** |
| Covered GPU kernels (matmul, softmax, axpy, conv, attention) | PyTorch CUDA on the RTX 3060 | **≤ 3× slower** |

The comparison is made with `bench/compare/` (below). Beating PyTorch is not
the goal; being in the same order of magnitude *without* BLAS/cuDNN is, and
the gap is documented rather than hidden.

**Escalation rule.** If a budget cannot be met without a vendor kernel
(BLAS/cuBLAS/cuDNN), the decision goes to ForgeFP as a *general* capability
(an opt-in BLAS-backed `fp::linalg` backend), never into `ml/`. The default
build stays dependency-free; the opt-in path is a documented flag, exactly
like the SYCL tier.

## Precision policy

The stack is `double` by default because correctness tests compare against
hand-computed values; PyTorch is `float32`. Performance parity requires both:

```cpp
namespace forgeml {
#if defined(FORGEML_SINGLE_PRECISION)
using Scalar = float;
#else
using Scalar = double;
#endif
}
```

Rules:

- **Kernels are templated on `Scalar`** (`Vector<Scalar>`, `Matrix<Scalar>`);
  no kernel hard-codes `double`.
- **Tests run in `double`** (the default); **benchmarks run both** and report
  which was used.
- Reductions accumulate in `double` even in a `float` build (`fp::linalg`
  already does this for means/variances) — a documented exception to "templated
  everywhere", because it is the difference between matching and not matching
  a reference.
- Mixed precision (fp16/bf16) is **not** in scope until ForgeFP has a half
  type; `inference/quantize.hpp` covers int8/int4 in the meantime.

## The levers, in order of measured impact

1. **No allocation in hot loops.** Every training step must reuse buffers
   (`fp::Buffer`, `fp::Arena`, `with_buffer`). `perf/alloc_counter.hpp` counts
   allocations; a test asserts a training step performs **zero** after warm-up.
2. **Blocking and SIMD.** `fp::linalg::matmul` is 4×4-blocked and vectorized;
   `fp::simd` for elementwise chains. New kernels benchmark against the naive
   loop and must state the measured factor.
3. **Threading.** `fp::par_for` over independent work (batches, output rows,
   ensemble members); the policy is "parallelize the outer loop only", and the
   thread count comes from the caller, never a hidden global.
4. **Fusion.** Elementwise chains (`relu(linear(x))`) run in one pass over the
   buffer instead of one pass per operation; `nn` layers expose fused forward
   paths where the benchmark shows a win.
5. **Memory layout.** Row-major contiguous matrices, weights transposed once at
   fit time, KV cache pre-allocated, batches packed without padding
   (`inference/batch.hpp`).
6. **Precision.** The `Scalar` switch above; float32 doubles throughput on
   SIMD-heavy kernels.
7. **Data feeding.** Batches arrive at device speed via dsio
   (`(prefetch + 2) × batch_bytes` resident, shard-size rule in
   [`dsio/docs/usage.md`](../../dsio/docs/usage.md)) — a slow loader caps
   everything else.

## Measurement

### The harness

```bash
forge bench                        # every bench/ file, Google Benchmark
forge bench --benchmark_filter=BM_Linear
forge bench --save                 # baseline
forge bench --compare --fail-over 5   # a >5% regression fails
```

Conventions:

- Benchmarks use **real time** (`UseRealTime()`) for I/O and mixed workloads,
  and report `SetBytesProcessed`/`SetItemsProcessed` so throughput is explicit.
- Each benchmark states its **shape** in the label (batch, features, sequence
  length, image size) — a number without a shape is not a result.
- The machine state (load average, precision, thread count) is recorded with
  the run, not assumed.

### The PyTorch comparison

`bench/compare/` holds small Python scripts (PyTorch is **not** a build
dependency) that run the same workloads and emit JSON:

```text
bench/compare/mlp_cpu.py      # torch, float32, same shapes
bench/compare/conv_cpu.py
bench/compare/attention_cpu.py
bench/compare/kernels_cuda.py # the RTX 3060 reference, when available
```

A `bench/compare/report.py` joins the two JSON files and prints the ratio
table that goes into the performance log below. The scripts pin threads
(`torch.set_num_threads`) and seeds so the comparison is reproducible.

### Profiling

- `perf/timer.hpp` — named phase timers over `fp::Stopwatch` (forward,
  backward, optimizer, data wait), so a slow step is attributable.
- `perf/alloc_counter.hpp` — allocation and byte counters for hot-loop tests.
- `perf/memory.hpp` — resident bytes of parameters, activations, caches and
  batch buffers; a model's memory is reported, not estimated.
- `perf`/hardware counters are **not** available on the reference machine
  (WSL2); wall time, CPU time and allocation counts are the evidence.

## Budgets and gates

Each model stage (5–8, 11–14) adds a benchmark and a budget:

| Stage | Benchmark | Budget (float32, CPU) |
|---|---|---|
| 5–6 | linear/logistic fit on 100k × 100 | ≤ 2× PyTorch |
| 7–8 | tree/forest fit and predict | ≤ 2× PyTorch (sklearn-scale) |
| 11 | MLP step, 256 × 256 × 10 layers | ≤ 3× PyTorch |
| 12 | CNN step, 64×32×32; LSTM step, 128×64 | ≤ 3× PyTorch |
| 13 | attention fwd+bwd, 8 heads, seq 128 | ≤ 3× PyTorch |
| 14 | char GPT step, 4 layers, seq 128 | ≤ 3× PyTorch |
| 15 | the covered kernels on the RTX 3060 | ≤ 3× PyTorch CUDA |
| E10 | decode throughput, KV cache on | ≥ 2× the uncached path |

A stage is not done until its benchmark runs, its budget holds, and the number
is recorded in the log below. A regression over 5% fails `forge bench
--compare`.

## Performance log

| Date | Workload | Shape | Precision | Result | vs reference |
|---|---|---|---|---|---|
| — | filled in from Stage 11 on; the format is fixed now | | | | |

## Gate

Every stage that touches a hot path has a benchmark with a shape, a budget and
a recorded number; a training step allocates nothing after warm-up; the
PyTorch ratio table exists for the reference workloads; a >5% regression fails
the bench gate.

# GPU acceleration (the ForgeFP SYCL tier)

ForgeML's hot paths are `matmul` (every dense layer), row-wise softmax
(attention), `axpy` (optimizer updates), and elementwise loss/activation math.
ForgeFP ships a GPU tier for exactly those kernels — `fp/gpu.hpp`, opt-in, with
a CPU fallback — and it is measured on the reference machine (RTX 3060 / WSL2)
in [`fp/GPU.md`](../../fp/GPU.md#measured-crossovers-rtx-3060-wsl2).

This document specifies **which ForgeML operations may use it, where data
crosses the bus, and what the tests must prove**. It is the GPU counterpart of
the "ForgeFP first" rule: ForgeML never writes a kernel.

## Scope

| In scope | Out of scope (for now) |
|---|---|
| Device-resident parameters and activations for `nn` | Distributed / multi-GPU execution |
| `matmul`, `batched_matmul`, `softmax_rows_wg` | Mixed precision (`fp::gpu` has no `half`/`bfloat16` yet — fp phase 5) |
| `axpy_inplace` for optimizers | Reverse-mode autodiff *on* the device |
| Elementwise losses/activations, `reduce`/`dot` metrics | cuBLAS-class tuning, autotuning, kernel fusion |
| Optional device-resident `Tensor` storage | Training large models (this is a correctness-first tier) |

## The rule: one API, two execution paths

The GPU is an **execution detail behind the same API**. Concretely:

1. Every accelerated operation has a CPU path that stays the default and stays
   tested. `fp::gpu` already falls back to the CPU when there is no SYCL
   toolchain, so a GPU-enabled build still runs on a laptop.
2. Selection is explicit at runtime, never implicit: a caller asks
   `ml::gpu::should_use(n)` (below) or constructs the device variant.
3. **Values match**: elementwise kernels are bit-identical to the CPU; matmul,
   softmax and reductions are compared with `fp::approx_equal` and the
   documented tolerance (the same rule `fp::simd` follows).
4. **If a kernel is missing, it goes into ForgeFP** — never into `ml/`. The
   fp boundary rule (numbers, not ML semantics) still holds.

## What runs on the device

Thresholds are the measured crossovers on the reference machine, where the CPU
path wins below and the GPU wins above. They are *data points*, not constants:
`ml/gpu/dispatch.hpp` keeps them in one place so they can be re-measured.

| ForgeML site | `fp::gpu` kernel | Crossover |
|---|---|---|
| `nn/linear.hpp` forward + backward | `matmul` | 128² (wins at every measured size) |
| `nn/attention.hpp`, `multi_head_attention.hpp` | `matmul`, `batched_matmul`, `transpose` (for `Q @ Kᵀ`) | 128² per score matrix |
| `nn/attention.hpp` causal mask | `transform_inplace_indexed` | per element; fuses into the scores buffer |
| `nn/attention.hpp` weight normalization | `softmax_rows_wg` | ~1 MiB (rows × cols) |
| `optim/gradient_descent.hpp`, `sgd.hpp` | `axpy_inplace` | 16 MiB of parameters, **already resident** |
| `optim/adam.hpp` | `zip_transform_inplace`, `zip3_transform_inplace` (moments, bias correction, update) | same residency rule |
| `math/losses.hpp`, `nn/activation.hpp` | `transform_inplace`, `map_to` | 16 MiB per tensor |
| `metrics/*`, `utils/gradient_check.hpp` | `reduce`, `dot` (with a reused `Scratch`) | 16 MiB |
| `data/scaling.hpp`, `math/stats.hpp` | `row_means`, `col_sums`, `add_row_broadcast` | 1 MiB |
| `nn/layer_norm.hpp`, `transformer_block.hpp` | `row_means`, `add_row_broadcast`, `transform_inplace` | 1 MiB per tensor |

Two consequences worth stating up front:

- **Small models stay on the CPU.** A 2-layer MLP on XOR, or any model whose
  tensors are below a megabyte, is faster on the CPU/SIMD path — the launch and
  transfer costs (0.14–0.36 ms per synchronous call) dominate. The GPU tier
  exists for the transformer/attention sizes in stages 12–14.
- **`reduce`/`dot` must be called with a reused `Scratch`.** Without one they
  allocate a partials buffer per call and lose to the CPU by 12× at 4M elements
  (fp/GPU.md). The metrics/loss paths own one scratch per thread.

## Device residency

Two options were considered:

**(a) A device mirror inside `Tensor`.** `Tensor<T>` grows a
`std::optional<fp::gpu::Buffer<T>>` plus dirty flags, and every operation
decides where to run. Fewer types, but `core/` becomes GPU-aware, and every
operation needs transfer logic.

**(b) A separate device type (chosen).** `ml/gpu/device_tensor.hpp` owns an
`fp::gpu::Buffer<T>` plus a `Shape`, with explicit `to_device()` / `to_host()` /
`is_resident()`; the CPU `Tensor` is untouched. Layers get device variants
(`ml/gpu/linear.hpp`, `attention.hpp`, `optim.hpp`) that mirror the CPU ones
and are tested against them.

(b) is the fp-consistent choice: opt-in, no cost when unused, and the CPU code
path cannot regress. The integrated design can come later if the mirror
bookkeeping proves worth it.

Rules for residency:

- **Parameters stay resident** across steps. `DeviceTensor` is created once
  (`to_device()`), updated in place by the optimizer (`axpy_inplace`), and only
  leaves the device for checkpointing.
- **Activations stay resident** within a step (forward → loss → backward →
  update). A device network is a sequence of device calls; no per-layer
  round-trip.
- **Batches cross once per step** (`copy_from` with a pinned `HostBuffer`, or
  pageable `from_host` for small batches). The dataset stays on the host.
- **Scalars cross once per step**: a loss/metric is `reduce`d on the device and
  read back into a `HostBuffer` (4–8 bytes), not per-element.

## API sketch

```cpp
namespace ml::gpu {

// True when the process was built with SYCL and the selected device is usable.
bool available();

// Data-driven dispatch: true when a tensor of `n` elements (or an m×k×n
// matmul) is past the measured crossover.
bool should_use(std::size_t n);
bool should_use_matmul(std::size_t m, std::size_t k, std::size_t n);

// Device tensor: shape + fp::gpu::Buffer, explicit transfers.
template <class T> class DeviceTensor {
public:
  static fp::Result<DeviceTensor<T>> from_host(Tensor<T> const &);
  fp::Result<Tensor<T>> to_host() const;
  fp::Result<void> copy_from(HostBuffer<T> const &);   // pinned path
  std::size_t size() const;
  Shape const &shape() const;
  fp::gpu::Buffer<T> &buffer();
};

} // namespace ml::gpu
```

`DeviceTensor` is deliberately thin: it adds shape + ownership, nothing else.
All math goes to `fp::gpu`.

## Testing

| Level | Where | What it proves |
|---|---|---|
| Compile + fallback | CI, no toolchain | the device code path compiles and produces CPU values (fp's fallback) |
| CPU SYCL stub | CI, `-I fp/test/support` | the SYCL branch type-checks and runs sequentially (fp's stub harness) |
| Device vs CPU, per op | `test/gpu_test.cpp` (device-gated) | `DeviceTensor` round-trips, matmul/softmax/optimizer match the CPU within tolerance |
| End-to-end | `test/gpu_training_test.cpp` (device-gated) | a fixed-seed MLP/attention trains to the same loss as the CPU within tolerance |
| Benchmarks | `bench/gpu_bench.cpp` | the thresholds in this document are re-measured, not assumed |

The device-gated tests follow fp's pattern: they check
`fp::gpu::usable()` and skip (not fail) without a device, so the suite is green
everywhere.

Tolerances: elementwise = bit-identical; `matmul`/`softmax_rows_wg`/`reduce` =
`fp::approx_equal` with the fp-documented tolerance; training = the CPU and GPU
runs agree on the loss trajectory to within a documented tolerance over N steps
(they will not be bit-identical because reductions reorder).

## Cost model (reference machine)

The numbers the dispatch thresholds come from (fp/GPU.md, phase 1b/2/3):

| Cost | Value |
|---|---|
| Synchronous kernel call (launch + wait) | 0.14–0.36 ms |
| H2D / D2H, pageable | 4.5 / 5.3 GB/s |
| H2D / D2H, pinned (`HostBuffer`) | 12.4 / 12.1 GB/s |
| Elementwise kernel | 78 GB/s |
| `matmul` | 532 GFLOP/s at 1024² (CPU: 24) |
| `softmax_rows_wg` | 44 GB/s effective |
| `reduce`/`dot` with scratch | 54 / 35 GB/s |

A training step on a 12-layer transformer with a 512-token batch is ~0.5 GFLOP
of matmul — 1 ms of GPU math against ~20 ms of CPU math, plus one batch
transfer (~4 MB pinned ≈ 0.3 ms). That is the case this tier is for.

**Synchronization is the other half of the budget.** Measured on the reference
machine: 50 small kernels cost 0.056–0.076 ms each when every call waits, but
0.016–0.017 ms each when submitted together and waited once. A tiny char GPT
(~2 blocks, d=128, 32 tokens) does ~50 kernels and ~0.14 ms of actual GPU math
per step, so a per-call-wait design spends ~20× the compute on synchronization.
Two consequences for this tier:

1. **Batch the submissions.** A device training step should submit all of its
   kernels and wait once per step (ForgeFP phase 4's async API), not wait per
   kernel. Until that exists, keep tiny models on the CPU — the dispatch
   thresholds below assume it.
2. **Fewer, larger kernels beat many small ones.** Prefer one `matmul` over a
   loop of `matvec`s, and fuse the scale + mask + softmax passes where the
   shapes allow.

## Roadmap placement

The CPU stack must exist first. Concretely:

- **After stage 4** (`optim`): the optimizer is where `axpy` first appears, and
  a device optimizer is testable against `w - lr * g` without any network.
- **After stage 11** (`nn` core): `DeviceTensor` + a device `Linear` +
  gradient checks against the CPU.
- **After stage 13** (attention/transformers): the case that justifies the tier
  — device attention with `softmax_rows_wg` and `batched_matmul`.
- **Stage 15** in [roadmap.md](roadmap.md) collects the work with its gates.

Do not start the GPU tier before stage 11's CPU gate passes: the CPU path is
the reference every device test compares against.

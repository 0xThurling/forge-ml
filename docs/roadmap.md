# Roadmap

Build order follows one rule: **never implement a file before the files it
depends on exist**. Each stage has a deliverable, tests, and a gate. Do not
start the next stage until the gate passes.

Legend: `[ ]` todo, `[~]` in progress, `[x]` done. Copy this file or keep it
updated as the project progresses.

The GPU tier (stage 15) is opt-in and comes last: it is specified in
[gpu.md](gpu.md) and depends on the CPU stack being green, because every device
test compares against the CPU reference.

## Stage −1 — ForgeFP (already landed)

The general infrastructure this stack consumes is implemented and tested in
ForgeFP; see [`fp/EXTENSIONS.md`](../../fp/EXTENSIONS.md) and the
[`.docs/`](../../fp/.docs/README.md) reference. Do **not** reimplement any of
it here.

- [x] `fp/inplace.hpp`, `fp/scope.hpp`, `fp/memory.hpp`, arena extensions
- [x] `fp/views.hpp` / `fp/grid.hpp` extensions
- [x] `fp/ops.hpp` / `fp/numerics.hpp`
- [x] `fp/linalg.hpp` + `fp/simd.hpp` extensions
- [x] `fp/random.hpp`, `fp/str`/`fp/io`/`fp/time.hpp` extensions
- [x] `fp/concurrent.hpp` extensions
- [x] `fp/serialize.hpp`, `fp/autodiff.hpp` (opt-in)

## Stage 0 — Project scaffolding

Deliverables:

- [ ] `forge.lua` with ForgeFP + GoogleTest dependencies, `testing = true`
- [ ] `src/main.cpp` that compiles and prints the library version string
- [ ] `test/smoke_test.cpp` (`EXPECT_TRUE(true)`)
- [ ] `docs/` in place (this directory)

Gate: `forge build` and `forge test` (or `ctest`) both succeed.

## Stage 1 — Core containers

Files:

- [ ] `src/utils/assertions.hpp` — `ML_ASSERT`, `MlError`, tolerance helpers
      (built on `fp::approx_equal`)
- [ ] `src/core/shape.hpp` — `Shape`, compatibility helpers returning
      `fp::Result`
- [ ] `src/core/vector.hpp` — `Vector<T>` (contiguous, STL range interface, so
      fp ranges/simd apply directly)
- [ ] `src/core/matrix.hpp` — `Matrix<T>` (grid-backed: `std::vector<Vector<T>>`,
      so `fp::linalg`/`fp::grid` apply directly)
- [ ] `src/core/tensor.hpp` — `Tensor<T>` (N-d, `fp::Buffer<T>` storage,
      strides)

ForgeFP usage: `fp::Buffer`, `fp::Result`, `fp::fail`, `fp::map`, `fp::to_vector`,
`fp::fold_left`, `fp::sum`, `fp::dot`, `fp::approx_equal`.

Tests:

- [ ] `test/core_test.cpp` — indexing, bounds, reshape, copy/move, shape checks
- [ ] `test/tensor_test.cpp` — `at` matches flat indexing on a 3-D case;
      `fp::for_each`/`fp::transform_inplace` work on the buffer

Gate: containers round-trip, `Matrix` rows/cols correct, `fp::sum`/`fp::fold_left`
over each container produces the hand-computed value.

## Stage 2 — Math layer (the fp remainder)

Files:

- [ ] `src/math/activations.hpp` — ML-specific `gelu`, `leaky_relu`, and
      vectorized wrappers; `sigmoid`/`relu`/`softmax`/`log_softmax` are
      `fp::numerics`
- [ ] `src/math/losses.hpp` — MSE, MAE, BCE, CCE, hinge, L1/L2 penalties,
      implemented with fp primitives
- [ ] `src/math/stats.hpp` — impurity (`gini`, `entropy`, `information_gain`,
      `variance_reduction`), class counts, `col_variances`/`col_stddevs`;
      `mean`/`variance`/`argmax` are `fp::linalg`

ForgeFP usage: `fp::softmax`, `fp::log_softmax`, `fp::logsumexp`, `fp::sigmoid`,
`fp::relu`, `fp::clamp`, `fp::zip_with`, `fp::fold_left`, `fp::sum`,
`fp::dot`, `fp::transform_inplace`, `fp::central_difference`.

Tests:

- [ ] `test/activations_test.cpp` — known values; wrappers equal fp results
- [ ] `test/losses_test.cpp` — hand-computed loss + gradient values; every
      gradient matches `fp::central_difference` at `h = 1e-7` within `1e-6`
- [ ] `test/stats_test.cpp` — entropy of a fair coin = 1; Gini of a pure node = 0

Gate: all math tests pass; every loss gradient is finite-difference checked.

## Stage 3 — Data layer

Files:

- [ ] `src/data/dataset.hpp` — `Dataset<Y>`, validation with `fp::Validation`
- [ ] `src/data/split.hpp` — train/test/val split, k-fold indices (`fp::Rng`)
- [ ] `src/data/scaling.hpp` — `StandardScaler`, `MinMaxScaler` (`fp::col_means`,
      `fp::map2d`, `fp::inplace`)
- [ ] `src/data/encoding.hpp` — `LabelEncoder`, one-hot helpers (`fp::str`)
- [ ] `src/data/dataloader.hpp` — mini-batch iteration with shuffle
      (`fp::Rng::shuffle`, `fp::views::chunk`)

ForgeFP usage: `fp::Rng`, `fp::Validation`, `fp::traverse`, `fp::views::chunk`,
`fp::map_to`, `fp::read_lines`, `fp::str::split_any`,
`fp::str::parse_numbers`, `fp::col_means`, `fp::map2d_indexed`.

Tests:

- [ ] `test/data_test.cpp` — split fractions, no overlap, k-fold covers once
- [ ] `test/scaling_test.cpp` — inverse transform round-trip; zero variance safe
- [ ] `test/encoding_test.cpp` — round-trip labels; one-hot sum = 1
- [ ] `test/dataloader_test.cpp` — batch count, remainder handling, shuffle order differs

Gate: a CSV-like table can be split, scaled, encoded, and iterated in batches
with reproducible results.

## Stage 4 — Metrics and optimizers

Files:

- [ ] `src/metrics/regression.hpp` — mse, rmse, mae, r2_score
- [ ] `src/metrics/confusion_matrix.hpp` — binary/multiclass counts
- [ ] `src/metrics/classification.hpp` — accuracy, precision, recall, f1, specificity
- [ ] `src/optim/optimizer.hpp` — optimizer interface over `Vector`/`Matrix`
- [ ] `src/optim/gradient_descent.hpp` — batch GD (+ momentum)
- [ ] `src/optim/sgd.hpp` — mini-batch SGD (+ momentum, weight decay)
- [ ] `src/optim/adam.hpp` — Adam, AdamW
- [ ] `src/optim/lr_scheduler.hpp` — constant, step, linear, cosine

ForgeFP usage: `fp::mean/variance/argmax`, `fp::zip_with`, `fp::fold_left`,
`fp::dot`, `fp::axpy_inplace`, `fp::zip_transform_inplace`,
`fp::for_each`, `fp::approx_equal`.

Tests:

- [ ] `test/metrics_test.cpp` — hand-computed values incl. zero-division cases
- [ ] `test/optim_test.cpp` — one step matches `w - lr * grad`; Adam bias
      correction; `axpy_inplace` matches a scalar loop

Gate: one GD step on `f(w) = w²` halves `w` with `lr = 0.5`; metrics match
hand-computed confusion matrices.

## Stage 5 — First model

Files:

- [ ] `src/model/model.hpp` — `Regressor` / `Classifier` interfaces
- [ ] `src/model/linear_regression.hpp` — GD + optional closed form, L1/L2

ForgeFP usage: `fp::matvec`, `fp::matmul`, `fp::transpose`, `fp::solve`,
`fp::zip_with`, `fp::scale`, `fp::transform_inplace`, `fp::dot`.

Tests:

- [ ] `test/linear_regression_test.cpp` — recovers `y = 3x + 2`; R² > 0.99
- [ ] `test/model_test.cpp` — interface conformance, `fit` on empty data fails

Demo:

- [ ] `src/main.cpp` trains linear regression on synthetic data and prints MSE/R²

Gate (milestone M1): linear regression on `y = 3x + 2 + noise` reaches
`R² > 0.99` on a held-out split. **Stop and fix the foundation if it does not.**

## Stage 6 — Second model

Files:

- [ ] `src/model/logistic_regression.hpp` — sigmoid + BCE, L1/L2, threshold

ForgeFP usage: `fp::sigmoid`, `fp::matvec`, `fp::transpose`, `fp::matmul`,
`fp::transform_inplace`, `fp::central_difference` (test).

Tests:

- [ ] `test/logistic_regression_test.cpp` — separable 2-D blobs reach > 95% accuracy
- [ ] gradient check on the loss w.r.t. weights

Gate (milestone M2): logistic regression separates two Gaussian blobs and the
decision boundary is stable across seeds.

## Stage 7 — Non-gradient models

Files:

- [ ] `src/model/naive_bayes.hpp` — Gaussian NB (Multinomial later)
- [ ] `src/model/decision_tree.hpp` — CART classifier (Gini/entropy)

ForgeFP usage: `fp::mean/variance`, `fp::softmax`, `fp::logsumexp`,
`fp::log`, `fp::group_by`, `fp::fix` (recursive tree build), `fp::sort_by`,
`fp::views::enumerate`, `fp::argmax`.

Tests:

- [ ] `test/naive_bayes_test.cpp` — known posteriors; epsilon handles zero variance
- [ ] `test/decision_tree_test.cpp` — XOR-like split; max_depth respected; pure leaf stops

Gate (milestone M3): on an iris-like 3-class toy dataset, both models reach
> 90% accuracy with deterministic splits.

## Stage 8 — SVM

Files:

- [ ] `src/model/svm.hpp` — linear SVM, labels in {-1, +1}, hinge loss + L2

ForgeFP usage: `fp::matvec`, `fp::zip_transform_inplace`,
`fp::scale`, `fp::fold_left`, `fp::axpy_inplace`.

Tests:

- [ ] `test/svm_test.cpp` — margin > 0 on separable data; subgradient step matches formula

Gate: linear SVM reaches 100% on a linearly separable toy set and the margin
increases over epochs.

## Stage 9 — Evaluation layer

Files:

- [ ] `src/eval/scoring.hpp` — metric adapters, maximize/minimize
- [ ] `src/eval/cross_validation.hpp` — k-fold + stratified CV driver
- [ ] `src/eval/grid_search.hpp` — Cartesian hyperparameter search over CV

ForgeFP usage: `fp::cartesian_product`, `fp::sort_by`, `fp::views::enumerate`,
`fp::str::join`, `fp::Rng`, `fp::Result`.

Tests:

- [ ] `test/eval_test.cpp` — folds partition data once; CV mean/std correct;
      grid search picks the known-best config on a synthetic problem

Gate (milestone M4): grid search over `(learning_rate, lambda)` for logistic
regression finds a config within 2% of the best achievable accuracy on a fixed
dataset.

## Stage 10 — Utils finish line

Files:

- [ ] `src/utils/logging.hpp` — levels, logger, epoch callback (`fp::time`,
      `fp::str`)
- [ ] `src/utils/serialization.hpp` — checkpoint schema over `fp::serialize`
      and `fp::io`
- [ ] `src/utils/gradient_check.hpp` — finite-difference harness over
      `fp::central_difference` / `fp::ad::derivative`

ForgeFP usage: `fp::Stopwatch`, `fp::now_seconds`, `fp::str::to_string`,
`fp::to_text`/`from_text`, `fp::to_bytes`/`from_bytes`, `fp::write_bytes`,
`fp::read_bytes`, `fp::ensure_directory`, `fp::central_difference`,
`fp::derivative`, `fp::approx_equal`.

Tests:

- [ ] `test/serialization_test.cpp` — round-trip equality within `1e-12`
- [ ] `test/logging_test.cpp` — level filtering, no output in library mode

Gate: every model can be saved and reloaded with identical predictions.

## Stage 11 — Neural network core

Files:

- [ ] `src/nn/parameter.hpp` — `Parameter` (value + grad)
- [ ] `src/nn/layer.hpp` — layer interface (forward/backward/parameters)
- [ ] `src/nn/linear.hpp` — dense layer
- [ ] `src/nn/activation.hpp` — ReLU/Sigmoid/Tanh/Softmax layers (fp numerics)
- [ ] `src/nn/network.hpp` — sequential container + backward pass
- [ ] `src/nn/autodiff.hpp` — optional: the scalar `Value` engine (teaching example)

ForgeFP usage: `fp::matmul`, `fp::add_row_broadcast`, `fp::softmax_rows`,
`fp::dot`, `fp::map_to`, `fp::par_for`, `fp::Buffer`,
`fp::central_difference` (gradient checks).

Tests:

- [ ] `test/nn_core_test.cpp` — XOR MLP reaches 100% with fixed seed
- [ ] gradient check for every layer and the whole network

Gate (milestone M5): 2-layer MLP solves XOR; all gradient checks < `1e-6`.

## Stage 12 — Sequence and vision layers

Files:

- [ ] `src/nn/conv2d.hpp`, `src/nn/pooling.hpp`, `src/nn/flatten.hpp`
- [ ] `src/nn/rnn_cell.hpp`, `src/nn/rnn.hpp`
- [ ] `src/nn/lstm_cell.hpp`, `src/nn/lstm.hpp`
- [ ] `src/nn/embedding.hpp`

ForgeFP usage: `fp::matmul`, `fp::add_row_broadcast`, `fp::sigmoid`,
`fp::clamp`, `fp::windows2d`, `fp::Buffer`, `fp::par_for`.

Tests:

- [ ] `test/conv_test.cpp` — known 3x3 kernel convolution; gradient check
- [ ] `test/rnn_test.cpp` — memory of a toy sequence; BPTT gradient check
- [ ] `test/embedding_test.cpp` — lookup + scatter-add backward

Gate (milestone M6): CNN classifies a tiny synthetic image task > 95%; LSTM
learns a delayed-copy/parity toy task.

## Stage 13 — Attention and transformers

Files:

- [ ] `src/nn/attention.hpp`, `src/nn/multi_head_attention.hpp`
- [ ] `src/nn/positional_encoding.hpp`, `src/nn/feed_forward.hpp`
- [ ] `src/nn/layer_norm.hpp`, `src/nn/transformer_block.hpp`
- [ ] `src/nn/encoder_transformer.hpp`, `src/nn/decoder_transformer.hpp`

ForgeFP usage: `fp::matmul`, `fp::batched_matmul`, `fp::transpose`,
`fp::softmax_rows`, `fp::softmax`, `fp::mean/variance`,
`fp::map2d_indexed`, `fp::par_for`, `fp::Buffer`.

Tests:

- [ ] `test/attention_test.cpp` — softmax weights sum to 1; causal mask blocks future
- [ ] gradient checks for attention, LayerNorm, transformer block
- [ ] `test/transformer_test.cpp` — overfits a tiny copy task

Gate: a 2-block decoder transformer overfits a 32-token sequence to near-zero
loss with a fixed seed.

## Stage 14 — Tiny LLM

Files:

- [ ] `src/llm/tokenizer.hpp` — char-level first, then byte-pair merges
- [ ] `src/llm/dataset.hpp` — token stream, block sampling, input/target shift
- [ ] `src/llm/causal_lm.hpp` — decoder-only model, weight tying
- [ ] `src/llm/trainer.hpp` — AdamW, warmup + cosine, grad clipping, checkpoints
- [ ] `src/llm/sampling.hpp` — greedy, temperature, top-k, top-p
- [ ] `src/llm/checkpoint.hpp` — save/load weights + config + tokenizer

ForgeFP usage: `fp::Rng` (`rng.categorical`), `fp::softmax`, `fp::logsumexp`,
`fp::sort_by`, `fp::scan`, `fp::Stopwatch`, `fp::to_text`/`from_text`,
`fp::write_bytes`, `fp::ensure_directory`, `fp::str::split_any`,
`fp::transform_inplace`, `fp::par_for`.

Tests:

- [ ] `test/tokenizer_test.cpp` — encode/decode round-trip on unicode input
- [ ] `test/llm_test.cpp` — perplexity decreases on a small text corpus; sampling
      produces tokens from the vocabulary; KV cache equals full recompute

Gate (milestone M7): train a character-level GPT on a small text file until it
generates recognizable words/structure; perplexity drops by at least 2x from
initialization.

## Stage 15 — GPU acceleration (opt-in)

Spec: [gpu.md](gpu.md). Prerequisite: stages 4 and 11–13 are green on the CPU —
the CPU path is the reference every device test compares against, and it stays
the default.

Files:

- [ ] `src/gpu/dispatch.hpp` — `available()`, `should_use(n)`,
      `should_use_matmul(m, k, n)`; the measured thresholds in one place
- [ ] `src/gpu/device_tensor.hpp` — `DeviceTensor<T>`: shape +
      `fp::gpu::Buffer<T>`; `from_host` / `to_host` / `copy_from` (pinned
      staging) / `is_resident()`
- [ ] `src/gpu/optim.hpp` — device optimizer steps over resident parameters
      (`fp::gpu::axpy_inplace`, `transform_inplace`)
- [ ] `src/gpu/linear.hpp` — device dense layer mirroring `nn/linear.hpp`
- [ ] `src/gpu/attention.hpp` — device scaled dot-product attention
      (`fp::gpu::matmul`, `softmax_rows_wg`)
- [ ] `test/gpu_test.cpp` — device-gated: round-trips, matmul/softmax/optimizer
      vs the CPU within tolerance
- [ ] `test/gpu_training_test.cpp` — device-gated end-to-end: a fixed-seed
      MLP/attention reaches the same loss as the CPU within tolerance

ForgeFP usage: `fp::gpu::Buffer`, `HostBuffer`, `Scratch`, `matmul`,
`batched_matmul`, `transpose`, `softmax_rows_wg`, `transform_inplace`,
`transform_inplace_indexed` (causal mask), `zip_transform_inplace` /
`zip3_transform_inplace` (AdamW), `axpy_inplace`, `reduce`, `dot`, `row_means`,
`add_row_broadcast`, `fp::approx_equal`.

Tests:

- [ ] device vs CPU per operation (skip without a device, never fail)
- [ ] one batch through pinned and pageable staging gives identical values
- [ ] the dispatch thresholds agree with `bench/gpu_bench.cpp` on the machine

Gate (milestone M8): with a device present, a 2-block decoder attention block
trained on the CPU and on the device with the same seed agrees within tolerance;
without a device, the whole suite still passes.

Effort: 2–3 days — the kernels already exist in ForgeFP, so this is storage,
dispatch, and tests.

## Milestones

| Milestone | After stage | Passing condition |
|---|---|---|
| M1 | 5 | Linear regression R² > 0.99 on held-out synthetic data |
| M2 | 6 | Logistic regression > 95% on separable blobs |
| M3 | 7 | NB and decision tree > 90% on a 3-class toy set |
| M4 | 9 | Grid search finds the known-best config within 2% |
| M5 | 11 | MLP solves XOR; every gradient check < 1e-6 |
| M6 | 12 | CNN > 95% on synthetic images; LSTM learns a toy sequence |
| M7 | 14 | Char GPT loss decreases; samples look like the corpus |
| M8 | 15 | CPU and GPU training agree within tolerance; the suite passes without a device |

## Effort guide

| Stage | Rough size | Depends on |
|---|---|---|
| 0–1 | 1 day | ForgeFP |
| 2 | 1 day | 1 (most math is fp) |
| 3 | 1–2 days | 1, 2 |
| 4 | 1–2 days | 2 |
| 5–6 | 2 days | 3, 4 |
| 7–8 | 2–3 days | 3, 4 |
| 9 | 1–2 days | 5–8 |
| 10 | 1 day | 5 |
| 11 | 2–3 days | 2, 4, 10 |
| 12 | 3–5 days | 11 |
| 13 | 3–5 days | 11, 12 |
| 14 | 2–4 days | 13 |
| 15 | 2–3 days | 4, 11–13 |

The estimates are lower than the original plan because ForgeFP already provides
linear algebra, numerics, randomness, serialization, autodiff, and the
zero-cost kernels.

## Working agreement for an implementing agent

1. Pick the first unchecked file in the earliest incomplete stage.
2. Read its module doc and the docs of its dependencies, **including the
   "ForgeFP usage" block**.
3. Implement the file exactly to the documented API, calling fp for everything
   fp provides.
4. Add the tests listed for that file.
5. Run the stage gate command; fix until green.
6. Check the file off in this roadmap before moving on.

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
- [ ] `src/data/shards.hpp` — disk-backed streams over **dsio** shards
      (CPU), with the GPUDirect seam noted for the GPU tier
      ([plan](../../dsio/docs/ml-integration.md),
      [API](../../dsio/docs/usage.md),
      [lesson](loading.md))

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

## Coverage expansion (E1–E11) — the comprehensive pass

Stages 0–15 cover the classical core, the neural stack and a tiny LLM. These
expansion stages fill the rest of the field. They are **independent of each
other** (E2 underpins E3–E9) and slot in after the stages named in their
headers; each ends with its own gate. Docs are written alongside each stage.

### E1 — Features and data preparation (after Stage 3)

Spec: [features.md](features.md).

Files:

- [ ] `src/features/impute.hpp` — mean/median/mode/constant/forward fill
- [ ] `src/features/binning.hpp` — equal-width, quantile, one-hot-of-bins
- [ ] `src/features/polynomial.hpp` — polynomial and interaction features
- [ ] `src/features/text.hpp` — bag-of-words, n-grams, TF-IDF, hashing
- [ ] `src/features/select.hpp` — variance threshold, mutual information, L1
      (with `model/linear`), recursive elimination
- [ ] `src/features/imbalance.hpp` — class weights, random over/under-sampling
      (SMOTE-lite: k-NN interpolation)
- [ ] `src/features/augment.hpp` — image flips/crops/rotation/color jitter,
      tabular noise; all `fp::Rng`-driven and deterministic
- [ ] `test/features_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::map2d`, `fp::col_means`, `fp::views::chunk`,
`fp::str::split_any`, `fp::sort_by`, `fp::Validation`.

Gate: an imbalanced, ragged, mixed-type table becomes a clean numeric matrix
with deterministic augmentation, and feature selection keeps a planted signal.

### E2 — Probability and statistics (after Stage 2)

Spec: [probability.md](probability.md).

Files:

- [ ] `src/prob/distributions.hpp` — Gaussian, Bernoulli, Categorical, Poisson,
      Exponential, Beta, Gamma, Dirichlet: `logpdf`, `pdf`, `cdf` (where
      closed-form), `mean`, `variance`, `sample(fp::Rng&)`
- [ ] `src/prob/info.hpp` — entropy, cross-entropy, KL, JS, mutual
      information, perplexity
- [ ] `src/prob/bayes.hpp` — conjugate updates (Beta–Bernoulli,
      Dirichlet–Categorical, Normal–Normal), MAP estimates
- [ ] `src/prob/sampling.hpp` — inverse-CDF, rejection, importance,
      Metropolis–Hastings, Gibbs
- [ ] `src/prob/stats.hpp` — MLE, bootstrap CIs, permutation tests, t/Welch,
      chi-square, KS
- [ ] `test/prob_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::numerics` (`logsumexp`), `fp::linalg`
(`mean`, `variance`, `dot`), `fp::Validation`, `fp::approx_equal`.

Gate: distribution moments match hand values; KL ≥ 0 with equality iff equal;
MCMC recovers a known posterior mean within tolerance.

### E3 — Nearest neighbors and density (after Stage 9)

Spec: [neighbors.md](neighbors.md).

Files:

- [ ] `src/neighbors/knn.hpp` — classification/regression, distance metrics
      (L1/L2/cosine), weighted voting, brute force + k-d tree
- [ ] `src/neighbors/kde.hpp` — Gaussian/Epanechnikov kernels, bandwidth rules
- [ ] `src/neighbors/lof.hpp` — local outlier factor
- [ ] `test/neighbors_test.cpp`

ForgeFP usage: `fp::sort_by_cached`, `fp::views::window`, `fp::linalg`
(`norm_l2`, `dot`), `fp::Rng`.

Gate: k-NN matches brute force against a k-d tree; LOF flags planted outliers.

### E4 — Clustering (after E2)

Spec: [clustering.md](clustering.md).

Files:

- [ ] `src/cluster/kmeans.hpp` — k-means++ init, Lloyd, mini-batch, k by
      elbow/silhouette
- [ ] `src/cluster/gmm.hpp` — EM, covariance types, BIC/AIC
- [ ] `src/cluster/dbscan.hpp` — density clustering, noise points
- [ ] `src/cluster/agglomerative.hpp` — single/complete/average/ward linkage
- [ ] `src/cluster/metrics.hpp` — silhouette, Davies–Bouldin, Calinski–Harabasz
- [ ] `test/cluster_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::linalg` (`row_sums`, `matvec`, `norm_l2`),
`fp::inplace`, `fp::views::chunk`.

Gate: k-means recovers three planted Gaussian blobs; DBSCAN separates them
from noise; cluster metrics rank the true k first.

### E5 — Dimensionality reduction (after E2)

Spec: [reduction.md](reduction.md).

Files:

- [ ] `src/reduce/pca.hpp` — covariance eigendecomposition, explained variance,
      whitening, projection/inverse
- [ ] `src/reduce/kernel_pca.hpp` — RBF/linear kernels
- [ ] `src/reduce/lda.hpp` — Fisher discriminant, multi-class
- [ ] `src/reduce/projections.hpp` — Gaussian and sparse random projections
- [ ] `src/reduce/nmf.hpp` — multiplicative updates
- [ ] `test/reduce_test.cpp`

ForgeFP usage: `fp::linalg` (`matmul`, `transpose`, `solve`, `dot`),
`fp::grid`, `fp::Rng`, `fp::approx_equal`.

Gate: PCA on correlated data keeps ≥ 95% variance in two components; LDA
separates two planted classes; random projections preserve pairwise distances
within the JL bound.

### E6 — Ensembles (after Stage 7 and E3)

Spec: [ensembles.md](ensembles.md).

Files:

- [ ] `src/ensemble/bagging.hpp` — bootstrap sampling, OOB error
- [ ] `src/ensemble/forest.hpp` — random forest (feature subsampling,
      importance, regression/classification)
- [ ] `src/ensemble/boosting.hpp` — AdaBoost, gradient boosting (shrinkage,
      subsampling, regression/classification)
- [ ] `src/ensemble/stacking.hpp` — voting/averaging, out-of-fold stacking
- [ ] `test/ensemble_test.cpp`

ForgeFP usage: `fp::Rng` (`sample_indices`), `fp::views::chunk`,
`fp::par_map` (independent learners), `fp::sort_by`.

Gate: a forest beats a single CART tree on a planted noisy dataset; boosting
drives training error toward zero; stacking beats its best base learner.

### E7 — Interpretation and calibration (after Stage 9)

Spec: [interpret.md](interpret.md).

Files:

- [ ] `src/interpret/importance.hpp` — permutation importance, tree importance
- [ ] `src/interpret/pdp.hpp` — partial dependence and ICE curves
- [ ] `src/interpret/calibration.hpp` — reliability curve, Platt scaling,
      isotonic regression
- [ ] `src/interpret/curves.hpp` — ROC/PR curves with thresholds, AUC/AP,
      threshold selection
- [ ] `src/interpret/learning.hpp` — learning/validation curves
- [ ] `test/interpret_test.cpp`

ForgeFP usage: `fp::Rng` (permutations), `fp::sort_by`, `fp::views::chunk`,
`fp::grid`, `fp::linalg` (`mean`).

Gate: permutation importance ranks a planted feature first; Platt scaling
turns a miscalibrated score into a calibrated one (ECE halves); the ROC curve
matches a hand-computed example.

### E8 — Time series (after E3)

Spec: [timeseries.md](timeseries.md).

Files:

- [ ] `src/timeseries/window.hpp` — lag/rolling features, expanding statistics
- [ ] `src/timeseries/stationary.hpp` — differencing, ACF/PACF
- [ ] `src/timeseries/arima.hpp` — AR/MA/ARIMA via conditional least squares
- [ ] `src/timeseries/smoothing.hpp` — SES, Holt, Holt–Winters
- [ ] `src/timeseries/backtest.hpp` — rolling-origin splits, forecast metrics
      (MAE/RMSE/MAPE/sMAPE/MASE)
- [ ] `test/timeseries_test.cpp`

ForgeFP usage: `fp::views::windows`, `fp::linalg` (`solve`, `dot`),
`fp::inplace`, `fp::Rng`.

Gate: ARIMA beats a naive last-value forecast on a planted AR(2) series;
Holt–Winters tracks a seasonal series; backtesting never leaks the future.

### E9 — Recommender systems (after E2)

Spec: [recommend.md](recommend.md).

Files:

- [ ] `src/recsys/mf.hpp` — matrix factorization by SGD and ALS (explicit and
      implicit feedback)
- [ ] `src/recsys/ranking.hpp` — precision@k, recall@k, MAP, NDCG
- [ ] `src/recsys/baseline.hpp` — most-popular and bias baselines
- [ ] `test/recsys_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::linalg` (`matmul`, `solve`, `dot`),
`fp::inplace`, `fp::sort_by_cached`.

Gate: MF beats the popularity baseline on held-out interactions; NDCG matches
a hand-computed ranking.

### E10 — LLM inference (after Stage 14)

Spec: [inference.md](inference.md).

Files:

- [ ] `src/inference/kv_cache.hpp` — pre-allocated KV cache, sliding window,
      eviction policy
- [ ] `src/inference/sampling.hpp` — temperature, top-k, top-p, typical,
      repetition penalty, beam search (extends `llm/sampling`)
- [ ] `src/inference/quantize.hpp` — int8/int4 group-wise quantization and
      dequantized matmul/embedding lookups
- [ ] `src/inference/batch.hpp` — batched/continuous decoding over a queue
- [ ] `src/inference/finetune.hpp` — LoRA adapters (train/merge)
- [ ] `test/inference_test.cpp`

ForgeFP usage: `fp::Buffer`, `fp::linalg`, `fp::numerics` (`softmax`,
`logsumexp`), `fp::inplace`, `fp::Rng`.

Gate: cached decoding is bit-identical to full recomputation and faster;
int8 quantization stays within tolerance of fp32; beam search beats greedy on
a planted sequence.

### E11 — Reinforcement learning (optional tier, after E3)

Spec: [rl.md](rl.md). Explicitly optional: nothing above depends on it.

Files:

- [ ] `src/rl/bandit.hpp` — ε-greedy, UCB1, Thompson sampling
- [ ] `src/rl/tabular.hpp` — Q-learning, SARSA, TD(0), eligibility traces
- [ ] `src/rl/env.hpp` — a tiny deterministic environment interface
      (gridworld, chain) for tests
- [ ] `test/rl_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::map_values`, `fp::views::chunk`, `fp::inplace`.

Gate: bandits converge to the best arm; Q-learning solves the test gridworld
with a fixed seed.

### E12 — Performance program (cross-cutting, from Stage 11 on)

Spec: [performance.md](performance.md). Not a layer: the harness, the
precision switch and the budgets every later stage must meet.

Files:

- [ ] `src/core/scalar.hpp` — `forgeml::Scalar` (float32 builds via
      `FORGEML_SINGLE_PRECISION`); kernels templated on it
- [ ] `perf/timer.hpp` — named phase timers (`forward`/`backward`/`optimizer`/`data wait`)
- [ ] `perf/alloc_counter.hpp` — allocation and byte counters
- [ ] `perf/memory.hpp` — parameters/activations/caches/batch resident bytes
- [ ] `bench/*.cpp` — one benchmark per hot module, shapes in the labels
- [ ] `bench/compare/*.py` + `report.py` — PyTorch reference scripts and the ratio table
- [ ] `test/perf_test.cpp` — a training step allocates nothing after warm-up

ForgeFP usage: `fp::Stopwatch`, `fp::par_for`, `fp::simd`, `fp::Buffer`,
`fp::Arena`.

Gate: the ratio table exists for the reference workloads (≤ 2× classical,
≤ 3× deep CPU, ≤ 3× covered GPU kernels); a training step performs zero
allocations after warm-up; `forge bench --compare --fail-over 5` is wired into
every hot stage.

### E13 — Deep RL (after E11)

Spec: [rl.md](rl.md). The tier stays isolated: nothing imports it, and its
benchmarks share no budget with the core.

Files:

- [ ] `src/rl/policy_gradient.hpp` — REINFORCE with returns-to-go and a baseline
- [ ] `src/rl/actor_critic.hpp` — A2C with an entropy bonus
- [ ] `src/rl/dqn.hpp` — replay ring, target network, ε decay
- [ ] `src/rl/ppo.hpp` — GAE(λ), clipped surrogate, KL early stop
- [ ] `src/rl/env.hpp` — `ContinuousEnv`, `CartPole`, `ContinuousBandit`
- [ ] `test/rl_deep_test.cpp` — seeded episode budgets per algorithm
- [ ] `test/rl_isolation_test.cpp` — nothing outside `rl/` includes `rl/`

ForgeFP usage: `fp::Rng`, `fp::numerics` (`log_softmax`), `fp::Buffer` (replay),
`fp::linalg::dot`, `fp::inplace`.

Gate: each algorithm reaches its environment's `solved_return()` within the
documented episode budget (mean over seeded runs with a confidence interval);
the isolation test passes.

### E14 — GPU depth (after Stage 15)

Spec: [gpu.md](gpu.md). Completes the coverage matrix and the transfer
overlap; the CPU path remains the reference.

Files:

- [ ] `src/gpu/conv.hpp` — im2col + device GEMM (a direct kernel is future work)
- [ ] `src/gpu/norm.hpp` — device LayerNorm/RMSNorm, dropout masks
- [ ] `src/gpu/kv_cache.hpp` — device-resident KV buffers for `inference/kv_cache.hpp`
- [ ] `src/gpu/overlap.hpp` — double-buffered pinned staging, batch `n+1` copying while step `n` computes
- [ ] `bench/compare/kernels_cuda.py` — the CUDA reference numbers
- [ ] `test/gpu_coverage_test.cpp` — every matrix row: device vs CPU within tolerance

ForgeFP usage: `fp::gpu::matmul`, `transform_inplace_indexed`,
`add_row_broadcast`, `row_means`, `HostBuffer`, `Scratch`, `fp::approx_equal`.

Gate: the coverage matrix is green on the RTX 3060 (within 3× CUDA for covered
kernels); the overlap hides the host copy (step time ≈ compute time); the
suite still passes with `FORGEML_DISABLE_GPU=1` and without a device.

### E15 — Graph neural networks (after Stage 11)

Spec: [gnn.md](gnn.md).

Files:

- [ ] `src/gnn/graph.hpp` — `Graph`, adjacency, self-loops, normalization, batching, permutation
- [ ] `src/gnn/gcn.hpp` — graph convolution layers + a masked-node training step
- [ ] `src/gnn/gat.hpp` — neighbour attention (multi-head)
- [ ] `src/gnn/pool.hpp` — mean/max/top-k graph readout
- [ ] `test/gnn_test.cpp`

ForgeFP usage: `fp::linalg::matmul`, `fp::numerics` (`softmax`), `fp::sort_by`,
`fp::views::chunk`, `fp::Rng`.

Gate: permutation equivariance for both layers; a planted community graph is
classified; GAT learns to weight an informative edge; batching and pooling
never leak across graphs; gradient checks pass.

### E16 — Bayesian optimization (after E12)

Spec: [bayesopt.md](bayesopt.md).

Files:

- [ ] `src/bayesopt/gp.hpp` — GP regression (RBF/Matérn), Cholesky, marginal likelihood
- [ ] `src/bayesopt/acquisition.hpp` — EI, PI, UCB
- [ ] `src/bayesopt/search.hpp` — the BO loop, random and grid baselines
- [ ] `test/bayesopt_test.cpp`

ForgeFP usage: `fp::linalg` (`matmul`, `solve`), `fp::Rng`,
`fp::sort_by_cached`, `fp::views::chunk`.

Gate: the GP matches a hand-computed posterior; acquisition functions match
hand-computed values; BO beats random search at an equal budget on planted
functions; a CV hyperparameter search improves on the default.

### E17 — Active learning (after Stage 9)

Spec: [active.md](active.md).

Files:

- [ ] `src/active/query.hpp` — least-confident, margin, entropy, committee, k-center
- [ ] `src/active/loop.hpp` — the label-budget loop with stopping rules
- [ ] `test/active_test.cpp`

ForgeFP usage: `fp::numerics` (`entropy`), `fp::sort_by_cached`, `fp::Rng`,
`fp::views::chunk`.

Gate: scores match hand-computed values; the loop respects the budget and
never repeats a label; on planted data active learning is measurably more
label-efficient than random sampling.

### E18 — Semi- and self-supervised learning (after Stage 11 and E3)

Spec: [semi.md](semi.md).

Files:

- [ ] `src/semi/pseudo.hpp` — self-training with a confidence threshold and class balance
- [ ] `src/semi/propagate.hpp` — label propagation on a k-NN graph
- [ ] `src/semi/contrastive.hpp` — InfoNCE with augmentations
- [ ] `test/semi_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::numerics` (`log_softmax`), `fp::linalg`,
`fp::sort_by_cached`.

Gate: each method beats its supervised-only baseline with few labels;
propagation recovers planted clusters from a handful of seeds; InfoNCE
embeddings are usable by a k-NN classifier where raw features are not.

### E19 — Anomaly detection (after E3)

Spec: [anomaly.md](anomaly.md).

Files:

- [ ] `src/anomaly/isolation.hpp` — isolation forest
- [ ] `src/anomaly/one_class.hpp` — one-class SVM (reuses the SMO solver)
- [ ] `src/anomaly/mahalanobis.hpp` — distance from the normal cloud
- [ ] `src/anomaly/evaluate.hpp` — ROC-AUC, PR-AUC, precision@k
- [ ] `test/anomaly_test.cpp`

ForgeFP usage: `fp::Rng`, `fp::linalg` (`solve`, `dot`), `fp::sort_by_cached`,
`fp::views::chunk`.

Gate: planted anomalies are ranked first (AUCs well above chance) by all three
detectors; thresholds derive from contamination or a chi-square quantile.

### E20 — Online learning and drift (after Stage 4)

Spec: [online.md](online.md).

Files:

- [ ] `src/online/welford.hpp` — running mean/variance, mergeable
- [ ] `src/online/incremental.hpp` — `partial_fit` protocol, online scaler
- [ ] `src/online/drift.hpp` — PSI, Page–Hinkley, ADWIN-lite
- [ ] `src/online/evaluate.hpp` — prequential evaluation
- [ ] `test/online_test.cpp`

ForgeFP usage: `fp::map2d`, `fp::inplace`, `fp::views::window`,
`fp::linalg` (`mean`, `variance`).

Gate: Welford matches two-pass statistics and merges exactly; `partial_fit`
converges to the batch solution; drift is detected within the documented
budget and stationary streams do not fire; prequential evaluation never leaks
a label.

### E21 — LLM evaluation harness (after Stage 14)

Spec: [llm-eval.md](llm-eval.md).

Files:

- [ ] `src/llm/eval.hpp` — perplexity (KV-cached), the JSON/markdown report
- [ ] `src/llm/fewshot.hpp` — prompt assembly, k-shot selection, deterministic generation
- [ ] `src/llm/text_metrics.hpp` — exact match, token F1, BLEU-lite, ROUGE-lite, MCQ scoring
- [ ] `src/llm/contamination.hpp` — 13-gram overlap check
- [ ] `test/llm_eval_test.cpp`

ForgeFP usage: `fp::numerics` (`logsumexp`), `fp::str`, `fp::Rng`,
`fp::sort_by_cached`, `fp::views::chunk`.

Gate: every metric matches a hand-computed example; the harness is
deterministic for a seed; MCQ scoring is length-normalized; the contamination
check flags an overlapping document and passes a clean one.

## Remaining gaps (audited)

Everything above is *specified*. This table is the honesty check: what is not
in the stack, and why — so the boundary is a decision, not an oversight.

| Area | Status | Disposition |
|---|---|---|
| Reverse-mode autodiff (general tape) | `nn`/`llm` have bespoke backprop; fp has forward-mode only | **Keep bespoke** (the tape is the models' engine); revisit only if a third consumer appears |
| Mixed precision (fp16/bf16) | excluded until fp has a half type | future work in **ForgeFP**; int8/int4 covered by [inference.md](inference.md) |
| Distributed training | non-goal | sketch only: data-parallel with gradient averaging; no dependency-free all-reduce exists, so no code |
| Vendor-kernel GEMM/conv | non-goal | escalate to fp as an opt-in BLAS backend if a budget needs it ([performance.md](performance.md)) |
| Graph neural networks | **specified** | [gnn.md](gnn.md) — E15 |
| Bayesian optimization / AutoML | **specified** | [bayesopt.md](bayesopt.md) — E16 |
| Active learning | **specified** | [active.md](active.md) — E17 |
| Semi-/self-supervised learning | **specified** | [semi.md](semi.md) — E18 |
| Anomaly detection | **specified** (LOF + isolation forest + one-class SVM + Mahalanobis) | [anomaly.md](anomaly.md) — E19 |
| Online learning / drift detection | **specified** | [online.md](online.md) — E20 |
| LLM evaluation harness | **specified** | [llm-eval.md](llm-eval.md) — E21 |
| Reproducibility contract | rules scattered | folded into [conventions.md](conventions.md#determinism) (bit-exact vs tolerance) |
| Causal inference, survival analysis | missing | **non-goal**, declared |
| Speech/audio, image codecs | missing | **non-goal** (external codecs); dsio moves the bytes, nothing decodes them |
| Serving (HTTP/gRPC), ONNX export | missing | **non-goal** |
| Fairness / robustness / adversarial suites | missing | **non-goal**, declared |
| Experiment tracking, data versioning | missing | **non-goal** (JSON logs + checkpoints cover the local case) |

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
| M9 | E4, E5 | k-means recovers planted blobs; PCA keeps ≥ 95% variance in two components |
| M10 | E6 | A forest beats a single tree on planted noisy data; stacking beats its best base learner |
| M11 | E8 | ARIMA beats the naive forecast on a planted AR(2) series; backtesting never leaks the future |
| M12 | E10 | Cached decoding is bit-identical and faster; int8 stays within tolerance of fp32 |
| M13 | E11 *(optional)* | Bandits converge to the best arm; Q-learning solves the test gridworld |
| M14 | E12 | The PyTorch ratio table meets the budgets (≤ 2× classical, ≤ 3× deep CPU, ≤ 3× covered GPU kernels) |
| M15 | E13 | REINFORCE, A2C, DQN and PPO solve their tasks; the isolation test passes |
| M16 | E14 | The GPU coverage matrix is green; the suite passes with `FORGEML_DISABLE_GPU=1` |
| M17 | E15 | A GCN classifies a planted community graph; permutation equivariance holds |
| M18 | E16 | BO beats random search at an equal budget; the GP matches a hand-computed posterior |
| M19 | E17 | Active learning reaches the target with fewer labels than random sampling |
| M20 | E18 | Self-training and contrastive pre-training beat their supervised-only baselines |
| M21 | E19 | Planted anomalies are ranked first by all three detectors (PR-AUC well above chance) |
| M22 | E20 | Drift is detected within the documented budget; stationary streams do not fire |
| M23 | E21 | Every LLM metric matches a hand-computed example; the harness is seed-reproducible |

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
| E1–E2 | 3 days | 2, 3 |
| E3–E5 | 4–5 days | E2 |
| E6–E7 | 3–4 days | 7, E3 |
| E8–E9 | 3 days | E2, E3 |
| E10 | 3–4 days | 14 |
| E11 *(optional)* | 1–2 days | E3 |
| E12 (performance) | 4–5 days, cross-cutting | 11 |
| E13 (deep RL) | 3–4 days | E11, 11, 4 |
| E14 (GPU depth) | 3–5 days | 15 |
| E15 (GNN) | 3–4 days | 11, 4 |
| E16 (BayesOpt) | 2–3 days | E12, E2 |
| E17 (active) | 2 days | 9, E6 |
| E18 (semi/self-supervised) | 3 days | 11, E3, E1 |
| E19 (anomaly) | 2 days | E3, 8 |
| E20 (online/drift) | 2 days | 4, E2 |
| E21 (LLM eval) | 2 days | 14 |

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

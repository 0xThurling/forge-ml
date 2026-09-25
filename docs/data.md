# data — preparing data for models

Turns raw tables into model-ready matrices: hold, validate, split, scale,
encode, batch. No model logic.

Depends on: `core`, `math/stats`, ForgeFP.

Files: `dataset.hpp`, `dataloader.hpp`, `split.hpp`, `scaling.hpp`,
`encoding.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `dataset.hpp` | `fp::Validation`, `fp::to_result`, `fp::fail`, `fp::read_lines`, `fp::str::split_any`, `fp::str::parse_numbers`, `fp::is_finite` |
| `split.hpp` | `fp::Rng`, `fp::Rng::shuffle`, `fp::Rng::sample_indices`, `fp::Result`, `fp::range` |
| `scaling.hpp` | `fp::col_means`, `fp::map2d`, `fp::map2d_indexed`, `fp::transform_inplace`, `fp::map_to` |
| `encoding.hpp` | `fp::str::to_lower`, `fp::sort`, `fp::unique`, `fp::argmax`, `fp::map` |
| `dataloader.hpp` | `fp::views::chunk`, `fp::Rng::shuffle`, `fp::map_to`, `fp::Buffer` (reused batch storage) |

Randomness is `fp::Rng` — there is no `core/random.hpp`.

---

## `data/dataset.hpp`

**Job:** hold features and targets together, guarantee `X.rows() == y.size()`,
and provide index-based views.

```cpp
namespace forgeml {

template <class Y = double>
struct Dataset {
  Matrix<double> X;     // (n_samples, n_features)
  Vector<Y> y;          // (n_samples)

  std::size_t num_samples() const { return X.rows(); }
  std::size_t num_features() const { return X.cols(); }

  // One row of features + its label, without copying the label vector.
  struct Sample { std::span<const double> features; Y label; };
  Sample sample(std::size_t i) const;

  Dataset<Y> subset(std::vector<std::size_t> const &indices) const;
  std::vector<std::size_t> indices() const;      // 0..n-1
};

// Build + validate. Collects every problem, not just the first.
fp::Validation<Dataset<Y>> validate(Dataset<Y> const &data);
fp::Result<Dataset<Y>> make_dataset(Matrix<double> X, Vector<Y> y);

// CSV loader: header optional, comma delimiter, numeric features,
// last column = target (or target_column index).
fp::Result<Dataset<double>> load_csv(std::string const &path,
                                     int target_column = -1,
                                     bool has_header = true);
}
```

Rules:

- `make_dataset` fails when `X.rows() != y.size()`, `X` is empty, or any value
  is non-finite. `validate` reports *all* problems via `fp::Validation`.
- `subset` preserves row order of `indices`; used by splits and CV.
- `load_csv` uses `fp::read_lines` and `fp::str::parse_numbers<double>` per
  row; a bad row returns `fp::fail("row 7: not a number: 'x'")`. Missing cells
  are an error (no silent zero-fill).
- `Dataset<Y>` is a plain aggregate; no inheritance, no virtuals.

Tests (`test/data_test.cpp`):

- `make_dataset` on mismatched rows returns `fail`
- `validate` on a dataset with two problems reports both messages
- `subset({2,0})` returns rows in that order
- CSV round-trip: write a temp file, load it, check values and shape
- CSV with a malformed row returns the row number in the message

---

## `data/split.hpp`

**Job:** reproducible train/validation/test and k-fold index generation. Pure
index math plus a shuffle; never touches features.

```cpp
namespace forgeml {

struct Split {
  std::vector<std::size_t> train, test;
};

struct ThreeWaySplit {
  std::vector<std::size_t> train, val, test;
};

// test_ratio in (0,1), shuffles indices with rng
fp::Result<Split> train_test_split(std::size_t n, double test_ratio, fp::Rng &rng);

fp::Result<ThreeWaySplit> train_val_test_split(std::size_t n, double val_ratio,
                                               double test_ratio, fp::Rng &rng);

// k folds; fold f is the test set, the rest is train
std::vector<Split> k_fold_indices(std::size_t n, std::size_t k, fp::Rng &rng);

// Stratified k-fold for classification: preserve class proportions per fold.
// labels must be integers 0..num_classes-1.
fp::Result<std::vector<Split>> stratified_k_fold_indices(
    std::vector<int> const &labels, std::size_t num_classes, std::size_t k,
    fp::Rng &rng);

// Convenience: split an actual dataset (uses subset)
template <class Y> Split train_test_split(Dataset<Y> const &data,
                                          double test_ratio, fp::Rng &rng);
}
```

Rules:

- Validate ratios: `0 < r < 1`, `val_ratio + test_ratio < 1`; otherwise
  `fp::fail`.
- Every index appears exactly once across all folds of `k_fold_indices`.
- Shuffle is the only source of randomness and uses the passed `fp::Rng`.
- Stratified: shuffle within each class, then distribute round-robin so fold
  sizes differ by at most one; class ratios are preserved within rounding.

Formulas / algorithm:

```text
indices = [0..n-1]; shuffle(indices)
test_n  = round(n * test_ratio)
test    = indices[0 : test_n]
train   = indices[test_n : n]
```

Tests (`test/data_test.cpp`):

- split sizes sum to `n`; no index in both sets
- `k_fold_indices(n, 5)` covers `[0,n)` exactly once and fold sizes differ by
  at most 1
- stratified folds keep class ratios within 5% for an imbalanced toy set
- invalid ratio returns `fail`

---

## `data/scaling.hpp`

**Job:** feature scaling that learns statistics on train and applies them
elsewhere. Prevents leakage by construction.

```cpp
namespace forgeml {

class StandardScaler {
public:
  // Fit on training data: store per-feature mean and stddev (ddof = 0).
  fp::Result<void> fit(Matrix<double> const &X);

  // (X - mean) / scale, with scale = stddev (zero variance -> 1)
  Matrix<double> transform(Matrix<double> const &X) const;
  Vector<double> transform(Vector<double> const &x) const;

  Matrix<double> fit_transform(Matrix<double> const &X);

  Matrix<double> inverse_transform(Matrix<double> const &X_scaled) const;

  Vector<double> const &mean() const;
  Vector<double> const &scale() const;

private:
  Vector<double> mean_, scale_;
};

class MinMaxScaler {
public:
  fp::Result<void> fit(Matrix<double> const &X);          // store min/max
  Matrix<double> transform(Matrix<double> const &X) const; // [0,1]
  Matrix<double> inverse_transform(Matrix<double> const &X_scaled) const;
private:
  Vector<double> min_, max_;
};
}
```

Rules:

- `transform` before `fit` is a bug: `ML_ASSERT(fitted_)`.
- Zero-variance feature: `scale = 1` (do not divide by zero); document it.
- `fit` returns `fail` on empty `X` or non-finite values.
- `inverse_transform(transform(X)) ≈ X` within `1e-9`.
- Scalers store only vectors, so they serialize trivially.

ForgeFP implementation: `fit` is `fp::col_means` plus
`math/stats.hpp::col_stddevs` (per-column variance is the one statistics gap in
fp today); `transform` is `fp::map2d_indexed(X, ...)` or
`fp::map_to(X, out, ...)` when the caller reuses a buffer;
`inverse_transform` mirrors it with `*`/`+`. Zero-variance columns use
`fp::clamp(scale, eps, inf)`-style guarding (a scale of `1`).

Formulas:

```text
StandardScaler: x' = (x - mean_j) / std_j
MinMaxScaler:   x' = (x - min_j) / (max_j - min_j)
```

Tests (`test/scaling_test.cpp`):

- mean/std of a known matrix; transformed columns have mean ~0, std ~1
- zero-variance column transforms to 0 without NaN
- inverse round-trip within `1e-9`
- fitting on train and transforming test uses train statistics (no leakage)

---

## `data/encoding.hpp`

**Job:** map labels between their raw form and the integer/one-hot form models
expect.

```cpp
namespace forgeml {

class LabelEncoder {
public:
  // Fit: sorted unique labels -> 0..k-1
  void fit(std::vector<std::string> const &labels);
  fp::Result<std::vector<int>> transform(std::vector<std::string> const &labels) const;
  std::vector<std::string> inverse_transform(std::vector<int> const &codes) const;
  std::size_t num_classes() const;
  std::vector<std::string> const &classes() const;
private:
  std::vector<std::string> classes_;
};

// One-hot: (n, k) matrix of 0/1.
Matrix<double> one_hot(std::vector<int> const &codes, std::size_t num_classes);

// Argmax decode uses fp::argmax_rows directly — do not reimplement it.

// Binary mapping for logistic regression / SVM: {a,b} -> {0,1} or {-1,+1}.
struct BinaryLabelMap {
  std::string negative, positive;
  std::vector<int> codes;      // 0/1
  std::vector<int> signs;      // -1/+1
};
fp::Result<BinaryLabelMap> binary_label_map(std::vector<std::string> const &labels);
}
```

Rules:

- `transform` on an unseen label returns `fail("unknown label: ...")`, never a
  silent `-1`.
- `one_hot` validates codes are in range.
- Decoding probabilities to classes is `fp::argmax_rows` (first max on
  ties); `encoding.hpp` does not duplicate it.
- Sorting/normalizing label text uses `fp::str::to_lower`/`trim`;
  `classes_` is built with `fp::sort` + `fp::unique`.

Tests (`test/encoding_test.cpp`):

- round-trip labels through codes
- unknown label returns `fail`
- one-hot rows sum to 1 and have the right column set
- `binary_label_map` on `{"no","yes"}` produces sorted classes and correct signs

---

## `data/dataloader.hpp`

**Job:** iterate a dataset in mini-batches, optionally shuffling each epoch.

```cpp
namespace forgeml {

template <class Y>
struct Batch {
  Matrix<double> X;   // (batch_size, n_features)
  Vector<Y> y;        // (batch_size)
};

template <class Y>
class DataLoader {
public:
  DataLoader(Dataset<Y> const *data, std::size_t batch_size,
             bool shuffle = false, fp::Rng *rng = nullptr);

  std::size_t num_batches() const;          // ceil(n / batch_size)
  std::size_t batch_size() const;

  // One pass over the data. Call repeatedly for epochs; shuffles if enabled.
  std::vector<Batch<Y>> epoch() const;

  // Streaming form for large data: calls f(Batch) per batch.
  void for_each_batch(std::function<void(Batch<Y> const &)> const &f) const;
};
}
```

Rules:

- `batch_size >= 1` (assert); last batch may be smaller (no dropping unless a
  `drop_last` flag is added later).
- Shuffle uses the passed `fp::Rng`; when `shuffle == true`, `rng` must not be
  null (assert).
- A full epoch visits every sample exactly once.
- Batches copy data (`Matrix`/`Vector`); this is the right trade for clarity.
  A zero-copy span-based batch is a documented future optimization.
- ForgeFP implementation: `epoch()` is `fp::Rng::shuffle` on the index vector,
  `fp::views::chunk(indices, batch_size)` for the slicing, and
  `fp::map_to` into reused batch storage. `for_each_batch` can reuse a
  single `Batch` object backed by `fp::Buffer` scratch instead of allocating
  per batch — the streaming API exists so callers can do exactly that.

Tests (`test/dataloader_test.cpp`):

- `num_batches` with and without a remainder
- two epochs with shuffle produce different orders (fixed seed)
- every sample appears exactly once per epoch
- `for_each_batch` sees the same batches as `epoch()`

---

## Disk-backed datasets (dsio)

`Dataset<Y>` and `DataLoader<Y>` above are **in-memory**: they own `Matrix` /
`Vector` storage. Datasets larger than RAM — LLM token streams, large shards —
come from **dsio**, the sibling storage library: a shard set plus a streaming
reader built on O_DIRECT and io_uring. The wiring plan is
[`dsio/docs/ml-integration.md`](../../dsio/docs/ml-integration.md); the API
reference is [`dsio/docs/usage.md`](../../dsio/docs/usage.md).

**New to this?** [loading.md](loading.md) is a guided lesson: why O_DIRECT and
alignment exist, how queue depth turns latency into bandwidth, why shard size
is the biggest knob on this machine, and how the GPU paths differ — with
experiments to run and checkpoints.

### CPU

- A dataset on disk is a `dsio::ShardSet` — a TSV manifest of
  `path<TAB>offset<TAB>length`, one shard per file (or a range inside one).
- `dsio::for_each_batch(shards, options, callback)` streams the shards as one
  continuous byte stream, `prefetch` batches ahead, with a seeded shuffle of
  the shard order. ml wraps it in `ShardStream<T>` (fixed-size records, see the
  plan) and copies each batch into ml storage.
- Layout rules that matter on the reference machine (WSL2): shards of
  **512 MiB–1 GiB** (the vhdx ramps with contiguous run length: 32 MiB shards
  ≈ 2–3 GiB/s, 1 GiB shards ≈ 6.2 GiB/s), **one reader stream** (concurrency
  dilutes the ramp), batches ≥ 1 MiB. Record-level shuffling stays in ml; dsio
  only shuffles shard order.

### GPU

Two paths, one API:

1. **Host staging → device copy** — works everywhere, including the CPU
   fallback: dsio fills host batches; ml copies them into device buffers with
   the `fp::gpu` tier (`fp::gpu::Buffer`, pinned `HostBuffer`; see
   [gpu.md](gpu.md)). This is the path that runs on WSL2 and on machines
   without GPUDirect.
2. **GPUDirect Storage (cuFile)** — a `DSIO_WITH_CUFILE` build plus an NVIDIA
   GPU with `nvidia-fs` lets `dsio::open_device_sink()` hand out device memory
   and `dsio::read_into()` DMA straight from the NVMe into it, with no host
   bounce buffer. `dsio::gpu_direct_available()` reports whether the
   build/driver allow it; without it `open_device_sink()` explains what is
   missing and the host path is used. The extension point is `dsio::Sink`
   (`region` / `kind` / `alignment` / `commit`).

The split of concerns: **`fp::gpu` computes on the device; dsio's sink is how
the bytes get there.** A fully device-resident pipeline needs both.

## Cross-cutting notes

- Index vectors are `std::vector<std::size_t>`; splits and loaders pass them
  around, so `Dataset::subset` is the only place that copies features.
- Data flows one way: `Dataset -> (split) -> (scale) -> (encode) -> DataLoader`.
  Never fit a scaler or encoder on the test set.
- `Dataset` is an aggregate, so structured bindings work:
  `auto [X, y] = data;` copies — prefer `data.X`, `data.y` in hot code.

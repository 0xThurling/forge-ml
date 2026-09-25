# features — preparing raw data for models

The `data/` layer loads, splits, scales and batches. This layer does the rest of
the preparation: missing values, non-linear and interaction terms, text,
selection, imbalance, and augmentation. Everything is deterministic when given
an `fp::Rng&`, and every fallible step returns `fp::Result` or
`fp::Validation` (all bad rows at once).

Depends on: `core`, `math/stats`, `prob` (mutual information), `model`
(L1/RFE), ForgeFP.

Files: `impute.hpp`, `binning.hpp`, `polynomial.hpp`, `text.hpp`, `select.hpp`,
`imbalance.hpp`, `augment.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `impute.hpp` | `fp::col_means`, `fp::map2d_indexed`, `fp::Validation`, `fp::is_finite` |
| `binning.hpp` | `fp::sort_by`, `fp::views::chunk`, `fp::map` |
| `polynomial.hpp` | `fp::views::chunk`, `fp::cartesian_product`, `fp::flat_map` |
| `text.hpp` | `fp::str::split_any`, `fp::str::to_lower`, `fp::map_values`, `fp::sort_by` |
| `select.hpp` | `fp::sort_by_cached`, `fp::views::chunk`, `fp::par_map` |
| `imbalance.hpp` | `fp::Rng::sample_indices`, `fp::linalg::norm_l2`, `fp::sort_by_cached` |
| `augment.hpp` | `fp::Rng`, `fp::grid`, `fp::inplace` |

---

## `features/impute.hpp`

**Job:** replace missing values (NaN or a caller-provided mask) so downstream
code sees a dense matrix.

```cpp
namespace forgeml {

enum class Impute { Mean, Median, Mode, Constant, Forward };

struct ImputeResult {
  Matrix<double> X;          // dense
  Vector<double> fill;       // the value used per column
  std::vector<std::size_t> imputed_rows;   // audit trail
};

fp::Result<ImputeResult> impute(Matrix<double> const &X, Impute how,
                                double constant = 0.0);
}
```

Rules:

- Fit statistics come from the **training split only**; `fill` is returned so
  the same values can be applied to test data (`apply_impute`).
- `Mode`/`Forward` are for categorical/ordered columns; `Mean`/`Median`
  require finite values in the observed cells.
- A column that is entirely missing is an error, not a zero.

Tests (`test/features_test.cpp`): each strategy on a hand-built matrix; test
data gets the training `fill`; all-missing column errors.

## `features/binning.hpp`

**Job:** discretize continuous columns — equal width or quantile — into
indices and (optionally) one-hot columns.

```cpp
struct Bins { Vector<double> edges; };
fp::Result<Bins> equal_width(Vector<double> const &x, std::size_t n);
fp::Result<Bins> quantile(Vector<double> const &x, std::size_t n);
fp::Result<Vector<std::size_t>> apply(Bins const &bins, Vector<double> const &x);
```

Tests: edges are monotone; quantile bins are near-equal-sized; applying to the
training data round-trips the bin counts.

## `features/polynomial.hpp`

**Job:** polynomial and interaction terms.

```cpp
fp::Result<Matrix<double>> polynomial(Matrix<double> const &X, int degree,
                                      bool interactions_only = false,
                                      bool bias = true);
```

Tests: degree-2 on two columns produces `[1, x, y, x², xy, y²]`; the column
count formula matches; interactions-only omits squares.

## `features/text.hpp`

**Job:** turn token lists into sparse-friendly numeric features without
external dependencies.

```cpp
struct Vocabulary { std::map<std::string, std::size_t> index; };
fp::Result<Vocabulary> build_vocab(std::vector<std::string> const &docs,
                                   std::size_t min_count = 1);
fp::Result<Matrix<double>> bag_of_words(Vocabulary const &v, std::vector<std::string> const &docs);
fp::Result<Matrix<double>> tf_idf(Vocabulary const &v, std::vector<std::string> const &docs);
Matrix<double> hashing_features(std::vector<std::string> const &docs, std::size_t buckets);
```

Rules: lower-casing and `split_any` tokenization are shared; n-grams
(`n = 1..3`) are a parameter; IDF uses the training corpus; hashing needs no
vocabulary (useful for streaming).

Tests: vocabulary respects `min_count`; TF-IDF downweights a document that
contains a frequent term; hashing keeps the same token in the same bucket.

## `features/select.hpp`

**Job:** drop uninformative columns, with the reason recorded.

```cpp
struct Selection {
  std::vector<std::size_t> keep;
  Vector<double> score;      // one per input column
};

fp::Result<Selection> variance_threshold(Matrix<double> const &X, double threshold);
fp::Result<Selection> mutual_information(Matrix<double> const &X, Vector<double> const &y, std::size_t keep);
fp::Result<Selection> l1_selection(Matrix<double> const &X, Vector<double> const &y, double lambda);
fp::Result<Selection> recursive_elimination(Model const &fit, Matrix<double> const &X, Vector<double> const &y, std::size_t keep);
```

Tests: a constant column is dropped; a planted informative column survives
MI and L1 selection; RFE with a linear model keeps the true features.

## `features/imbalance.hpp`

**Job:** class weights and resampling for skewed targets.

```cpp
Vector<double> class_weights(Vector<double> const &y);
fp::Result<Dataset<double>> random_oversample(Dataset<double> const &d, fp::Rng &rng);
fp::Result<Dataset<double>> random_undersample(Dataset<double> const &d, fp::Rng &rng);
fp::Result<Dataset<double>> smote(Dataset<double> const &d, fp::Rng &rng, std::size_t k = 5);
```

Rules: weights are `n / (classes * count)`; SMOTE interpolates between a
minority sample and one of its k-NN (from `neighbors/knn.hpp`) and **only on
the training split**.

Tests: weights sum to `n`; oversampling balances counts; SMOTE keeps feature
ranges and is deterministic for a seed.

## `features/augment.hpp`

**Job:** deterministic, label-preserving augmentation for the CNN/vision and
tabular paths.

```cpp
struct ImageAugment { bool flip_h = false, flip_v = false; int max_shift = 0; double max_rotate_deg = 0.0; double jitter = 0.0; };
fp::Result<Matrix<double>> augment_image(Matrix<double> const &image, ImageAugment const &a, fp::Rng &rng);
Matrix<double> add_noise(Matrix<double> const &X, double sigma, fp::Rng &rng);
```

Tests: flips are involutions; shifts preserve the value histogram; the same
seed reproduces the same augmentation; labels are never touched.

## Gate

An imbalanced, ragged, mixed-type table can be imputed, binned, expanded,
selected and augmented — deterministically — and feature selection keeps a
planted signal while dropping a constant column.

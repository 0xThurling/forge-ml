# anomaly — finding the rare and the wrong

Outliers without labels. Three complementary views: isolation (how easy is it
to separate this point), density (LOF, already in `neighbors/`), and
distance-from-normal (Mahalanobis, one-class SVM). Evaluation is ranking
quality, because thresholds move.

Depends on: `core`, `prob` (chi-square, Gaussian), `neighbors` (LOF),
`interpret` (PR/ROC curves), `model` (SMO solver for one-class SVM), ForgeFP.

Files: `isolation.hpp`, `one_class.hpp`, `mahalanobis.hpp`, `evaluate.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `isolation.hpp` | `fp::Rng`, `fp::sort_by_cached`, `fp::views::chunk` |
| `one_class.hpp` | `fp::linalg` (`dot`, `matmul`), `fp::inplace` |
| `mahalanobis.hpp` | `fp::linalg` (`mean`, `solve`, `dot`) |
| `evaluate.hpp` | `fp::sort_by_cached`, `fp::views::window` |

---

## `anomaly/isolation.hpp`

**Job:** isolation forest — anomalies are easier to isolate, so they have
shorter paths in random trees.

```cpp
namespace forgeml {

struct IsolationForest {
  std::vector<Cart> trees;      // depth-limited, random split features/values
  std::size_t sample_size = 256;
  double contamination = 0.1;
};

fp::Result<IsolationForest> isolation_forest(Matrix<double> const &X, std::size_t n_trees,
                                             std::size_t sample_size, fp::Rng &rng);
Vector<double> score(IsolationForest const &forest, Matrix<double> const &X);   // higher = odder
Vector<bool> predict(IsolationForest const &forest, Matrix<double> const &X);
}
```

Rules: each tree samples `sample_size` rows and splits on a random feature at
a random value within the observed range until depth `ceil(log2(sample_size))`;
the score uses the standard `2^(-E[path]/c(n))` normalization; `predict`
thresholds at the `contamination` quantile of the **training** scores (a
deterministic quantile, not a magic constant).

Tests: a point far from a dense cluster scores higher than a cluster point;
the training-score quantile matches `contamination`; a planted 5% outlier set
is mostly flagged; the score is stable for a seed.

## `anomaly/one_class.hpp`

**Job:** one-class SVM — fit a boundary around the normal data, reuse the SMO
solver from `model/svm`.

```cpp
struct OneClassSvm {
  Vector<double> alpha;         // dual weights (support vectors > 0)
  Matrix<double> support;       // support vectors
  double rho = 0.0;
  double gamma = 1.0;           // RBF width
};

fp::Result<OneClassSvm> one_class_svm(Matrix<double> const &X, double nu, double gamma);
Vector<double> decision_function(OneClassSvm const &model, Matrix<double> const &X);  // sign = normal
Vector<bool> predict(OneClassSvm const &model, Matrix<double> const &X);
```

Rules: the ν parameter bounds the fraction of outliers and the fraction of
support vectors; the dual is solved with the same SMO machinery as the
classifier (documented reuse, not a second solver); a `nu` outside `(0, 1]` is
an error.

Tests: a planted Gaussian plus far points is separated; the support-vector
fraction is within a factor of 2 of `nu`; decision values match a hand-computed
linear-kernel case; `nu = 1` marks everything as support.

## `anomaly/mahalanobis.hpp`

**Job:** distance from the normal cloud, in the cloud's own units.

```cpp
struct Mahalanobis { Vector<double> mean; Matrix<double> precision; };   // Σ⁻¹

fp::Result<Mahalanobis> fit_mahalanobis(Matrix<double> const &X, double ridge = 1e-6);
Vector<double> score(Mahalanobis const &model, Matrix<double> const &X);
double chi_square_threshold(std::size_t dimensions, double confidence);
```

Rules: the covariance gets a ridge before inversion (`fp::linalg::solve`, never
a raw inverse); the score is the squared Mahalanobis distance (so the
chi-square threshold applies directly); dimensions with zero variance are
dropped and reported, not divided by zero.

Tests: a hand-computed 2-D example matches; the threshold matches the
chi-square quantile; a point along the correlated axis needs a larger distance
to be flagged than along the short axis (the covariance matters).

## `anomaly/evaluate.hpp`

**Job:** ranking quality with the labels you have for the test set.

```cpp
struct AnomalyReport {
  double roc_auc = 0.0, pr_auc = 0.0;
  double precision_at_k = 0.0;
  std::size_t k = 0;
};
fp::Result<AnomalyReport> evaluate(Vector<double> const &scores, Vector<bool> const &is_outlier,
                                   std::size_t k = 0);
```

Rules: `pr_auc` comes from `interpret/curves.hpp` (average precision, the
right metric under class imbalance); `precision_at_k` defaults to the number
of true outliers; a single-class label vector is an error (the metric is
undefined).

Tests: a perfect ranking scores 1.0 on both AUCs; a random ranking scores
≈ 0.5 ROC-AUC; precision@k matches a hand example; single-class input errors.

## Gate

Plant anomalies are ranked first (ROC-AUC and PR-AUC well above chance) by all
three detectors; the metrics match hand-computed examples; thresholds derive
from contamination or a chi-square quantile — never a hard-coded score.

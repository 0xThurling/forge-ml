# neighbors — k-NN, density and outliers

Distance is the simplest model there is, and the one that keeps showing up
inside other algorithms (SMOTE interpolates between neighbors, LOF scores
local density, KDE estimates densities). One implementation of "who is near
whom", shared by all of them.

Depends on: `core`, `prob`, ForgeFP.

Files: `knn.hpp`, `kde.hpp`, `lof.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `knn.hpp` | `fp::sort_by_cached`, `fp::linalg` (`norm_l2`, `dot`), `fp::views::chunk` |
| `kde.hpp` | `fp::linalg` (`norm_l2`), `fp::map`, `fp::fold_left` |
| `lof.hpp` | `fp::sort_by_cached`, `fp::views::window`, `fp::linalg` |

---

## `neighbors/knn.hpp`

**Job:** k-nearest-neighbour classification and regression, brute force and
k-d tree, with the distance metric as a parameter.

```cpp
namespace forgeml {

enum class Metric { L2, L1, Cosine };
enum class Weights { Uniform, Distance };

struct Neighbors { Vector<std::size_t> index; Vector<double> distance; };

struct Knn {
  Matrix<double> X;      // training features
  Vector<double> y;      // labels or targets
  Metric metric = Metric::L2;
  Weights weights = Weights::Uniform;
  std::size_t k = 5;
  std::size_t tree_leaf = 16;   // k-d tree bucket size; 0 disables the tree
};

fp::Result<Knn> fit(Matrix<double> const &X, Vector<double> const &y,
                    Metric metric = Metric::L2, std::size_t k = 5);
fp::Result<Neighbors> query(Knn const &model, Vector<double> const &x);
fp::Result<double> predict(Knn const &model, Vector<double> const &x);
fp::Result<Vector<double>> predict_batch(Knn const &model, Matrix<double> const &X);
fp::Result<Vector<double>> predict_proba(Knn const &model, Matrix<double> const &X,
                                         std::size_t classes);
}
```

Rules:

- The k-d tree is built lazily and only for `L2`/`L1`; `Cosine` is brute force
  (with vectors pre-normalised, cosine becomes L2 — do that internally).
- Ties are broken by index, so results are deterministic.
- `k` is validated (`1 <= k <= n`); a distance-weighted vote of an exact match
  cannot divide by zero (weights clamp at a small epsilon).
- `k` selection is `eval`'s job: `cross_validate` over a `k` grid.

Tests: the k-d tree returns exactly the brute-force neighbors (random data,
several `k`); a hand-computed 3-NN example; weighted voting changes the label
for a planted close neighbor; `predict_proba` rows sum to 1.

## `neighbors/kde.hpp`

**Job:** non-parametric density estimation with Gaussian or Epanechnikov
kernels.

```cpp
struct Kde {
  Matrix<double> X;
  double bandwidth = 0.0;         // 0 = Silverman's rule
  enum class Kernel { Gaussian, Epanechnikov } kernel = Kernel::Gaussian;
};

fp::Result<Kde> fit_kde(Matrix<double> const &X, double bandwidth = 0.0,
                        Kde::Kernel kernel = Kde::Kernel::Gaussian);
double log_density(Kde const &kde, Vector<double> const &x);
Vector<double> log_density_batch(Kde const &kde, Matrix<double> const &X);
```

Rules: `log_density` uses `logsumexp` (never `log(sum(exp))`); Silverman's
rule is `1.06 * sigma * n^(-1/5)` per dimension with the product kernel;
bandwidth zero on a constant column falls back to the sample range.

Tests: density integrates to ~1 on a grid; a two-mode sample produces two
peaks; a smaller bandwidth produces a larger peak at a duplicated point.

## `neighbors/lof.hpp`

**Job:** local outlier factor — density relative to your neighbors.

```cpp
struct Lof {
  std::size_t k = 20;
  Vector<double> score;   // one per training row; 1.0 = typical
};

fp::Result<Lof> fit_lof(Matrix<double> const &X, std::size_t k = 20);
Vector<bool> outliers(Lof const &lof, double threshold = 1.5);
```

Rules: reachability distance is `max(k-distance(neighbor), d(x, neighbor))`;
scores near 1 are inliers; the threshold is explicit (no magic default inside
`fit_lof`).

Tests: a point far from a dense cluster scores well above 1.5; a point inside
the cluster scores near 1; the score is invariant to a global scale change.

## Gate

The k-d tree matches brute force; a hand-computed k-NN example is reproduced;
LOF flags planted outliers and leaves inliers alone; KDE integrates to one and
finds planted modes.

# clustering — finding structure without labels

Grouping, density and mixture models, with the metrics that say whether a
grouping is any good. All of it is unsupervised, so every result is a
*proposal*: the tests use planted structure so "correct" has a meaning.

Depends on: `core`, `math/stats`, `prob`, `neighbors` (k-NN for DBSCAN's
neighborhoods, distances for metrics), ForgeFP.

Files: `kmeans.hpp`, `gmm.hpp`, `dbscan.hpp`, `agglomerative.hpp`,
`metrics.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `kmeans.hpp` | `fp::Rng::sample_indices`, `fp::linalg` (`row_sums`, `norm_l2`, `mean`), `fp::inplace` |
| `gmm.hpp` | `fp::numerics` (`logsumexp`), `fp::linalg` (`matmul`, `solve`), `fp::Rng` |
| `dbscan.hpp` | `fp::views::chunk`, `fp::linalg::norm_l2`, `fp::sort_by_cached` |
| `agglomerative.hpp` | `fp::sort_by_cached`, `fp::views::window`, `fp::linalg` |
| `metrics.hpp` | `fp::linalg` (`row_means`, `norm_l2`), `fp::map2d` |

---

## `cluster/kmeans.hpp`

**Job:** k-means with a k-means++ start, a Lloyd loop and a mini-batch variant.

```cpp
namespace forgeml {

struct Kmeans {
  Matrix<double> centroids;   // (k, d)
  Vector<std::size_t> labels; // (n)
  double inertia = 0.0;       // sum of squared distances
  std::size_t iterations = 0;
};

fp::Result<Kmeans> kmeans(Matrix<double> const &X, std::size_t k,
                          fp::Rng &rng, std::size_t max_iterations = 300,
                          double tolerance = 1e-6, std::size_t restarts = 3);
fp::Result<Kmeans> mini_batch_kmeans(Matrix<double> const &X, std::size_t k,
                                     std::size_t batch, fp::Rng &rng,
                                     std::size_t max_iterations = 300);
fp::Result<Vector<std::size_t>> assign(Kmeans const &model, Matrix<double> const &X);
fp::Result<std::vector<double>> elbow_curve(Matrix<double> const &X, std::size_t max_k, fp::Rng &rng);
}
```

Rules: `restarts` keeps the lowest-inertia run; empty clusters are re-seeded
from the farthest point (never left empty); convergence is "centroids move less
than `tolerance`" **or** `max_iterations`.

Tests: three planted Gaussian blobs are recovered (each centroid within one
sigma of a true center); inertia decreases monotonically across Lloyd
iterations; a restart improves a deliberately bad start; labels are stable for
a seed.

## `cluster/gmm.hpp`

**Job:** Gaussian mixtures by EM, with covariance shapes and information
criteria for model selection.

```cpp
enum class Covariance { Full, Diagonal, Spherical };

struct Gmm {
  Vector<double> weight;        // (k)
  Matrix<double> mean;          // (k, d)
  std::vector<Matrix<double>> covariance;
  double log_likelihood = 0.0;
};

fp::Result<Gmm> gmm(Matrix<double> const &X, std::size_t k, fp::Rng &rng,
                    Covariance type = Covariance::Full,
                    std::size_t max_iterations = 200, double tolerance = 1e-6);
fp::Result<Matrix<double>> responsibilities(Gmm const &model, Matrix<double> const &X);
double bic(Gmm const &model, std::size_t n);
double aic(Gmm const &model, std::size_t n);
```

Rules: covariance matrices get a ridge (`1e-6 * trace`) so they stay
invertible; responsibilities are computed in log space via `logsumexp`; the
log-likelihood must not decrease between EM iterations (a test asserts it).

Tests: a two-component mixture recovers means and weights; the likelihood is
monotone across iterations; BIC prefers the true k on planted data; sampling
from the fitted model reproduces its moments.

## `cluster/dbscan.hpp`

**Job:** density-based clustering with explicit noise.

```cpp
struct Dbscan {
  Vector<int> labels;   // -1 = noise, else 0..k-1
  std::size_t clusters = 0;
};

fp::Result<Dbscan> dbscan(Matrix<double> const &X, double eps, std::size_t min_pts);
```

Rules: a core point has ≥ `min_pts` neighbors within `eps`; border points join
a core's cluster; everything else is noise. Region queries go through the
`neighbors` k-d tree when the metric is L2.

Tests: two dense blobs plus scattered noise produce exactly two clusters and
the noise labeled `-1`; lowering `eps` splits a chain of points; a single
point below `min_pts` is noise.

## `cluster/agglomerative.hpp`

**Job:** bottom-up merging with selectable linkage, and the merge history.

```cpp
enum class Linkage { Single, Complete, Average, Ward };

struct Dendrogram { std::vector<std::pair<std::size_t, std::size_t>> merges; Vector<double> height; };
fp::Result<Dendrogram> agglomerate(Matrix<double> const &X, Linkage linkage);
fp::Result<Vector<std::size_t>> cut(Dendrogram const &tree, std::size_t k);
```

Tests: single linkage merges two close points first (hand distance matrix);
Ward produces compact clusters on planted blobs; cutting at `k` gives exactly
`k` labels; the merge count is `n-1`.

## `cluster/metrics.hpp`

**Job:** internal validation — how good is this clustering, without labels.

```cpp
double silhouette(Matrix<double> const &X, Vector<std::size_t> const &labels);
double davies_bouldin(Matrix<double> const &X, Vector<std::size_t> const &labels);
double calinski_harabasz(Matrix<double> const &X, Vector<std::size_t> const &labels);
```

Rules: silhouette is O(n²) (documented; sampling is the caller's choice);
all three handle singleton clusters explicitly; a single cluster returns 0
rather than dividing by zero.

Tests: the planted k ranks first in a sweep; a perfect split scores higher
than a random labeling; hand-computed silhouette on a small example.

## Gate

k-means recovers planted blobs, GMM's likelihood is monotone and BIC picks the
true k, DBSCAN separates blobs from noise, and the internal metrics rank the
true `k` first.

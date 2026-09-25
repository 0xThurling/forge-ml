# reduction — fewer dimensions, same structure

Projections that make data smaller, cheaper and often easier to model: PCA and
its kernel variant, Fisher's LDA, random projections (with the Johnson–
Lindenstrauss guarantee), and NMF when the parts should stay non-negative.
Everything is fit on the training split and applied to the rest.

Depends on: `core`, `math/stats`, `prob`, ForgeFP.

Files: `pca.hpp`, `kernel_pca.hpp`, `lda.hpp`, `projections.hpp`, `nmf.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `pca.hpp` | `fp::linalg` (`matmul`, `transpose`, `mean`), `fp::grid`, `fp::sort_by_cached` |
| `kernel_pca.hpp` | `fp::linalg` (`matmul`, `transpose`), `fp::grid`, `fp::map2d` |
| `lda.hpp` | `fp::linalg` (`matmul`, `transpose`, `solve`), `fp::grid` |
| `projections.hpp` | `fp::Rng` (`normal`, `bernoulli`), `fp::linalg::matmul` |
| `nmf.hpp` | `fp::linalg` (`matmul`, `transpose`), `fp::inplace`, `fp::Rng` |

---

## `reduce/pca.hpp`

**Job:** principal components from the covariance's eigendecomposition, with
explained variance, projection, inverse and whitening.

```cpp
namespace forgeml {

struct Pca {
  Vector<double> mean;         // (d)
  Matrix<double> components;   // (k, d) rows = principal axes
  Vector<double> explained;    // (k) variances
  double total_variance = 0.0;
};

fp::Result<Pca> fit_pca(Matrix<double> const &X, std::size_t components = 0,
                        bool whiten = false);
Matrix<double> transform(Pca const &pca, Matrix<double> const &X);
Matrix<double> inverse_transform(Pca const &pca, Matrix<double> const &Z);
Vector<double> explained_variance_ratio(Pca const &pca);
fp::Result<std::size_t> components_for_variance(Pca const &pca, double target);
}
```

Rules: `components = 0` keeps all; components are sorted by explained variance
and **sign-fixed** (largest absolute loading positive) so results are
reproducible across libraries; `transform` subtracts the training mean;
`inverse_transform` adds it back; whitening divides by `sqrt(explained)`.

Tests: a 5-D dataset with a planted 2-D subspace keeps ≥ 95% variance in two
components; `transform` → `inverse_transform` round-trips within tolerance;
`components_for_variance(0.95)` picks the planted rank; a permuted column
order gives the same components.

## `reduce/kernel_pca.hpp`

**Job:** PCA in a feature space defined by a kernel, without ever forming it.

```cpp
struct KernelPca {
  Matrix<double> X;            // training rows (for the kernel)
  Matrix<double> alpha;        // (n, k) eigenvectors of the centered Gram matrix
  Vector<double> lambda;       // eigenvalues
  double gamma = 1.0;          // RBF width
  enum class Kernel { Rbf, Linear } kernel = Kernel::Kernel::Rbf;
};

fp::Result<KernelPca> fit_kernel_pca(Matrix<double> const &X, std::size_t components,
                                     KernelPca::Kernel kernel = KernelPca::Kernel::Rbf,
                                     double gamma = 1.0);
Matrix<double> transform(KernelPca const &pca, Matrix<double> const &X);
```

Rules: the Gram matrix is centered (`K - 1ₙK - K1ₙ + 1ₙK1ₙ`); components are
normalised by `sqrt(lambda)`; a linear kernel must reproduce `pca`'s
components up to sign.

Tests: RBF kernel PCA separates two concentric rings in one component (the
classic non-linear case); linear kernel PCA matches `pca`; centering makes the
projected training data zero-mean.

## `reduce/lda.hpp`

**Job:** Fisher's linear discriminant — supervised projection that maximizes
between-class over within-class scatter.

```cpp
struct Lda {
  Matrix<double> components;   // (k, d), k <= classes - 1
  Vector<double> classes;
};

fp::Result<Lda> fit_lda(Matrix<double> const &X, Vector<double> const &y,
                        std::size_t components = 0);
Matrix<double> transform(Lda const &lda, Matrix<double> const &X);
```

Rules: at most `classes - 1` components; the within-class scatter gets a ridge
before `solve`; class means and priors are returned for inspection.

Tests: two planted classes separate along the single component (means far
apart relative to spread); a third class needs two components; LDA beats PCA
on a dataset whose variance is orthogonal to the class direction.

## `reduce/projections.hpp`

**Job:** random projections with the Johnson–Lindenstrauss bound made
checkable.

```cpp
Matrix<double> gaussian_projection(std::size_t d, std::size_t k, fp::Rng &rng);
Matrix<double> sparse_projection(std::size_t d, std::size_t k, double density, fp::Rng &rng);
double jl_epsilon(std::size_t n, std::size_t k);          // the bound
fp::Result<double> max_distance_distortion(Matrix<double> const &X, Matrix<double> const &P);
```

Rules: Gaussian entries are `N(0, 1/k)`; sparse entries are
`±sqrt(1/(k*density))` with the given density; `max_distance_distortion`
reports the worst relative change in pairwise distance so a test can compare
it with `jl_epsilon`.

Tests: distortion stays under `jl_epsilon(n, k)` for a random matrix;
`k = d` is (nearly) an isometry; sparse and dense projections have comparable
distortion at the same `k`.

## `reduce/nmf.hpp`

**Job:** non-negative matrix factorization by multiplicative updates.

```cpp
struct Nmf { Matrix<double> W, H; double reconstruction_error = 0.0; };

fp::Result<Nmf> nmf(Matrix<double> const &X, std::size_t components, fp::Rng &rng,
                    std::size_t max_iterations = 200, double tolerance = 1e-6);
Matrix<double> reconstruct(Nmf const &model);
```

Rules: inputs must be non-negative (validated); W and H start from a seeded
uniform draw; the update is the standard Lee–Seung rule with a small epsilon
to avoid division by zero.

Tests: reconstruction error decreases monotonically; a rank-2 non-negative
matrix is recovered within tolerance; negative input errors.

## Gate

PCA keeps ≥ 95% of the variance of a planted 2-D subspace in two components
and round-trips; kernel PCA separates a non-linear shape that linear PCA
cannot; LDA beats PCA when the class direction is orthogonal to the variance;
random projections stay within the JL bound; NMF's error is monotone.

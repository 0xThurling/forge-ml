# math — the ML remainder

Almost all of the math a model needs already lives in ForgeFP:

| Need | ForgeFP |
|---|---|
| matrix/vector products, solve | `fp::linalg` (`matmul`, `matvec`, `outer`, `solve`, `transpose`) |
| reductions | `fp::linalg` (`dot`, `norm_l1/l2`, `argmax`, `mean`, `variance`, `row_sums`, `col_sums`, `row_means`, `col_means`, `argmax_rows`) |
| stable probabilities | `fp::numerics` (`softmax`, `softmax_rows`, `log_softmax`, `logsumexp`, `sigmoid`, `relu`, `clamp`) |
| elementwise math | `fp::ops` (`abs`, `sqrt`, `exp`, `log`, `log1p`, `sign`, `min_`, `max_`, `pow`, `clamp`) |
| numerical derivatives | `fp::central_difference`, `fp::ad::derivative` |
| grid math | `fp::grid` (`map2d`, `map2d_indexed`, `map2d_inplace`, `for_each_index`) |

What is left for `ml/math/` is only what is *about machine learning*:
activations that fp does not define, loss functions, and impurity/statistics
used by models and scalers.

Depends on: `core`, ForgeFP.

Files: `activations.hpp`, `losses.hpp`, `stats.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `activations.hpp` | `fp::map`, `fp::transform_inplace`, `fp::exp`, `fp::sigmoid`, `fp::relu` |
| `losses.hpp` | `fp::zip_with`, `fp::fold_left`, `fp::sum`, `fp::dot`, `fp::log_softmax`, `fp::softmax`, `fp::clamp`, `fp::max_`, `fp::transform_inplace`, `fp::central_difference` (tests) |
| `stats.hpp` | `fp::mean/variance`, `fp::fold_left`, `fp::count_if`, `fp::map`, `fp::sort`, `fp::for_each`, `fp::dot` |

---

## `math/activations.hpp`

**Job:** the ML-specific nonlinearities and their derivatives. The general ones
are already in fp and are *not* duplicated here.

```cpp
namespace forgeml {

// Provided by fp::numerics (documented here for convenience):
//   fp::sigmoid(z), fp::relu(z), fp::clamp(v, lo, hi)
//   fp::softmax(r), fp::softmax_rows(g), fp::log_softmax(r)

inline double leaky_relu(double z, double alpha = 0.01);
inline double leaky_relu_derivative(double z, double alpha = 0.01);

inline double gelu(double z);                 // tanh approximation
inline double gelu_derivative(double z);

// Vectorized forms (thin wrappers over fp::map).
Vector<double> gelu(Vector<double> const &z);
Vector<double> leaky_relu(Vector<double> const &z, double alpha = 0.01);
}
```

Formulas:

```text
leaky_relu(z)  = z > 0 ? z : alpha * z
gelu(z)        ~= 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3)))
```

Rules:

- Only activations fp lacks live here. Adding `sigmoid` here would violate the
  division of labour.
- Derivatives take the **pre-activation** `z` (not the output), because
  `gelu`/`leaky_relu` derivatives are cheapest that way.
- Vectorized forms use `fp::map`; they return a new vector. In-place callers use
  `fp::transform_inplace`.

Tests (`test/activations_test.cpp`):

- `leaky_relu(-1) == -0.01`, `gelu(0) == 0`
- vectorized wrappers equal the scalar versions elementwise
- finite differences via `fp::central_difference` match the analytic
  derivatives within `1e-6`

---

## `math/losses.hpp`

**Job:** loss values and their gradients with respect to predictions. Losses
that take probabilities clip for stability; losses that take logits use the
stable fused form (`fp::log_softmax`). Every loss exposes `value` and `grad`.

```cpp
namespace forgeml {

struct LossResult {
  double value;
  Vector<double> grad;      // dL/d(prediction)
};

// Regression
LossResult mse(Vector<double> const &y_pred, Vector<double> const &y_true);
LossResult mae(Vector<double> const &y_pred, Vector<double> const &y_true);

// Binary classification: p in (0,1), y in {0,1}
LossResult binary_cross_entropy(Vector<double> const &p, Vector<double> const &y);

// Multiclass: logits (n x k), y_true are class indices
double categorical_cross_entropy(Matrix<double> const &logits,
                                 std::vector<int> const &y_true);
Matrix<double> categorical_cross_entropy_grad(Matrix<double> const &logits,
                                              std::vector<int> const &y_true);

// Margin: y in {-1, +1}, scores = w^T x
LossResult hinge(Vector<double> const &scores, Vector<double> const &y);

// Regularization (value only; gradients are added by the optimizer/model)
double l1_penalty(Vector<double> const &w, double lambda);
double l2_penalty(Vector<double> const &w, double lambda);
Vector<double> l1_grad(Vector<double> const &w, double lambda);   // lambda * sign(w)
Vector<double> l2_grad(Vector<double> const &w, double lambda);   // 2 * lambda * w
}
```

Formulas and gradients:

```text
MSE:  L = (1/n) sum (y_hat_i - y_i)^2          dL/dy_hat_i = (2/n)(y_hat_i - y_i)
MAE:  L = (1/n) sum |y_hat_i - y_i|            dL/dy_hat_i = sign(y_hat_i - y_i)/n
BCE:  L = -(1/n) sum [y log p + (1-y) log(1-p)]
      dL/dp_i = (p_i - y_i) / (n * p_i * (1 - p_i))
CCE (logits z, target t):
      L = -(1/n) sum_i log_softmax(z_i)[t_i]
      dL/dz_ij = (softmax(z_i)_j - 1[j == t_i]) / n
Hinge: L = (1/n) sum max(0, 1 - y_i * s_i)
      dL/ds_i = -y_i / n if y_i * s_i < 1 else 0
```

Implementation notes (fp-first):

- MSE/MAE: `fp::zip_with` for the residuals, `fp::fold_left`/`fp::sum` for the
  mean, `fp::transform_inplace` for the gradient scaling.
- BCE: clip `p` with `fp::clamp(p, 1e-12, 1 - 1e-12)` before `fp::log`;
  gradients use the logit form `(p - y)/n`, so no division by `p(1-p)`.
- CCE: `fp::log_softmax` per row (or `fp::softmax_rows` for the gradient),
  never `log(softmax(x))`.
- Hinge: `fp::max_(0.0)` for the margin term, `fp::fold_left` for the mean.
- Regularization: `fp::norm_l1`/`fp::norm_l2` from `fp::linalg` for values;
  gradients via `fp::map(w, fp::sign)` (L1) and `fp::scale(w, 2 * lambda)` (L2).
- Empty inputs assert; `n` is `size()`.

Tests (`test/losses_test.cpp`):

- MSE of predictions `{2,4}` vs truth `{2,2}` is 2; grad is `(0, 2)` for `n=2`
- BCE at `p = y` is near 0; at `p = 1 - y` is large but finite
- CCE of uniform logits over 2 classes is `log 2`; grad rows sum to 0
- Hinge is 0 when all margins >= 1; subgradient correct at the boundary
- Every gradient matches `fp::central_difference` (`h = 1e-7`) within `1e-6`

---

## `math/stats.hpp`

**Job:** the statistics that are specific to ML workflows: impurity measures for
decision trees, class counts, and the per-column variance/stddev that feature
scaling needs. Everything generic (`mean`, `variance`, `argmax`, sums) is
`fp::linalg`.

```cpp
namespace forgeml {

// Classification helpers
std::vector<int> unique_sorted(std::vector<int> const &labels);
std::vector<std::size_t> class_counts(std::vector<int> const &labels,
                                      std::size_t num_classes);

// Impurity (decision trees). Probabilities are class frequencies.
double gini_impurity(std::vector<double> const &class_probs);
double entropy(std::vector<double> const &class_probs);          // base 2
double information_gain(double parent_entropy,
                        std::vector<double> const &child_weights,
                        std::vector<double> const &child_entropies);

// Regression-tree impurity
double variance_reduction(double parent_variance,
                          std::size_t n_left, double left_variance,
                          std::size_t n_right, double right_variance);

// The one generic-statistics gap in fp today: per-column variance/stddev.
Vector<double> col_variances(Matrix<double> const &m, std::size_t ddof = 0);
Vector<double> col_stddevs(Matrix<double> const &m, std::size_t ddof = 0);

// Convenience re-exports of the fp reductions (thin, no reimplementation).
inline double mean(Vector<double> const &v) { return fp::mean(v); }
inline double variance(Vector<double> const &v, std::size_t ddof = 0) {
  return fp::variance(v, ddof);
}
}
```

Rules:

- `entropy`/`gini` ignore zero-probability classes (`0 * log 0 = 0`) and
  `ML_ASSERT` probabilities are in `[0,1]` and sum to ~1 (`fp::approx_equal`).
- `col_variances` is built from `fp::col_means` plus one pass over the grid
  (`fp::for_each_index`); it is a candidate for a future `fp::linalg`
  extension, so it is written as a thin function with no ML concepts.
- `unique_sorted` uses `fp::sort` + `fp::unique`; `class_counts` uses
  `fp::count` per class (or one pass with a `std::vector<std::size_t>`).

Formulas:

```text
Gini            = 1 - sum_k p_k^2
Entropy         = -sum_k p_k log2 p_k
InformationGain = H(parent) - sum_i (n_i / n) H(child_i)
col_variance_j  = (1/(n-ddof)) sum_i (m[i][j] - mean_j)^2
```

Tests (`test/stats_test.cpp`):

- entropy of `{0.5, 0.5}` is 1; gini is 0.5; pure class entropy is 0
- information gain of a perfect split equals the parent entropy
- `col_variances` matches hand-computed per-column values and agrees with
  `fp::variance` on a single column
- `class_counts` sums to the number of labels

---

## Where the old `linalg.hpp` / `calc.hpp` went

| Old file | Replacement |
|---|---|
| `math/linalg.hpp` | `fp::linalg` (`matmul`, `matvec`, `outer`, `solve`, `hadamard`, `scale`, `add_row_broadcast`, `dot`, norms, `argmax`, `mean`, `variance`, row/column reductions) |
| `math/calc.hpp` | `fp::central_difference`; for exact derivatives `fp::ad::derivative`; the gradient-check harness is `utils/gradient_check.hpp` |

Shape preconditions are documented and asserted in debug inside `fp::linalg`;
domain code validates shapes at its own boundary (see
[conventions](conventions.md#error-model)) and then calls fp directly.

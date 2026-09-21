# model — classical algorithms

Each model owns its parameters, implements `fit` / `predict`, and delegates all
shared work to `core`, `math`, `data`, `optim`, `metrics` — and to ForgeFP for
the math. A model file contains: a config struct, the model class, learned
parameters, `fit`, `predict`, and small private helpers. Nothing else.

Depends on: `core`, `math`, `data`, `optim`, `metrics`, ForgeFP.

Files: `model.hpp`, `linear_regression.hpp`, `logistic_regression.hpp`,
`naive_bayes.hpp`, `decision_tree.hpp`, `svm.hpp`.

**ForgeFP usage at a glance**

| Model | ForgeFP functions |
|---|---|
| `linear_regression` | `fp::matvec`, `fp::matmul`, `fp::transpose`, `fp::solve`, `fp::zip_with`, `fp::mean`, `fp::axpy_inplace`, `fp::transform_inplace` |
| `logistic_regression` | `fp::sigmoid`, `fp::matvec`, `fp::matmul`, `fp::transpose`, `fp::transform_inplace`, `fp::mean` |
| `naive_bayes` | `fp::mean/variance`, `fp::softmax`, `fp::log`, `fp::logsumexp`, `fp::group_by`, `fp::argmax`, `fp::map` |
| `decision_tree` | `fp::fix` (recursive build), `fp::sort_by`, `fp::views::enumerate`, `fp::count_if`, `fp::remove_if` |
| `svm` | `fp::matvec`, `fp::zip_transform_inplace`, `fp::scale`, `fp::fold_left`, `fp::axpy_inplace` |
| `model.hpp` | `fp::Result`, `fp::fail`, `fp::mean`, `fp::approx_equal` |

---

## `model/model.hpp`

**Job:** the two interfaces every supervised model implements.

```cpp
namespace forgeml {

class Regressor {
public:
  virtual ~Regressor() = default;
  virtual fp::Result<void> fit(Matrix<double> const &X, Vector<double> const &y) = 0;
  virtual Vector<double> predict(Matrix<double> const &X) const = 0;
};

class Classifier {
public:
  virtual ~Classifier() = default;
  virtual fp::Result<void> fit(Matrix<double> const &X, Vector<int> const &y) = 0;
  virtual std::vector<int> predict(Matrix<double> const &X) const = 0;
  // Probabilities: (n, k). Binary models return (n, 2).
  virtual fp::Result<Matrix<double>> predict_proba(Matrix<double> const &X) const = 0;
};

// Shared helpers implemented once:
fp::Result<void> check_fit_input(Matrix<double> const &X, std::size_t n_labels);
double score_regression(Regressor const &m, Matrix<double> const &X,
                        Vector<double> const &y);          // R^2
double score_classification(Classifier const &m, Matrix<double> const &X,
                            std::vector<int> const &y);    // accuracy
}
```

Rules:

- `fit` returns `Result` for bad input; it does not throw.
- `predict` on an unfitted model asserts (bug, not input error).
- `predict_proba` is mandatory for classifiers; logistic regression and Naive
  Bayes provide real probabilities, SVM provides a squashed margin.
- No model stores the training data (except decision trees, which store the
  tree, not `X`).

---

## `model/linear_regression.hpp`

```cpp
namespace forgeml {

struct LinearRegressionConfig {
  double learning_rate = 0.01;
  std::size_t epochs = 1000;
  bool fit_intercept = true;
  enum class Penalty { None, L1, L2 } penalty = Penalty::None;
  double lambda = 0.0;
  double tolerance = 1e-8;          // stop when loss change < tolerance
  bool closed_form = false;         // normal equations instead of GD
  std::size_t log_every = 0;        // 0 = silent
};

class LinearRegression : public Regressor {
public:
  explicit LinearRegression(LinearRegressionConfig config = {});

  fp::Result<void> fit(Matrix<double> const &X, Vector<double> const &y) override;
  Vector<double> predict(Matrix<double> const &X) const override;

  Vector<double> const &weights() const;
  double bias() const;
  std::vector<double> const &loss_history() const;
  void zero_grad();

private:
  LinearRegressionConfig config_;
  Vector<double> w_, grad_w_;
  double b_ = 0.0, grad_b_ = 0.0;
  bool fitted_ = false;
  std::vector<double> loss_history_;
};
}
```

Training loop:

```text
for epoch in 1..epochs:
    y_hat = X w + b
    err   = y_hat - y
    grad_w = (2/n) X^T err + reg_grad(w)
    grad_b = (2/n) sum(err)
    w <- w - lr * grad_w        (via optimizer)
    b <- b - lr * grad_b
    if |loss(epoch) - loss(epoch-1)| < tolerance: stop
```

Closed form (`closed_form = true`):

```text
Xb = [1 | X]                      (if fit_intercept)
w_full = fp::solve(Xb^T Xb + lambda I, Xb^T y)   (ridge when lambda > 0)
```

Rules:

- `fit` fails when `X.rows() != y.size()`, `X` is empty, or values are
  non-finite.
- Regularization does not penalize the intercept.
- L1 uses the subgradient `lambda * sign(w)`.
- `loss_history` records MSE (without penalty) per logged epoch for plotting.

Tests (`test/linear_regression_test.cpp`):

- recovers `y = 3x + 2` from noise-free data within `1e-6`
- with noise, held-out R² > 0.99 (fixed seed)
- `closed_form` and GD reach the same weights within `1e-4` on a small problem
- L2 shrinks weights relative to unregularized fit
- mismatched rows returns `fail`

---

## `model/logistic_regression.hpp`

```cpp
namespace forgeml {

struct LogisticRegressionConfig {
  double learning_rate = 0.1;
  std::size_t epochs = 500;
  bool fit_intercept = true;
  enum class Penalty { None, L1, L2 } penalty = Penalty::None;
  double lambda = 0.0;
  double threshold = 0.5;
  double tolerance = 1e-8;
  std::size_t log_every = 0;
};

class LogisticRegression : public Classifier {
public:
  explicit LogisticRegression(LogisticRegressionConfig config = {});

  fp::Result<void> fit(Matrix<double> const &X, Vector<int> const &y) override;
  std::vector<int> predict(Matrix<double> const &X) const override;
  fp::Result<Matrix<double>> predict_proba(Matrix<double> const &X) const override;

  Vector<double> const &weights() const;
  double bias() const;
  std::vector<double> const &loss_history() const;
  void zero_grad();

private:
  LogisticRegressionConfig config_;
  Vector<double> w_, grad_w_;
  double b_ = 0.0, grad_b_ = 0.0;
  bool fitted_ = false;
  std::vector<double> loss_history_;
};
}
```

Training loop:

```text
for epoch:
    z = X w + b
    p = sigmoid(z)
    grad_w = (1/n) X^T (p - y) + reg_grad(w)
    grad_b = (1/n) sum(p - y)
    w <- w - lr * grad_w ; b <- b - lr * grad_b
```

Rules:

- Labels must be exactly `{0, 1}`; anything else returns
  `fp::fail("logistic_regression: labels must be 0/1")`.
- BCE uses the clipped stable form; gradients use the logit form
  `(p - y) / n` (no division by `p(1-p)`).
- `predict_proba` returns `(n, 2)`: `[1-p, p]`.
- `predict` uses `p >= threshold`.

Tests (`test/logistic_regression_test.cpp`):

- separable 2-D blobs reach > 95% accuracy with a fixed seed
- probabilities are in `[0,1]` and sum to 1 per row
- non-binary labels return `fail`
- L2 regularization keeps `||w||` smaller than unregularized on the same data
- gradient matches finite differences at the first step

---

## `model/naive_bayes.hpp`

```cpp
namespace forgeml {

struct GaussianNaiveBayesConfig {
  double var_smoothing = 1e-9;     // epsilon added to variances
  bool use_log = true;             // log-space prediction (default)
};

class GaussianNaiveBayes : public Classifier {
public:
  explicit GaussianNaiveBayes(GaussianNaiveBayesConfig config = {});

  fp::Result<void> fit(Matrix<double> const &X, Vector<int> const &y) override;
  std::vector<int> predict(Matrix<double> const &X) const override;
  fp::Result<Matrix<double>> predict_proba(Matrix<double> const &X) const override;

  std::size_t num_classes() const;
  Vector<double> const &priors() const;
  Matrix<double> const &means() const;       // (classes, features)
  Matrix<double> const &variances() const;   // (classes, features)

private:
  GaussianNaiveBayesConfig config_;
  std::size_t num_classes_ = 0;
  Vector<double> priors_;
  Matrix<double> means_, variances_;
  bool fitted_ = false;
};
}
```

Training (one pass, no gradients):

```text
for each class c:
    Xc       = rows of X where y == c
    prior_c  = |Xc| / n
    mean_cj  = mean of feature j over Xc
    var_cj   = variance of feature j over Xc + var_smoothing
```

Prediction:

```text
log P(c | x) ∝ log prior_c + sum_j log N(x_j; mean_cj, var_cj)
N(x; mu, s2) = exp(-(x-mu)^2 / (2 s2)) / sqrt(2 pi s2)
predict = fp::argmax over the log posteriors (ties -> smallest c)
predict_proba = fp::softmax(log posteriors)
log posterior_c = fp::log(prior_c) + sum_j gaussian_log_density(x_j)
```

Rules:

- Labels must be `0..k-1`; `fit` derives `k` from the data and fails on gaps
  (e.g. only labels `{0,2}`).
- Log-space only for `predict`/`predict_proba`; the raw density is never
  exponentiated into a product.
- `var_smoothing` handles zero variance (constant features).

Tests (`test/naive_bayes_test.cpp`):

- two well-separated 1-D Gaussians are classified correctly
- a constant feature does not produce NaN (variance smoothing)
- priors equal class frequencies
- probabilities sum to 1 per row
- labels with gaps return `fail`

---

## `model/decision_tree.hpp`

**Job:** CART classification tree. The first structure-heavy model; it owns a
tree, not a weight vector.

```cpp
namespace forgeml {

enum class TreeCriterion { Gini, Entropy };

struct DecisionTreeConfig {
  TreeCriterion criterion = TreeCriterion::Gini;
  std::size_t max_depth = 10;
  std::size_t min_samples_split = 2;
  std::size_t min_samples_leaf = 1;
  double min_impurity_decrease = 0.0;
  std::size_t max_features = 0;    // 0 = all features
};

struct TreeNode {
  bool is_leaf = false;
  std::size_t feature = 0;
  double threshold = 0.0;
  std::size_t left = 0, right = 0;   // indices into the node pool
  int prediction = 0;                // leaf class
  double impurity = 0.0;
  std::size_t n_samples = 0;
};

class DecisionTree : public Classifier {
public:
  explicit DecisionTree(DecisionTreeConfig config = {});

  fp::Result<void> fit(Matrix<double> const &X, Vector<int> const &y) override;
  std::vector<int> predict(Matrix<double> const &X) const override;
  fp::Result<Matrix<double>> predict_proba(Matrix<double> const &X) const override;

  std::vector<TreeNode> const &nodes() const;   // node 0 is the root
  std::size_t depth() const;

private:
  DecisionTreeConfig config_;
  std::vector<TreeNode> nodes_;
  std::size_t num_classes_ = 0;
  bool fitted_ = false;

  std::size_t build(Matrix<double> const &X, Vector<int> const &y,
                    std::vector<std::size_t> const &idx, std::size_t depth);
  // best split search over features/thresholds
};
}
```

Split search:

```text
for each candidate feature f:
    sort unique values of X[:, f] (or use quantiles for large n)
    for each midpoint threshold t:
        left  = {i : X[i][f] <= t}, right = rest
        if |left| < min_samples_leaf or |right| < min_samples_leaf: skip
        gain = impurity(parent) - (|left|/n) impurity(left) - (|right|/n) impurity(right)
    keep the (f, t) with the largest gain
stop when: depth == max_depth, n < min_samples_split, node is pure,
           or gain < min_impurity_decrease
leaf prediction = majority class (ties -> smallest class)
```

- Use `fp::fix` for the recursion if it reads better; otherwise a private
  `build` method with a node pool (`std::vector<TreeNode>`), never raw
  `new`/`delete`.
- Impurity helpers come from `math/stats.hpp` (`gini_impurity`, `entropy`,
  `information_gain`).
- `predict_proba` returns the leaf class distribution.

Tests (`test/decision_tree_test.cpp`):

- a hand-made XOR dataset is learned with depth >= 2 (accuracy 100%)
- `max_depth = 0` yields a single leaf predicting the majority class
- `min_samples_leaf` is respected by every leaf
- a pure dataset stops with one leaf
- deterministic across runs (no RNG in the tree)

---

## `model/svm.hpp`

**Job:** linear SVM with hinge loss and L2 regularization, trained by
subgradient descent. Labels are `{-1, +1}`.

```cpp
namespace forgeml {

struct SvmConfig {
  double learning_rate = 0.01;
  std::size_t epochs = 1000;
  double lambda = 1e-4;          // L2 strength
  bool fit_intercept = true;
  double tolerance = 1e-8;
  std::size_t log_every = 0;
};

class LinearSvm : public Classifier {
public:
  explicit LinearSvm(SvmConfig config = {});

  fp::Result<void> fit(Matrix<double> const &X, Vector<int> const &y) override;
  std::vector<int> predict(Matrix<double> const &X) const override;
  fp::Result<Matrix<double>> predict_proba(Matrix<double> const &X) const override;

  Vector<double> const &weights() const;
  double bias() const;
  double margin() const;                 // 2 / ||w||
  std::vector<double> const &loss_history() const;
  void zero_grad();

private:
  SvmConfig config_;
  Vector<double> w_, grad_w_;
  double b_ = 0.0;
  bool fitted_ = false;
  std::vector<double> loss_history_;
};
}
```

Objective and subgradient:

```text
L(w,b) = (1/n) sum_i max(0, 1 - y_i (w^T x_i + b)) + lambda ||w||^2

per sample i:
    margin_i = y_i (w^T x_i + b)
    if margin_i < 1:
        grad_w += (1/n)(-y_i x_i) + 2 lambda w
        grad_b += (1/n)(-y_i)
    else:
        grad_w += 2 lambda w
```

Rules:

- `fit` maps labels `{-1,+1}` or `{0,1}` to signs internally; store the mapping
  in the model so `predict` returns the original labels.
- `predict_proba` returns a two-column matrix from the sigmoid of the margin
  (documented as a heuristic, not a calibrated probability).
- `margin()` asserts `||w|| > 0`.

Tests (`test/svm_test.cpp`):

- perfectly separable data reaches 100% training accuracy
- `||w||` decreases (margin increases) over epochs on separable data
- one subgradient step matches the formula by hand
- labels outside the supported set return `fail`

---

## Shared rules for all models

- One `fit` = one deterministic procedure for a fixed seed; no hidden RNG.
- `zero_grad()` exists on every gradient-trained model and resets both weights
  and bias gradients.
- `loss_history_` is recorded only when `log_every > 0` (or every epoch; pick
  one and document it in the model doc).
- Serialization is external: `utils/serialization.hpp` reads `weights()` /
  config; models do not implement `save`/`load` themselves.
- Metrics are never computed inside `fit`; callers use `metrics/` or
  `eval/scoring.hpp`.

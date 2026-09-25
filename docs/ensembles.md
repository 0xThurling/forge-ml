# ensembles — many weak learners, one strong one

Bagging, forests, boosting and stacking, all built from the base learners in
`model/` and `cluster/` (CART today, anything with `fit`/`predict` tomorrow).
Independence makes this layer the natural home of `fp::par_map`.

Depends on: `core`, `model` (CART, linear), `prob`, `neighbors`, ForgeFP.

Files: `bagging.hpp`, `forest.hpp`, `boosting.hpp`, `stacking.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `bagging.hpp` | `fp::Rng::sample_indices`, `fp::par_map`, `fp::views::chunk` |
| `forest.hpp` | `fp::Rng`, `fp::par_map`, `fp::sort_by_cached`, `fp::map_values` |
| `boosting.hpp` | `fp::Rng`, `fp::linalg` (`dot`), `fp::inplace`, `fp::map` |
| `stacking.hpp` | `fp::views::chunk`, `fp::linalg::matmul`, `fp::Rng` |

---

## `ensemble/bagging.hpp`

**Job:** bootstrap aggregation around any base learner, with out-of-bag error.

```cpp
namespace forgeml {

template <class Learner>
struct Bagging {
  std::vector<Learner> members;
  std::vector<std::vector<std::size_t>> samples;   // in-bag indices
  bool classification = false;
};

template <class Learner, class Fit>
fp::Result<Bagging<Learner>> bag(Matrix<double> const &X, Vector<double> const &y,
                                 Fit &&fit_one, std::size_t n_members,
                                 double sample_ratio, fp::Rng &rng);
template <class Learner>
double oob_error(Bagging<Learner> const &bag, Matrix<double> const &X, Vector<double> const &y);
template <class Learner>
Vector<double> predict(Bagging<Learner> const &bag, Matrix<double> const &X);
}
```

Rules: samples are drawn with replacement (`fp::Rng::sample_indices`); a
classification ensemble votes (majority, ties by class order), a regression
ensemble averages; OOB uses only the members that did **not** see a row.

Tests: OOB error tracks a held-out error within a few points; a bagged tree
beats a single tree on planted noisy data; a 100% sample ratio makes members
identical (documented, not an error).

## `ensemble/forest.hpp`

**Job:** random forests — bagging plus per-split feature subsampling — with
importances.

```cpp
struct Forest {
  std::vector<Cart> trees;
  bool classification = false;
  std::size_t n_features_sampled = 0;
};

fp::Result<Forest> random_forest(Matrix<double> const &X, Vector<double> const &y,
                                 std::size_t n_trees, std::size_t max_depth,
                                 std::size_t min_samples_leaf, double feature_fraction,
                                 fp::Rng &rng, bool classification);
Vector<double> predict(Forest const &forest, Matrix<double> const &X);
Vector<double> predict_proba(Forest const &forest, Matrix<double> const &X, std::size_t classes);
Vector<double> feature_importance(Forest const &forest, Matrix<double> const &X, Vector<double> const &y,
                                  std::size_t n_repeats, fp::Rng &rng);
double oob_error(Forest const &forest, Matrix<double> const &X, Vector<double> const &y);
```

Rules: `feature_fraction` defaults to `sqrt(d)` for classification and `d/3`
for regression when passed as 0; importance is **permutation** importance
(`interpret/importance.hpp`), not impurity, so it is comparable across
models; trees are trained with `fp::par_map`.

Tests: the forest beats a single CART on planted noisy data; the planted
feature ranks first in importance; more trees do not hurt (monotone-ish OOB);
predictions are deterministic for a seed.

## `ensemble/boosting.hpp`

**Job:** AdaBoost (stumps) and gradient boosting (regression and
classification) with shrinkage and subsampling.

```cpp
struct AdaBoost { std::vector<DecisionStump> stumps; Vector<double> alpha; };
fp::Result<AdaBoost> adaboost(Matrix<double> const &X, Vector<double> const &y,
                              std::size_t n_estimators, fp::Rng &rng);
Vector<double> predict(AdaBoost const &model, Matrix<double> const &X);

struct GradientBoosting {
  std::vector<Cart> trees;
  double learning_rate = 0.1;
  bool classification = false;
};
fp::Result<GradientBoosting> gradient_boosting(Matrix<double> const &X, Vector<double> const &y,
                                               std::size_t n_estimators, std::size_t max_depth,
                                               double learning_rate, double subsample,
                                               fp::Rng &rng, bool classification);
Vector<double> predict(GradientBoosting const &model, Matrix<double> const &X);
```

Rules: AdaBoost's weights are renormalised each round and a perfect stump ends
training early (zero error); gradient boosting fits each tree to the negative
gradient of the loss (squared error; log-loss for classification) and applies
`learning_rate` shrinkage; `subsample < 1` draws rows without replacement.

Tests: AdaBoost drives training error to zero on separable data; boosting
reduces training loss monotonically; the learning rate trades iterations for
accuracy (a test fixes both and checks the trend); a degenerate base learner
(always one class) is handled without division by zero.

## `ensemble/stacking.hpp`

**Job:** voting, averaging and out-of-fold stacking.

```cpp
struct Voting { std::vector<Cart> members; bool soft = true; };
Vector<double> predict(Voting const &v, Matrix<double> const &X);
fp::Result<Matrix<double>> oof_predictions(std::vector<Model> const &base,
                                           Matrix<double> const &X, Vector<double> const &y,
                                           std::size_t folds, fp::Rng &rng);
fp::Result<LinearRegression> stack(std::vector<Model> const &base, Matrix<double> const &X,
                                   Vector<double> const &y, std::size_t folds, fp::Rng &rng);
```

Rules: out-of-fold predictions are produced by `eval`'s k-fold splitter, so
the meta-learner never sees in-fold predictions; `stack` fits the meta-learner
on those predictions and returns it together with the base models (a small
`Stacked` struct).

Tests: stacking beats the best base learner on planted data; soft voting beats
hard voting when probabilities are informative; OOF predictions never use a
row's own fold (a leak check).

## Gate

A forest beats a single tree on planted noisy data; boosting drives training
error down and handles a degenerate learner; stacking beats its best base
learner; importances rank a planted feature first.

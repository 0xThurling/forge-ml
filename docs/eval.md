# eval — comparing and tuning models

Higher-level workflow code: it uses models, data, and metrics together. Nothing
here is used by a model; this is the outermost classical-ML layer.

Depends on: `data`, `metrics`, `model`, `core`, ForgeFP.

Files: `scoring.hpp`, `cross_validation.hpp`, `grid_search.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `scoring.hpp` | `fp::Result`, `fp::fail`, `std::function` wrappers only |
| `cross_validation.hpp` | `fp::Rng`, `fp::views::enumerate`, `fp::map`, `fp::fold_left`, `fp::linalg::mean/variance` |
| `grid_search.hpp` | `fp::cartesian_product`, `fp::sort_by`, `fp::str::join`, `fp::views::enumerate`, `fp::map` |

The Cartesian grid is `fp::cartesian_product`; ranking is `fp::sort_by` with
the scorer direction; candidate names are built with `fp::str::join`. No loop
bookkeeping is written by hand here.

---

## `eval/scoring.hpp`

**Job:** one uniform callable shape so CV and grid search do not care which
metric is used, plus the maximize/minimize convention.

```cpp
namespace forgeml {

enum class Direction { Maximize, Minimize };

struct Score {
  double value;
  Direction direction;
};

// Regression: metric takes (y_true, y_pred)
using RegressionMetric = std::function<double(Vector<double> const &,
                                              Vector<double> const &)>;

// Classification: metric takes (y_true, y_pred) labels
using ClassificationMetric = std::function<double(std::vector<int> const &,
                                                  std::vector<int> const &)>;

struct RegressionScorer {
  RegressionMetric metric;
  Direction direction = Direction::Minimize;
  std::string name;
};

struct ClassificationScorer {
  ClassificationMetric metric;
  Direction direction = Direction::Maximize;
  std::string name;
};

// Ready-made scorers (wrap metrics/)
RegressionScorer mse_scorer();          // minimize
RegressionScorer r2_scorer();           // maximize
ClassificationScorer accuracy_scorer(); // maximize
ClassificationScorer f1_scorer();       // maximize (fails -> 0.0)

// Compare two scores honoring direction.
bool is_better(Score candidate, Score best);
}
```

Rules:

- A metric that returns `fp::Result` is wrapped so a failure yields a
  worst-case score (`+inf` for Minimize, `-inf` for Maximize) and is logged,
  never silently treated as 0.
- `name` is used in logs and result tables.

Tests:

- `is_better` respects direction
- `mse_scorer` on a known pair returns the MSE
- a failing metric maps to the worst score

---

## `eval/cross_validation.hpp`

**Job:** k-fold cross-validation driver: split, fit, predict, score, average.

```cpp
namespace forgeml {

template <class Y>
struct FoldResult {
  double score = 0.0;
  std::size_t train_size = 0, test_size = 0;
};

template <class Y>
struct CrossValResult {
  std::vector<FoldResult<Y>> folds;
  double mean = 0.0;
  double stddev = 0.0;
  std::size_t k = 0;
  std::string scorer_name;
};

// Model factory: creates a fresh, unfitted model for each fold.
using RegressorFactory = std::function<std::unique_ptr<Regressor>()>;
using ClassifierFactory = std::function<std::unique_ptr<Classifier>()>;

CrossValResult<double> cross_validate(Dataset<double> const &data,
                                      std::size_t k,
                                      RegressorFactory make_model,
                                      RegressionScorer const &scorer,
                                      fp::Rng &rng);

CrossValResult<int> cross_validate(Dataset<int> const &data,
                                   std::size_t k,
                                   ClassifierFactory make_model,
                                   ClassificationScorer const &scorer,
                                   fp::Rng &rng,
                                   bool stratified = false);

// Convenience: fit a final model on all data after CV.
template <class Model, class Y>
fp::Result<std::unique_ptr<Model>> fit_full(Dataset<Y> const &data,
                                            std::function<std::unique_ptr<Model>()> make);
}
```

Algorithm:

```text
folds = k_fold_indices(n, k, rng)           (or stratified version)
for each fold f:
    train = data.subset(f.train)
    test  = data.subset(f.test)
    model = make_model()                     (fresh per fold)
    model->fit(train.X, train.y)             (Result -> record failure)
    pred  = model->predict(test.X)
    score = scorer.metric(test.y, pred)
    record
mean = average(scores); stddev = population stddev of scores
```

Rules:

- `k >= 2` and `k <= n`; otherwise `fp::fail`.
- A fold whose `fit` fails is recorded as the worst score and reported in the
  result (`FoldResult::fit_error` can be added if needed).
- Stratified CV preserves class ratios in each test fold (uses
  `stratified_k_fold_indices`).
- Standard deviation uses `ddof = 0`.
- The scorer direction is irrelevant to CV itself; it is applied by
  `grid_search`.

Tests (`test/eval_test.cpp`):

- on a fixed dataset, each sample appears in exactly one test fold
- the number of fitted models equals `k`
- `mean`/`stddev` match hand-computed values
- a factory returning a model whose `fit` fails yields the worst score
- stratified CV keeps class proportions within 5% per fold

---

## `eval/grid_search.hpp`

**Job:** brute-force hyperparameter search over CV. Deliberately simple: a list
of named variants, not a type-level framework.

```cpp
namespace forgeml {

// One candidate configuration: a name plus a factory for the model.
template <class Factory>
struct Candidate {
  std::string name;      // e.g. "lr=0.01,lambda=0.001"
  Factory make_model;
};

struct GridResult {
  std::string name;
  double mean = 0.0;
  double stddev = 0.0;
};

// Run every candidate through cross_validate and rank by scorer direction.
template <class Y, class Factory>
std::vector<GridResult> grid_search(Dataset<Y> const &data, std::size_t k,
                                    std::vector<Candidate<Factory>> const &candidates,
                                    auto const &scorer, fp::Rng &rng);

// Helper to build candidates from parameter lists:
//   candidates_from(params, [](double lr, double lambda) { ... })
template <class... Ts, class Make>
auto candidates_from(std::vector<std::tuple<Ts...>> const &grid, Make make);
}
```

Algorithm:

```text
results = []
for each candidate:
    cv = cross_validate(data, k, candidate.make_model, scorer, rng)
    results.push_back({candidate.name, cv.mean, cv.stddev})
sort results by scorer direction (best first)
return results
```

Rules:

- Candidate names must be unique; duplicates are an assert.
- The same `fp::Rng` is used sequentially, so the run is reproducible but folds
  differ between candidates (documented; pass a fresh seed per candidate if
  identical folds are required).
- `candidates_from` uses `fp::cartesian_product` over the parameter lists and
  formats names with `fp::str::join`.
- Return the full ranked table, not just the winner; the caller prints it.

Example usage:

```cpp
std::vector<std::tuple<double, double>> grid = {
    {0.001, 0.0}, {0.01, 0.0}, {0.01, 0.001}, {0.1, 0.01}};

auto candidates = candidates_from(grid, [](double lr, double lambda) {
  return std::make_unique<LogisticRegression>(
      LogisticRegressionConfig{.learning_rate = lr, .lambda = lambda});
});

auto table = grid_search(data, 5, candidates, accuracy_scorer(), rng);
// table[0] is the best configuration
```

Tests (`test/eval_test.cpp`):

- a grid containing a known-best config ranks it first on a synthetic problem
- ranking respects minimize vs maximize
- duplicate names assert

---

## Relationship to `metrics/`

`metrics/` answers "how good is this prediction vector"; `eval/` answers "how
good is this *procedure*". CV and grid search never compute metrics directly —
they call the scorer, which calls `metrics/`. This keeps the dependency arrow
one-way and makes swapping metrics trivial.

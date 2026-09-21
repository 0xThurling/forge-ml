# metrics — measuring model performance

Metrics are pure functions over predictions and truth. They never train, never
touch models, and never throw for bad input: they return `fp::Result` when a
metric is undefined (e.g. precision with no positive predictions).

Depends on: `core`, ForgeFP.

Files: `regression.hpp`, `confusion_matrix.hpp`, `classification.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `regression.hpp` | `fp::zip_with`, `fp::fold_left`, `fp::sum`, `fp::linalg::mean`, `fp::simd::dot`, `fp::approx_equal`, `fp::Result` |
| `confusion_matrix.hpp` | `fp::count_if`, `fp::Result`, `fp::fail` |
| `classification.hpp` | `fp::linalg::argmax_rows`, `fp::sort_by`, `fp::scan`, `fp::count`, `fp::Result` |

All reductions (`sum`, `mean`, `variance`, `dot`) come from fp — metrics only
define the ML-specific formulas on top.

---

## `metrics/regression.hpp`

```cpp
namespace forgeml {

double mse(Vector<double> const &y_true, Vector<double> const &y_pred);
double rmse(Vector<double> const &y_true, Vector<double> const &y_pred);
double mae(Vector<double> const &y_true, Vector<double> const &y_pred);

// R^2 = 1 - SS_res / SS_tot; undefined (fail) when SS_tot == 0.
fp::Result<double> r2_score(Vector<double> const &y_true,
                            Vector<double> const &y_pred);

// Mean absolute percentage error, safe for zero targets:
// mean( |y - yhat| / max(|y|, eps) )
double mape(Vector<double> const &y_true, Vector<double> const &y_pred);

// Residuals helper
Vector<double> residuals(Vector<double> const &y_true,
                         Vector<double> const &y_pred);
}
```

Formulas:

```text
MSE   = (1/n) sum (y_i - yhat_i)^2
RMSE  = sqrt(MSE)
MAE   = (1/n) sum |y_i - yhat_i|
R2    = 1 - sum (y_i - yhat_i)^2 / sum (y_i - mean(y))^2
```

Rules:

- All functions assert `y_true.size() == y_pred.size()`.
- `mse`/`mae` assert non-empty.
- `r2_score` returns `fp::fail("r2_score: constant target has no variance")`
  when `SS_tot < 1e-12`.
- ForgeFP implementation: `residuals` is `fp::zip_with(y_true, y_pred, fp::minus)`;
  the squared/absolute sums use `fp::simd::dot` (squares) or
  `fp::fold_left(..., fp::plus)`; `rmse` is `std::sqrt` of that;
  `r2_score` uses `fp::linalg::mean` for the total sum of squares.

Tests:

- MSE of `{1,2,3}` vs `{1,2,3}` is 0; vs `{2,2,2}` is `2/3`
- MAE of `{1,2,3}` vs `{2,2,2}` is `2/3`
- R² is 1 for perfect predictions and 0 for predicting the mean
- constant target returns `fail`

---

## `metrics/confusion_matrix.hpp`

**Job:** count TP/FP/TN/FN (binary) or a k x k table (multiclass). All other
classification metrics derive from this.

```cpp
namespace forgeml {

struct BinaryCounts {
  std::size_t tp = 0, fp = 0, tn = 0, fn = 0;
  std::size_t total() const { return tp + fp + tn + fn; }
};

// Labels are 0/1. positive_label selects which class is "positive".
fp::Result<BinaryCounts> binary_counts(std::vector<int> const &y_true,
                                       std::vector<int> const &y_pred,
                                       int positive_label = 1);

// Multiclass k x k table: matrix[true][pred].
fp::Result<Matrix<std::size_t>> confusion_matrix(
    std::vector<int> const &y_true, std::vector<int> const &y_pred,
    std::size_t num_classes);

// Derived counts for one class c treated as positive (one-vs-rest).
BinaryCounts one_vs_rest_counts(Matrix<std::size_t> const &cm,
                                std::size_t c);
}
```

Rules:

- Assert equal sizes; `fail` when a label is outside `[0, num_classes)`.
- `confusion_matrix` rows sum to the true class counts.
- No floating point here; pure integer counting. Build the table with one pass
  (`fp::for_each_index`/`fp::zip_for_each`) or `fp::count_if` per cell; the
  domain part is the TP/FP/TN/FN vocabulary, not the counting loop.

Tests:

- hand-computed binary counts from a small list
- multiclass table for 3 classes; row sums equal class counts
- label out of range returns `fail`

---

## `metrics/classification.hpp`

```cpp
namespace forgeml {

double accuracy(std::vector<int> const &y_true, std::vector<int> const &y_pred);

// All of these are undefined when the denominator is 0 -> fp::Result.
fp::Result<double> precision(BinaryCounts const &c);   // tp / (tp + fp)
fp::Result<double> recall(BinaryCounts const &c);      // tp / (tp + fn)
fp::Result<double> specificity(BinaryCounts const &c); // tn / (tn + fp)
fp::Result<double> f1(BinaryCounts const &c);          // 2PR / (P + R)

// Convenience wrappers that build counts internally (binary, positive = 1)
fp::Result<double> precision(std::vector<int> const &y_true,
                             std::vector<int> const &y_pred);
fp::Result<double> recall(std::vector<int> const &y_true,
                          std::vector<int> const &y_pred);
fp::Result<double> f1(std::vector<int> const &y_true,
                      std::vector<int> const &y_pred);

// Macro averages over classes (multiclass)
struct MacroScores { double precision, recall, f1; };
fp::Result<MacroScores> macro_scores(Matrix<std::size_t> const &cm);

// Threshold helper: probabilities -> labels
std::vector<int> threshold(Vector<double> const &probs, double t = 0.5);

// ROC-AUC (binary, rank-based / Mann-Whitney U). O(n log n).
fp::Result<double> roc_auc(std::vector<int> const &y_true,
                           Vector<double> const &scores);
}
```

Formulas:

```text
accuracy    = (tp + tn) / total
precision   = tp / (tp + fp)
recall      = tp / (tp + fn)
specificity = tn / (tn + fp)
F1          = 2 * precision * recall / (precision + recall)
ROC-AUC     = (sum of ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg)
```

Rules:

- Denominator zero -> `fp::fail` with a metric-specific message
  (`"precision: no predicted positives"`).
- `threshold` uses `p >= t` for the positive class.
- `roc_auc` handles ties by average ranks; `fail` if only one class is present.
- Multiclass one-vs-rest is available via `confusion_matrix` +
  `one_vs_rest_counts`; do not silently pick a "positive" class.

Tests:

- accuracy on a hand-computed list
- precision/recall/F1 on a known confusion matrix
- denominator-zero cases return `fail`
- ROC-AUC is 1 for perfectly ranked scores, 0.5 for random-ish fixed case,
  and handles a tie by averaging
- macro scores on a 3-class table

---

## How metrics are consumed

- `eval/scoring.hpp` wraps a metric as `ScoreFn = std::function<double(
  std::vector<int> const&, std::vector<int> const&)>` (classification) or the
  regression analogue, so `cross_validation` and `grid_search` stay generic.
- Models never call metrics during training; training loss lives in
  `math/losses.hpp`. This separation is deliberate: a model can be optimized
  with MSE and evaluated with R².

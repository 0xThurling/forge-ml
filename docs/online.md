# online — learning from a stream

Data that arrives in order and never ends: running statistics, models that
update per batch (`partial_fit`), drift detection, and the prequential
evaluation protocol that scores a stream honestly.

Depends on: `core`, `prob`, `model`, `eval`, ForgeFP.

Files: `welford.hpp`, `incremental.hpp`, `drift.hpp`, `evaluate.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `welford.hpp` | `fp::map2d`, `fp::inplace`, `fp::Validation` |
| `incremental.hpp` | `fp::linalg` (`dot`, `matmul`), `fp::inplace`, `fp::views::chunk` |
| `drift.hpp` | `fp::views::window`, `fp::linalg` (`mean`, `variance`), `fp::sort_by_cached` |
| `evaluate.hpp` | `fp::views::chunk`, `fp::map` |

---

## `online/welford.hpp`

**Job:** mean and variance in one pass, numerically stable, mergeable across
workers.

```cpp
namespace forgeml {

struct RunningStats {
  std::size_t count = 0;
  double mean = 0.0;
  double m2 = 0.0;              // sum of squared deviations
  double variance() const;      // m2 / count
};

void update(RunningStats &s, double x);
RunningStats merge(RunningStats const &a, RunningStats const &b);

struct ColumnStats { std::vector<RunningStats> columns; };
void update(ColumnStats &s, Vector<double> const &x);
Vector<double> means(ColumnStats const &s);
Vector<double> variances(ColumnStats const &s);
}
```

Rules: the Welford recurrence is used exactly (no `Σx²` shortcut, which loses
precision); `merge` is the parallel-combination formula; a column that has
seen no data has variance 0 and `count` 0 (callers must check).

Tests: matches a two-pass mean/variance on random data; merging two halves
equals one pass; a constant stream has variance 0; a long stream with a large
offset stays accurate (the classic Welford case).

## `online/incremental.hpp`

**Job:** a `partial_fit` protocol and the models that implement it.

```cpp
struct SgdClassifier {
  Vector<double> weights;       // + bias at index 0
  double lr = 0.01;
  enum class Schedule { Constant, InverseTime } schedule = Schedule::InverseTime;
  std::size_t steps = 0;
};
fp::Result<void> partial_fit(SgdClassifier &model, Matrix<double> const &X,
                             Vector<double> const &y, double lr_scale = 1.0);
Vector<double> predict_proba(SgdClassifier const &model, Matrix<double> const &X);

struct OnlineScaler { ColumnStats stats; };
void partial_fit(OnlineScaler &scaler, Matrix<double> const &X);
Matrix<double> transform(OnlineScaler const &scaler, Matrix<double> const &X);
```

Rules: the learning rate follows the schedule (`lr / (1 + lr * λ * steps)` for
inverse-time); `partial_fit` never allocates after construction (weights are
sized once); the online scaler is fit incrementally and **must not** be refit
on the test stream; a model that has not seen a class yet predicts it with
probability 0, not NaN.

Tests: `partial_fit` over batches converges to the batch-trained solution on a
linear problem; the inverse-time schedule converges where a constant one
oscillates; the online scaler matches a batch scaler; a zero-row batch is a
no-op.

## `online/drift.hpp`

**Job:** detect that the stream has changed — before the model silently rots.

```cpp
struct Psi { double value = 0.0; bool drifted = false; };
Psi population_stability_index(Vector<double> const &reference, Vector<double> const &current,
                               std::size_t bins = 10, double threshold = 0.2);

struct PageHinkley { double mean = 0.0, cumulative = 0.0, min_cumulative = 0.0; double delta = 0.005, lambda = 50.0; };
bool update(PageHinkley &detector, double error);

struct AdwinLite {
  std::vector<double> window;
  double delta = 0.002;         // confidence
  std::size_t min_window = 30;
};
bool update(AdwinLite &detector, double value);   // true when a split is significant
```

Rules: PSI bins the reference into deciles (quantiles) and compares the
current proportions with the standard `(p-q)·ln(p/q)` formula; Page–Hinkley
tracks the cumulative deviation of a *streaming error* (not the raw feature);
ADWIN-lite tests the best split of its window with a Hoeffding-style bound and
shrinks the window when it fires; all three are deterministic.

Tests: PSI flags a planted mean shift and stays low on a stationary stream;
Page–Hinkley fires within N samples of a planted change (fixed seed, N
documented); ADWIN-lite detects a variance change; a stationary stream does
not fire over a long run (false-positive check).

## `online/evaluate.hpp`

**Job:** the prequential protocol — test, then train, one sample at a time.

```cpp
struct Prequential {
  Vector<double> error;         // after each step
  Vector<double> rolling;       // windowed mean error
};
template <class Model>
Prequential prequential(Model &model, Matrix<double> const &X, Vector<double> const &y,
                        std::size_t window = 100);
```

Rules: each sample is predicted **before** it is learned from (no leakage); the
rolling error uses the same window as the drift detectors so the two line up;
the final error is not the metric — the trace is.

Tests: on a stationary stream the rolling error flattens; on a planted shift it
rises after the change point; prequential error tracks a batch-trained model's
error within tolerance on a stationary stream; the first prediction is the
initial model's, not the first label's.

## Gate

Welford matches two-pass statistics and merges exactly; `partial_fit`
converges to the batch solution; PSI/Page–Hinkley/ADWIN-lite detect planted
changes within their documented budgets and stay quiet on stationary streams;
prequential evaluation never leaks a label.

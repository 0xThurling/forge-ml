# interpret — trusting a model before shipping it

Scores say *how much* a model is right; interpretation says *where* and *why*.
Importance, partial dependence, calibration and the curves that pick an
operating point. Every function here is model-agnostic where it can be, so it
works for a linear model, a tree or a forest.

Depends on: `core`, `metrics`, `eval` (splits), `model`, `prob`, ForgeFP.

Files: `importance.hpp`, `pdp.hpp`, `calibration.hpp`, `curves.hpp`,
`learning.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `importance.hpp` | `fp::Rng::shuffle`, `fp::sort_by_cached`, `fp::linalg::mean` |
| `pdp.hpp` | `fp::grid`, `fp::views::chunk`, `fp::linalg::mean` |
| `calibration.hpp` | `fp::sort_by`, `fp::views::chunk`, `fp::linalg` |
| `curves.hpp` | `fp::sort_by_cached`, `fp::views::window` |
| `learning.hpp` | `fp::views::chunk`, `fp::Rng`, `fp::map` |

---

## `interpret/importance.hpp`

**Job:** how much does each feature matter? Permutation importance for any
model, impurity importance for trees.

```cpp
namespace forgeml {

struct Importance {
  Vector<double> mean;      // one per feature
  Vector<double> stddev;    // across repeats
};

template <class Model>
Importance permutation_importance(Model const &model, Matrix<double> const &X,
                                  Vector<double> const &y, double (*score)(Vector<double> const &, Vector<double> const &),
                                  std::size_t n_repeats, fp::Rng &rng);
Vector<double> tree_importance(Cart const &tree);
}
```

Rules: a feature is permuted by shuffling its column (`fp::Rng::shuffle`); the
score drop is averaged over repeats with its spread reported; permutation
happens on a **validation** split when one exists; a feature whose importance
is within one stddev of zero is reported as uninformative.

Tests: the planted feature ranks first and noise features are near zero; more
repeats shrink the reported stddev; permuting a constant column changes
nothing.

## `interpret/pdp.hpp`

**Job:** partial dependence and ICE curves — the model's average response as
one feature moves.

```cpp
template <class Model>
Vector<double> partial_dependence(Model const &model, Matrix<double> const &X,
                                  std::size_t feature, std::size_t grid_points = 20);
template <class Model>
Matrix<double> ice_curves(Model const &model, Matrix<double> const &X, std::size_t feature,
                          std::size_t grid_points = 20, std::size_t sample = 100);
```

Rules: the grid spans the observed feature range (quantiles when
`grid_points` is small); the other columns stay at their observed values;
ICE curves are subsampled for readability but the PDP uses every row.

Tests: a linear model's PDP is a straight line with the right slope; a
planted interaction shows up as diverging ICE curves; PDP of an irrelevant
feature is flat.

## `interpret/calibration.hpp`

**Job:** are the probabilities honest, and how do we make them so?

```cpp
struct Reliability { Vector<double> confidence, accuracy; Vector<std::size_t> count; };
fp::Result<Reliability> reliability_curve(Vector<double> const &probabilities,
                                          Vector<double> const &y, std::size_t bins = 10);
double expected_calibration_error(Reliability const &r);

struct Platt { double a = 1.0, b = 0.0; };
fp::Result<Platt> platt_scaling(Vector<double> const &scores, Vector<double> const &y);
Vector<double> isotonic_regression(Vector<double> const &scores, Vector<double> const &y);  // PAVA
```

Rules: Platt fits `sigmoid(a*score + b)` by Newton iterations on the log-loss;
isotonic uses pool-adjacent-violators and is monotone by construction; both
are fit on a validation split, never on the test set.

Tests: a deliberately miscalibrated score halves its ECE after Platt; isotonic
output is non-decreasing; a perfectly calibrated score has ECE near zero.

## `interpret/curves.hpp`

**Job:** ROC and precision–recall curves with thresholds, and how to choose
one.

```cpp
struct Roc { Vector<double> threshold, tpr, fpr; };
Roc roc_curve(Vector<double> const &scores, Vector<double> const &y);
double auc(Roc const &roc);
struct Pr { Vector<double> threshold, precision, recall; };
Pr pr_curve(Vector<double> const &scores, Vector<double> const &y);
double average_precision(Pr const &pr);
std::size_t best_threshold(Roc const &roc);   // Youden's J
```

Rules: thresholds are the sorted unique scores; ties are handled by grouping;
AUC uses the trapezoid rule; `best_threshold` returns the index into the
curve's arrays.

Tests: AUC matches a hand-computed 4-point example; a perfect ranking gives
AUC 1 and AP 1; a random ranking gives AUC ≈ 0.5; `best_threshold` picks the
known optimum on a hand example.

## `interpret/learning.hpp`

**Job:** learning and validation curves — is more data or more capacity the
answer?

```cpp
template <class Fit>
struct Curve { Vector<double> x, train_score, val_score; };
template <class Fit>
fp::Result<Curve<double>> learning_curve(Fit &&fit, Matrix<double> const &X, Vector<double> const &y,
                                         Vector<double> const &fractions, std::size_t folds,
                                         fp::Rng &rng);
template <class Fit>
fp::Result<Curve<double>> validation_curve(Fit &&fit, Matrix<double> const &X, Vector<double> const &y,
                                           Vector<double> const &parameter, std::size_t folds,
                                           fp::Rng &rng);
```

Rules: both use `eval`'s k-fold splitter, so the val score is honest; the
train score at small `n` is expected to be optimistic — the *gap* is the
signal.

Tests: on a small noisy set, train score > val score and the gap shrinks as
`fractions` grow; the validation curve's peak matches a planted optimum.

## Gate

Permutation importance ranks a planted feature first; Platt scaling halves a
miscalibrated model's ECE; the ROC curve matches a hand-computed example; the
learning curve shows the expected train/val gap and its closure.

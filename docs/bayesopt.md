# bayesopt — searching a hyperparameter space on purpose

Grid and random search are baselines, not strategies. This layer adds a
Gaussian-process surrogate and acquisition functions so an expensive objective
(a cross-validated score) is evaluated where it is most informative — and
proves the win against the baselines it replaces.

Depends on: `core`, `prob`, `eval` (CV objective), `reduce` (PCA for the
candidate pool when `d` is large), ForgeFP.

Files: `gp.hpp`, `acquisition.hpp`, `search.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `gp.hpp` | `fp::linalg` (`matmul`, `solve`), `fp::map2d`, `fp::Validation` |
| `acquisition.hpp` | `fp::numerics` (`logsumexp`), `fp::linalg` |
| `search.hpp` | `fp::Rng`, `fp::sort_by_cached`, `fp::views::chunk` |

---

## `bayesopt/gp.hpp`

**Job:** Gaussian-process regression — the surrogate — with a Cholesky solve.

```cpp
namespace forgeml {

enum class Kernel { Rbf, Matern32 };
struct GaussianProcess {
  Matrix<double> X;              // observed points (n, d)
  Vector<double> y;              // observed values
  Vector<double> length_scale;   // per dimension
  double signal = 1.0, noise = 1e-6;
  Kernel kernel = Kernel::Rbf;
  Matrix<double> chol;           // cached Cholesky factor of K + noise I
};

fp::Result<GaussianProcess> fit_gp(Matrix<double> const &X, Vector<double> const &y,
                                   Vector<double> const &length_scale, double signal,
                                   double noise, Kernel kernel = Kernel::Rbf);
struct GpPrediction { Vector<double> mean, variance; };
GpPrediction predict(GaussianProcess const &gp, Matrix<double> const &X);
double log_marginal_likelihood(GaussianProcess const &gp);
fp::Result<GaussianProcess> tune_length_scale(GaussianProcess const &gp, std::vector<double> const &grid);
}
```

Rules: the kernel matrix is built with `exp(-0.5 * ||x-x'||²/l²)` (RBF) or the
Matérn-3/2 form; the Cholesky factor is computed once and reused for predict
and the marginal likelihood; a non-positive-definite matrix is an error (ridge
escalation is documented, not silent); predictive variance is never negative
(clamped to 0).

Tests: the posterior mean interpolates the observations; a hand-computed
3-point GP matches; predictive variance grows away from the data; the marginal
likelihood prefers the true length scale on a planted function.

## `bayesopt/acquisition.hpp`

**Job:** where to sample next — expected improvement, probability of
improvement, upper confidence bound.

```cpp
enum class Acquisition { Ei, Pi, Ucb };
Vector<double> expected_improvement(GpPrediction const &pred, double best, double xi = 0.01);
Vector<double> probability_improvement(GpPrediction const &pred, double best, double xi = 0.01);
Vector<double> upper_confidence_bound(GpPrediction const &pred, double beta = 2.0);
std::size_t argmax(Vector<double> const &values);
```

Rules: EI uses the closed form with the normal CDF/PDF; `best` is the best
observed value (maximization convention); ties break by index; the functions
return a value per candidate, never a scalar decision.

Tests: EI of a candidate far below the best is ~0; EI at the best is positive
(exploration bonus); UCB with β=0 equals the posterior mean; a hand-computed
EI example matches.

## `bayesopt/search.hpp`

**Job:** the loop that ties surrogate, acquisition and objective together —
and the baselines to beat.

```cpp
struct Bounds { Vector<double> low, high; };
struct Trial { Vector<double> point; double value; };
struct Result { std::vector<Trial> trials; Trial best; };

fp::Result<Result> bayes_opt(std::function<double(Vector<double> const &)> const &objective,
                             Bounds const &bounds, std::size_t initial,
                             std::size_t budget, Acquisition acquisition,
                             fp::Rng &rng);

fp::Result<Result> random_search(std::function<double(Vector<double> const &)> const &objective,
                                 Bounds const &bounds, std::size_t budget, fp::Rng &rng);
fp::Result<Result> grid_search(std::function<double(Vector<double> const &)> const &objective,
                               std::vector<Vector<double>> const &grid);
```

Rules: the initial design is space-filling (Latin-hypercube-ish, seeded);
candidates come from a seeded pool (or a grid when `d` is small); the loop is
greedy on the acquisition but always keeps the best *observed* trial; the
objective is never called twice on the same point; a CV objective plugs in via
`eval` (`cross_validate`).

Tests: on a planted 2-D function, `bayes_opt` finds a better optimum than
`random_search` with the same budget; on a 1-D quadratic it matches the
analytic optimum within tolerance; grid search visits exactly the grid; the
objective-call count equals the budget (no hidden evaluations).

## Gate

The GP matches a hand-computed posterior; the acquisition functions match
hand-computed values; Bayesian optimization beats random search at an equal
budget on planted functions; a cross-validated hyperparameter search improves
on the default configuration.

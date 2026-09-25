# probability — distributions, information and statistics

The layer every model quietly relies on: distributions with a uniform
interface, information-theoretic measures, conjugate Bayesian updates,
sampling, and the statistical tests that tell you whether a difference is
real. ForgeFP supplies the randomness and the numerics; this layer supplies the
*meaning*.

Depends on: `core`, `math/stats`, ForgeFP.

Files: `distributions.hpp`, `info.hpp`, `bayes.hpp`, `sampling.hpp`,
`stats.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `distributions.hpp` | `fp::Rng`, `fp::numerics` (`logsumexp`), `fp::linalg` (`dot`, `matvec`), `fp::Validation` |
| `info.hpp` | `fp::numerics` (`logsumexp`), `fp::map`, `fp::fold_left` |
| `bayes.hpp` | `fp::Rng`, `fp::numerics`, `fp::Validation` |
| `sampling.hpp` | `fp::Rng`, `fp::numerics`, `fp::linalg` |
| `stats.hpp` | `fp::sort`, `fp::linalg` (`mean`, `variance`), `fp::Rng::sample_indices` |

---

## `prob/distributions.hpp`

**Job:** one interface for the distributions the stack needs, each with a
closed-form `logpdf` (the numerically safe entry point) and a `sample`.

```cpp
namespace forgeml {

struct Gaussian   { double mu = 0.0, sigma = 1.0; };
struct Bernoulli  { double p = 0.5; };
struct Categorical{ Vector<double> p; };              // normalised on construction
struct Poisson    { double lambda = 1.0; };
struct Exponential{ double rate = 1.0; };
struct Beta       { double a = 1.0, b = 1.0; };
struct Gamma      { double shape = 1.0, rate = 1.0; };
struct Dirichlet  { Vector<double> alpha; };

template <class D> double logpdf(D const &d, double x);
template <class D> double pdf(D const &d, double x);       // exp(logpdf)
template <class D> double mean(D const &d);
template <class D> double variance(D const &d);
template <class D> double sample(D const &d, fp::Rng &rng);
fp::Result<double> cdf(Gaussian const &d, double x);       // erf-based
}
```

Rules:

- Every distribution validates its parameters (`fp::Result` from the
  constructors' helpers; invalid parameters never reach `logpdf`).
- `logpdf` is the primitive; `pdf` is `std::exp` of it — for tiny
  probabilities only `logpdf` is meaningful.
- Sampling uses only `fp::Rng` (`uniform`, `normal`, `categorical`), so a seed
  reproduces a stream exactly.

Tests: moments match hand values; samples' empirical mean/variance converge;
`logpdf` sums to a known value for a hand example; invalid parameters error.

## `prob/info.hpp`

**Job:** entropy, divergences and mutual information over discrete and
binned-continuous data.

```cpp
double entropy(Vector<double> const &p);                     // nats
double cross_entropy(Vector<double> const &p, Vector<double> const &q);
double kl(Vector<double> const &p, Vector<double> const &q);
double js(Vector<double> const &p, Vector<double> const &q);
double mutual_information(Matrix<double> const &X, Vector<double> const &y, std::size_t bins = 0);
double perplexity(Vector<double> const &p);
```

Rules: `p` and `q` are normalised; zero-probability handling follows the
mathematical limit (`0 log 0 = 0`; `p > 0, q = 0` → `+inf`, reported as
`std::numeric_limits<double>::infinity()`). Continuous columns are binned with
`features/binning.hpp`.

Tests: `kl(p, p) == 0`; `kl >= 0` over random pairs; `js` is symmetric and
bounded by `log 2`; MI of a copied label equals its entropy.

## `prob/bayes.hpp`

**Job:** conjugate updates and posterior predictives — Bayesian inference
without a sampler.

```cpp
struct BetaBernoulli { double a, b; };
BetaBernoulli update(BetaBernoulli prior, std::size_t successes, std::size_t trials);
double posterior_mean(BetaBernoulli const &p);

struct DirichletCategorical { Vector<double> alpha; };
DirichletCategorical update(DirichletCategorical prior, Vector<std::size_t> const &counts);

struct NormalNormal { double mu0, sigma0, sigma; };   // known variance
NormalNormal update(NormalNormal prior, Vector<double> const &data);
```

Tests: a Beta prior updated with `k/n` heads has the hand-computed
`(a+k, b+n-k)`; the Dirichlet posterior normalises; Normal–Normal shrinks
toward the prior when `n` is small and toward the data when `n` is large.

## `prob/sampling.hpp`

**Job:** the samplers a from-scratch stack needs, all `fp::Rng`-driven.

```cpp
template <class F> double inverse_cdf(F const &cdf, double u);            // bisection
template <class LogPdf> fp::Result<std::vector<double>> rejection(LogPdf const &logp, double xmin, double xmax, std::size_t n, fp::Rng &rng);
struct Importance { Vector<double> samples, weights; double ess() const; };
template <class LogPdf, class LogQ> Importance importance(LogPdf const &logp, LogQ const &logq, std::size_t n, fp::Rng &rng);
template <class LogPdf> std::vector<double> metropolis_hastings(LogPdf const &logp, double x0, double step, std::size_t n, fp::Rng &rng);
template <class LogCond> std::vector<double> gibbs(LogCond const &log_conditional, Vector<double> x0, std::size_t n, fp::Rng &rng);
```

Rules: burn-in and thinning are explicit parameters; the acceptance rate is
returned for diagnostics; importance sampling exposes effective sample size
(`ess`) so users can see when the proposal is bad.

Tests: rejection sampling of a Beta matches its moments; MH on a Gaussian
recovers mean/variance; ESS decreases when the proposal is far from the target.

## `prob/stats.hpp`

**Job:** estimation and the tests that keep conclusions honest.

```cpp
double mle_mean(Vector<double> const &x);
fp::Result<std::pair<double, double>> bootstrap_ci(Vector<double> const &x,
                                                   std::size_t n_resamples,
                                                   double confidence, fp::Rng &rng);
struct TestResult { double statistic, p_value; std::size_t df; };
fp::Result<TestResult> t_test(Vector<double> const &a, Vector<double> const &b);       // Welch
fp::Result<TestResult> chi_square(Vector<double> const &observed, Vector<double> const &expected);
fp::Result<TestResult> ks_test(Vector<double> const &x, Vector<double> const &y);
fp::Result<TestResult> permutation_test(Vector<double> const &a, Vector<double> const &b,
                                        std::size_t n_permutations, fp::Rng &rng);
```

Tests: Welch's t on two shifted samples matches a hand-computed value; a
permutation test on identical samples gives a p-value near 1; the KS statistic
of a sample against itself is 0; the bootstrap CI covers the true mean in a
fixed-seed simulation.

## Gate

Distribution moments match hand values; `KL ≥ 0` with equality only for equal
distributions; a conjugate update matches its closed form; MCMC recovers a
known posterior mean within tolerance; the tests reject a planted difference
and accept a null one.

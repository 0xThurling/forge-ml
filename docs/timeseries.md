# timeseries — data with an order

Rows are not exchangeable: yesterday predicts today. Windowing, stationarity
tools, ARIMA, exponential smoothing and backtesting that never leaks the
future.

Depends on: `core`, `math/stats`, `prob` (ACF uses covariance), ForgeFP.

Files: `window.hpp`, `stationary.hpp`, `arima.hpp`, `smoothing.hpp`,
`backtest.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `window.hpp` | `fp::views::windows`, `fp::linalg` (`mean`, `variance`) |
| `stationary.hpp` | `fp::linalg` (`dot`, `solve`), `fp::views::windows` |
| `arima.hpp` | `fp::linalg` (`solve`, `dot`), `fp::views::chunk`, `fp::inplace` |
| `smoothing.hpp` | `fp::sort_by_cached`, `fp::inplace`, `fp::map` |
| `backtest.hpp` | `fp::views::chunk`, `fp::linalg`, `fp::Rng` |

---

## `timeseries/window.hpp`

**Job:** turn a series into supervised rows — lags, rolling and expanding
statistics, differencing.

```cpp
namespace forgeml {

Matrix<double> lag_matrix(Vector<double> const &series, std::size_t lags, bool bias = true);
Vector<double> difference(Vector<double> const &series, std::size_t order = 1);
Vector<double> rolling_mean(Vector<double> const &series, std::size_t window);
Vector<double> rolling_std(Vector<double> const &series, std::size_t window);
Vector<double> expanding_mean(Vector<double> const &series);
}
```

Rules: `lag_matrix` row `t` holds `[1, x(t-1), …, x(t-lags)]` aligned with
target `x(t)`, so the first `lags` observations have no row; rolling outputs
start at index `window-1`; differencing of order 2 applies order 1 twice.

Tests: a hand-built series produces the expected rows; lag count is `n-lags`;
differencing a linear series leaves a constant; rolling mean of a constant is
that constant.

## `timeseries/stationary.hpp`

**Job:** is the series stationary, and what do the correlations say?

```cpp
Vector<double> acf(Vector<double> const &series, std::size_t max_lag);
Vector<double> pacf(Vector<double> const &series, std::size_t max_lag);   // Durbin–Levinson
struct Adf { double statistic; std::size_t lags; };
fp::Result<Adf> adf_statistic(Vector<double> const &series, std::size_t lags = 0);
bool is_stationary(Adf const &adf, double alpha = 0.05);
```

Rules: ACF is normalized by the sample variance; PACF uses the Levinson
recursion; `adf_statistic` regresses `Δx` on `x(t-1)` and lagged differences
and returns the t-statistic of the first coefficient; the critical values are
a small table (1%, 5%, 10%) with the caveat documented; `is_stationary` uses
the 5% value.

Tests: white noise has ACF ≈ 0 beyond lag 0 and is stationary; a random walk
is not; ACF of an AR(1) decays at the right rate; PACF of AR(p) cuts off after
`p`.

## `timeseries/arima.hpp`

**Job:** AR, MA and ARIMA via conditional least squares, with forecasts and
prediction intervals.

```cpp
struct Arima {
  std::size_t p = 0, d = 0, q = 0;
  Vector<double> ar, ma;         // coefficients
  double mean = 0.0;
  double sigma2 = 0.0;           // residual variance
};

fp::Result<Arima> fit_arima(Vector<double> const &series, std::size_t p, std::size_t d, std::size_t q);
struct Forecast { Vector<double> mean; Vector<double> lower, upper; };
fp::Result<Forecast> forecast(Arima const &model, Vector<double> const &history,
                              std::size_t horizon, double confidence = 0.95);
```

Rules: AR coefficients come from OLS on `lag_matrix` (via `fp::linalg::solve`);
MA terms are estimated by iterating on residuals (documented iteration count);
intervals widen with the horizon by `sigma2` and the AR/MA recursion; `d` is
applied with `difference` and undone on the forecast.

Tests: AR(2) coefficients are recovered within tolerance on a long planted
series; ARIMA beats a naive last-value forecast on planted AR(2) data; the
forecast intervals widen with the horizon; a white-noise series fits with
`ar ≈ 0`.

## `timeseries/smoothing.hpp`

**Job:** exponential smoothing — the cheap, robust baseline.

```cpp
struct Ses { double alpha; double level; };
fp::Result<Ses> fit_ses(Vector<double> const &series);        // alpha by grid + SSE
struct Holt { double alpha, beta, level, trend; };
fp::Result<Holt> fit_holt(Vector<double> const &series);
struct HoltWinters { double alpha, beta, gamma, level, trend; Vector<double> season; std::size_t period; };
fp::Result<HoltWinters> fit_holt_winters(Vector<double> const &series, std::size_t period,
                                         bool multiplicative = false);
Vector<double> forecast(Ses const &m, std::size_t h);
Vector<double> forecast(Holt const &m, std::size_t h);
Vector<double> forecast(HoltWinters const &m, std::size_t h);
```

Rules: parameters are optimized by a coarse grid followed by a local refine
(deterministic, no RNG); smoothing constants are validated to `[0, 1]`;
seasonal indices are normalized to sum to `period` (additive) or mean 1
(multiplicative).

Tests: SES tracks a constant; Holt tracks a linear trend; Holt–Winters tracks
a planted seasonal series with lower error than Holt; the fitted parameters
are stable across a shuffled-but-identical input (deterministic).

## `timeseries/backtest.hpp`

**Job:** honest evaluation on ordered data.

```cpp
struct Split { std::size_t train_end, test_end; };
std::vector<Split> rolling_origin(std::size_t n, std::size_t initial,
                                  std::size_t horizon, std::size_t step = 1);
double mae(Vector<double> const &y, Vector<double> const &pred);
double rmse(Vector<double> const &y, Vector<double> const &pred);
double mape(Vector<double> const &y, Vector<double> const &pred);
double smape(Vector<double> const &y, Vector<double> const &pred);
double mase(Vector<double> const &y, Vector<double> const &pred, std::size_t season = 1);
```

Rules: every split satisfies `train_end <= test_start` (a leak check is a
test); `mase` scales by the in-sample naive error, so `< 1` means "better than
naive"; `mape` is undefined for zero targets and returns an error rather than
infinity.

Tests: rolling-origin splits never overlap train and test; MASE of the naive
forecast is 1; MASE of a good model is below 1; metrics match hand-computed
values.

## Gate

ARIMA beats the naive forecast on a planted AR(2) series; Holt–Winters tracks
a seasonal series; ACF/PACF match the theory; backtesting never leaks the
future; MASE is the comparison metric of record.

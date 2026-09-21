# utils — assertions, logging, checkpoints, gradient checks

The boring but load-bearing helpers. Everything generic is ForgeFP; what is
left here is the *schema* and *policy* that is specific to this stack.

Depends on: `core`, ForgeFP.

Files: `assertions.hpp`, `logging.hpp`, `serialization.hpp`,
`gradient_check.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `assertions.hpp` | `fp::approx_equal`, `fp::is_finite`, `fp::nan_to_num` |
| `logging.hpp` | `fp::Stopwatch`, `fp::now_seconds`, `fp::str::to_string`, `fp::str::join` |
| `serialization.hpp` | `fp::to_text`/`from_text`, `fp::to_bytes`/`from_bytes`, `fp::write_bytes`, `fp::read_bytes`, `fp::ensure_directory`, `fp::str::split`, `fp::Result` |
| `gradient_check.hpp` | `fp::central_difference`, `fp::ad::derivative`, `fp::approx_equal` |

---

## `utils/assertions.hpp`

**Job:** contract checking (bugs) and tolerance helpers. Runtime input
validation belongs in `fp::Result`, not here.

```cpp
namespace forgeml {

class MlError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void assertion_failed(char const *expr, char const *file,
                                          int line, char const *msg);

// Tolerance helpers, built on fp.
inline bool approximately_equal(double a, double b, double eps = 1e-9) {
  return fp::approx_equal(a, b, eps);
}
inline bool all_close(Vector<double> const &a, Vector<double> const &b,
                      double eps = 1e-9);          // elementwise fp::approx_equal
inline bool all_close(Matrix<double> const &a, Matrix<double> const &b,
                      double eps = 1e-9);
inline bool is_finite(Vector<double> const &v);    // fp::is_finite over the range
inline bool is_finite(Matrix<double> const &m);
inline double sanitize(double x) { return fp::nan_to_num(x, 0.0); }
}

#define ML_ASSERT(cond, msg) \
  do { if (!(cond)) ::forgeml::assertion_failed(#cond, __FILE__, __LINE__, (msg)); } while (false)
```

Rules:

- `ML_ASSERT` is for programmer errors: index bounds, "unfitted model", shape
  invariants, `batch_size >= 1`.
- User-triggerable failures use `fp::Result`; do not turn them into asserts.
- Tolerance helpers delegate to `fp::approx_equal` (relative at scale, absolute
  near zero); do not hand-roll `fabs(a - b) < eps`.
- `is_finite` is the training-stability guard; `sanitize` is the repair.

Tests (`test/utils_test.cpp`):

- `all_close` true/false cases including NaN
- `ML_ASSERT` throws `MlError` with the expression text in the message
- `sanitize(nan) == 0.0`, `sanitize(inf) == 0.0`

---

## `utils/logging.hpp`

**Job:** a tiny leveled logger so library code never writes to streams
directly, and a training callback type shared by all trainers.

```cpp
namespace forgeml {

enum class LogLevel { Debug, Info, Warn, Error };

struct LogRecord {
  LogLevel level;
  std::string message;
};

class Logger {
public:
  using Sink = std::function<void(LogRecord const &)>;

  explicit Logger(LogLevel min_level = LogLevel::Info,
                  Sink sink = default_sink());

  void log(LogLevel level, std::string message);
  void debug(std::string message);
  void info(std::string message);
  void warn(std::string message);
  void error(std::string message);

  void set_level(LogLevel level);
  LogLevel level() const;

  static Sink default_sink();    // std::clog, "[info] message"
  static Sink null_sink();       // discards (tests)

private:
  LogLevel min_level_;
  Sink sink_;
};

// One epoch of training, shared by classical models and the NN/LLM trainers.
struct EpochStats {
  std::size_t epoch = 0;
  double train_loss = 0.0;
  double val_loss = 0.0;
  double learning_rate = 0.0;
  double seconds = 0.0;          // measured with fp::Stopwatch
};

using EpochCallback = std::function<void(EpochStats const &)>;

// Convenience: format a stats line with fp::str.
inline std::string format_stats(EpochStats const &s);
}
```

Rules:

- No global logger instance. Objects that log take `Logger &` or an
  `EpochCallback`.
- `log` filters by `min_level_` before invoking the sink.
- `seconds` is produced by `fp::Stopwatch::lap()` / `fp::now_seconds()` at the
  call site; the logger never touches the clock itself.
- `format_stats` uses `fp::str::to_string(value, 4)` and `fp::str::join`, so the
  output is locale-free and consistent with the rest of the library.

Tests (`test/logging_test.cpp`):

- level filtering (debug suppressed at Info)
- a custom sink receives records in order
- `null_sink` produces no output
- `format_stats` contains the epoch and a four-decimal loss

---

## `utils/serialization.hpp`

**Job:** the checkpoint **schema** — which tensor is which, plus a version —
on top of ForgeFP's generic encoding. fp owns `to_text`/`from_text`/
`to_bytes`/`from_bytes`; this file owns the named entries and the file layout.

```cpp
namespace forgeml {

inline constexpr int kCheckpointVersion = 1;

struct ModelFile {
  std::string model_name;
  std::vector<std::pair<std::string, Matrix<double>>> matrices;
  std::vector<std::pair<std::string, Vector<double>>> vectors;
  std::vector<std::pair<std::string, double>> scalars;
};

// Text form: a version line, then `name kind dims data` records.
std::string to_text(ModelFile const &file);
fp::Result<ModelFile> from_text(std::string_view text);

// Binary form: fp::to_bytes per entry, fp's size checks on load.
std::vector<std::byte> to_bytes(ModelFile const &file);
fp::Result<ModelFile> from_bytes(std::span<std::byte const> data);

// Files: directory creation + byte write/read.
fp::Result<void> save(std::string const &path, ModelFile const &file);
fp::Result<ModelFile> load(std::string const &path);

// Training checkpoint: parameters + optimizer step + loss.
fp::Result<void> save_checkpoint(std::string const &dir, std::size_t step,
                                 double loss, ModelFile const &params);
fp::Result<ModelFile> load_checkpoint(std::string const &dir, std::size_t &step,
                                      double &loss);
}
```

Rules:

- Encoding is delegated: each entry uses `fp::to_text(values, 17)` /
  `fp::from_text<T>` (text) or `fp::to_bytes` / `fp::from_bytes<T>` (binary).
  Doubles round-trip exactly at precision 17.
- Text records are split with `fp::str::split`; a bad token returns
  `fp::fail("line 5: expected a number, got 'x'")` — the schema reports the
  line, fp reports the token.
- Dimensions are written before the data and validated on load.
- Unknown record kinds are skipped with a warning (forward compatibility); a
  wrong version number is an error.
- `save` calls `fp::ensure_directory` on the parent, then `fp::write_bytes`;
  `load` calls `fp::read_bytes`.
- The schema knows nothing about models; callers name their entries.

Tests (`test/serialization_test.cpp`):

- text and binary round-trip within `1e-12`
- dimension mismatch on load returns `fail` with the line number
- unknown version returns `fail`
- save/load a checkpoint preserves step, loss, and every entry
- writing into a nested directory creates it

---

## `utils/gradient_check.hpp`

**Job:** the safety net for every layer and loss with a backward pass. The
finite differences come from `fp::central_difference`; the exact comparison is
available through `fp::ad::derivative`. This file owns the *harness* (which
parameters, tolerances, reporting).

```cpp
namespace forgeml {

struct GradientCheckResult {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::size_t worst_index = 0;
  bool passed = true;
};

// f: flat parameters -> scalar loss. grad_f: parameters -> analytic gradient.
GradientCheckResult check_gradient(
    std::function<double(Vector<double> const &)> const &f,
    std::function<Vector<double> const &(Vector<double> const &)> const &grad_f,
    Vector<double> const &params, double eps = 1e-6,
    double tolerance = 1e-6);

// Exact alternative for scalar functions: fp::ad::derivative, no eps.
template <class F, class T = double> T exact_derivative(F f, T x) {
  return fp::derivative(f, x);
}

// Convenience for networks: forward+loss, backward, compare every parameter.
template <class ForwardLoss, class Backward>
GradientCheckResult check_network(ForwardLoss forward_loss, Backward backward,
                                  double eps = 1e-6, double tolerance = 1e-6);
}
```

Algorithm:

```text
for each parameter index i:
    orig = params[i]
    numeric = fp::central_difference([&](double h) {
        params[i] = orig + h;  return f(params);
    }, 0.0, eps)              // fp computes (f(+h) - f(-h)) / (2h)
    params[i] = orig
    analytic = grad_f(params)[i]
    rel_err = |numeric - analytic| / max(1e-8, |numeric| + |analytic|)
    passed &= rel_err <= tolerance
```

Rules:

- Always central differences (`fp::central_difference`), never forward.
- `eps = 1e-6` for `double`; fp documents the truncation-vs-round-off trade-off.
- For a scalar function, prefer `fp::derivative` (exact, one pass) and compare
  *that* to the analytic gradient; use finite differences to validate the AD.
- The harness is a test utility; it is not compiled into release paths.
- Every NN layer and every loss gradient has a gradient-check test.

Tests (`test/gradient_check_test.cpp`):

- a known-correct gradient (e.g. `f(x) = sum x^2`, grad `2x`) passes
- a deliberately wrong gradient fails with `worst_index` pointing at the
  offending entry
- `fp::ad::derivative` and `fp::central_difference` agree within `1e-6` on a
  smooth function

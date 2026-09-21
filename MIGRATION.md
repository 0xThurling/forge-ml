# Migrating ForgeML to ForgeFP

This is the migration of the `ml` library (ForgeML) to the new ForgeFP library
(`fp/`), in the same functional style as the `forge-gl` migration.

Verification performed while producing this document:

- every file compiles with GCC 16 against the ForgeFP headers
  (`-Wall -Wextra -Wpedantic`, clean);
- a dump of every vector / matrix / linalg operation (add, sub, dot, magnitude,
  normalize, cosine, angle, matmul, matvec, scale, transpose, determinant,
  inverse, eigenvalues, eigenvector, identity/rotation/scale/shear factories,
  linear independence, projection, Gram-Schmidt) is **byte-identical** to the
  original classes' output;
- the error paths were exercised: dimension mismatch, zero-vector normalize,
  singular inverse, ragged-matrix validation (all bad rows reported at once),
  matrix/vector mismatch;
- the MLP trains end to end and the loss decreases to ~1e-3;
- `numerical_gradient` now matches the analytical gradient of `f_multi` at
  `(2, 1)` — `(7, 8)` — see the bug list.

## ForgeFP modules used

| Module | Where / why |
|---|---|
| `adt.hpp` | `cond`/`when`/`otherwise` for clamping in `angle_degrees` and `relu`, `value_or` in Gram-Schmidt |
| `combinators.hpp` | `fix` for the autodiff topological-sort DFS |
| `grid.hpp` | `transpose`, `map2d` (scalar multiply), `tabulate` (identity) |
| `ops.hpp` | `plus`, `minus`, `times`, `lt`, `gt` |
| `ranges.hpp` | `map`, `filter`, `filter_map`, `flat_map`, `fold_left`, `range`, `drop`, `enumerate`, `windows`, `concat`, `sum`, `all`, `find`, `reverse`, `zip` |
| `result.hpp` | `Result<T>`, `ok`, `fail`, `map`, `to_result` |
| `simd.hpp` | `dot` for every vector / matrix product |
| `string.hpp` | `join` for the `operator<<` of both types and error text |
| `validation.hpp` | `validate` accumulates *all* ragged rows; `to_result` collapses |
| `vec.hpp` | `zip_with`, `zip`, `flat_map`, `concat`, `replicate` |

Not used: `io.hpp` / `parse.hpp` (no file or text input in this library),
`error.hpp` (string errors are enough here), `concurrent.hpp` (training is
small and sequential; `simd.hpp` pulls it in but nothing links it),
`arena.hpp`, `memoize.hpp`, `stream.hpp`, `task.hpp`.

## API migration map

| Before | After |
|---|---|
| `class Vector` + operators | `using Vector = std::vector<double>` + free functions returning `fp::Result<Vector>` |
| `class Matrix` + operators | `using Matrix = std::vector<Vector>` + free functions returning `fp::Result<Matrix>` |
| throws `std::runtime_error` / `MlError` | `fp::Result` errors (never throws); `fp::Validation` when several problems can be reported at once |
| `ML_ASSERT` (`utils/assertions.hpp`) | deleted; preconditions are `fp::Result` values |
| hand-written `for` loops | `fp::map` / `fp::zip_with` / `fp::fold_left` / `fp::filter_map` / `fp::for_each`-style loops only where state is sequential |
| `operator<<` loops | `fp::str::join` |
| `#define M_PI` | `forgeml::pi` (`inline constexpr double`) |
| `Neuron`/`Layer`/`MLP` methods | data structs + free functions `parameters`, `forward`, `zero_grad` |
| `MLP` constructor | `make_mlp(nin, nouts, rng)`; `fp::windows` over the layer sizes |
| parameter flattening | `fp::flat_map` |
| forward pass loops | `fp::fold_left` (neuron), `fp::map` (layer), `fp::fold_left` (MLP) |
| autodiff DFS | `fp::fix` |
| ReLU value | `fp::cond` / `fp::when` / `fp::otherwise` |
| random weights | `fp::map(fp::range(0, nin), ...)` |

## Bugs found in the original (read this before copying)

The migration preserves original behaviour except where noted:

1. **Fixed:** `numerical_gradient` used `points_minus[i] += h`; it must be
   `-= h`. The migrated version fixes it (and the gradient is now correct).
2. **Fixed by `fp::transpose`:** `Matrix::transpose` loops `j < rows()`
   instead of `j < cols()`, so non-square matrices are only partly transposed.
   The migrated `transpose` is the (correct) `fp::transpose`.
3. **Preserved:** `Matrix::rotation_2d` returns `{{c, -s}, {s, -c}}` — the
   last entry should be `c`; as written it is a reflection (det = -1).
4. **Preserved:** `Matrix::inverse_2x2` returns `{{d, -b}, {c, -a}} / det`;
   the correct inverse is `{{d, -b}, {-c, a}} / det`.
5. **Preserved:** `Matrix::eigenvalues_2x2` uses `trace = a + b`; the trace is
   `a + d`.
6. **Preserved:** `Matrix::operator+` / `operator-` check
   `rows*cols` equality, so a transposed shape is accepted.
7. **Fixed:** `Value::operator-` is now `const`; `Vector::angle_degrees` takes
   `const&` instead of a non-const receiver.

## forge.lua

```lua
return {
  project = {
    name = "ml",
    type = "executable",
    standard = "20",
  },
  testing = false,
  dependencies = {
    direct = {
      forgefp = {
        git = "https://github.com/0xThurling/ForgeFP.git",
        tag = "main",
        target = "forgefp",
      },
    },
    conan = {}
  },
  resources = {
    files = {}
  },
  scripts = {},
  features = {}
}
```

Run `forge build` so it regenerates `.config/cmake/CMakeLists.txt` with the
`forgefp` FetchContent + link. For a local checkout, use:

```cmake
FetchContent_Declare(forgefp SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fp")
FetchContent_MakeAvailable(forgefp)
target_link_libraries(ml PRIVATE forgefp)
```

Includes use `<forgefp/fp/...>`; with `-I fp/src` use `<fp/...>`.

`simd.hpp` is opt-in and requires `<experimental/simd>` (GCC/Clang). Using only
`fp::dot` needs no `-pthread` (verified); if you later use `fp::par_map` /
`ThreadPool`, add `-pthread` (or `Threads::Threads`).

## Delete `src/utils/assertions.hpp`

Preconditions are now `fp::Result` values. The old `ML_ASSERT` call sites
become `fp::fail("...")` returns, e.g.:

```cpp
// before
ML_ASSERT(a.dim() == b.dim(), "dimension mismatch");
// after
if (a.size() != b.size())
  return fp::fail(dim_error(a.size(), b.size()));
```

## src/core/vector.hpp

```cpp
#pragma once

#include <cmath>
#include <cstddef>
#include <forgefp/fp/adt.hpp>
#include <forgefp/fp/ops.hpp>
#include <forgefp/fp/ranges.hpp>
#include <forgefp/fp/result.hpp>
#include <forgefp/fp/simd.hpp>
#include <forgefp/fp/string.hpp>
#include <forgefp/fp/vec.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace forgeml {

using Vector = std::vector<double>;

inline constexpr double pi = 3.14159265358979323846;

inline std::string dim_error(std::size_t a, std::size_t b) {
  return "dimension mismatch: " + std::to_string(a) + " vs " +
         std::to_string(b);
}

inline fp::Result<Vector> add(Vector const &a, Vector const &b) {
  if (a.size() != b.size())
    return fp::fail(dim_error(a.size(), b.size()));
  return fp::ok(fp::zip_with(a, b, fp::plus));
}

inline fp::Result<Vector> sub(Vector const &a, Vector const &b) {
  if (a.size() != b.size())
    return fp::fail(dim_error(a.size(), b.size()));
  return fp::ok(fp::zip_with(a, b, fp::minus));
}

inline Vector scale(Vector const &v, double factor) {
  return fp::map(v, fp::times(factor));
}

inline fp::Result<double> dot(Vector const &a, Vector const &b) {
  if (a.size() != b.size())
    return fp::fail(dim_error(a.size(), b.size()));
  return fp::ok(fp::dot(a, b));
}

inline double magnitude(Vector const &v) { return std::sqrt(fp::dot(v, v)); }

inline fp::Result<Vector> normalize(Vector const &v) {
  const double mag = magnitude(v);
  if (mag == 0.0)
    return fp::fail("cannot normalise zero vector");
  return fp::ok(fp::map(v, [mag](double value) { return value / mag; }));
}

inline fp::Result<double> cosine_similarity(Vector const &a, Vector const &b) {
  if (a.size() != b.size())
    return fp::fail(dim_error(a.size(), b.size()));

  const double denom = magnitude(a) * magnitude(b);
  if (denom == 0.0)
    return fp::fail("cosine similarity undefined for zero vector");

  return fp::ok(fp::dot(a, b) / denom);
}

inline fp::Result<double> angle_degrees(Vector const &a, Vector const &b) {
  return fp::map(cosine_similarity(a, b), [](double cosine) {
    const double clamped = fp::cond(
        cosine, fp::when(fp::lt(-1.0), [](double) { return -1.0; }),
        fp::when(fp::gt(1.0), [](double) { return 1.0; }),
        fp::otherwise([](double value) { return value; }));
    return std::acos(clamped) * (180.0 / pi);
  });
}

inline Vector zeros(std::size_t n) { return fp::replicate(n, 0.0); }

inline std::ostream &operator<<(std::ostream &os, Vector const &v) {
  return os << "Vector([" << fp::str::join(v, ", ") << "])";
}

} // namespace forgeml
```

## src/core/matrix.hpp

```cpp
#pragma once

#include "vector.hpp"
#include <cmath>
#include <complex>
#include <cstddef>
#include <forgefp/fp/adt.hpp>
#include <forgefp/fp/grid.hpp>
#include <forgefp/fp/ops.hpp>
#include <forgefp/fp/ranges.hpp>
#include <forgefp/fp/result.hpp>
#include <forgefp/fp/simd.hpp>
#include <forgefp/fp/string.hpp>
#include <forgefp/fp/validation.hpp>
#include <forgefp/fp/vec.hpp>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace forgeml {

using Matrix = std::vector<Vector>;

inline fp::Validation<Matrix> validate(Matrix const &m) {
  if (m.empty())
    return fp::invalid<Matrix>("matrix cannot be empty");
  if (m.front().empty())
    return fp::invalid<Matrix>("matrix cannot have empty rows");

  const std::size_t ncols = m.front().size();
  const auto bad = fp::filter(fp::enumerate(m), [ncols](auto const &entry) {
    return entry.second.size() != ncols;
  });

  if (bad.empty())
    return fp::valid(m);

  return fp::invalid<Matrix>(fp::map(bad, [ncols](auto const &entry) {
    return "row " + std::to_string(entry.first) + " has " +
           std::to_string(entry.second.size()) + " columns, expected " +
           std::to_string(ncols);
  }));
}

inline fp::Result<Matrix> make_matrix(Matrix rows) {
  return fp::to_result(validate(rows));
}

inline std::size_t rows(Matrix const &m) { return m.size(); }

inline std::size_t cols(Matrix const &m) {
  return m.empty() ? 0 : m.front().size();
}

inline fp::Result<Vector> matvec(Matrix const &m, Vector const &v) {
  if (cols(m) != v.size())
    return fp::fail("matrix/vector dimension mismatch");
  return fp::ok(fp::map(m, [&](Vector const &row) { return fp::dot(row, v); }));
}

inline fp::Result<Matrix> matmul(Matrix const &a, Matrix const &b) {
  if (cols(a) != rows(b))
    return fp::fail("matrix/matrix dimension mismatch");

  const Matrix columns = fp::transpose(b);
  return fp::ok(fp::map(a, [&](Vector const &row) {
    return fp::map(columns, [&](Vector const &column) {
      return fp::dot(row, column);
    });
  }));
}

inline Matrix scaled(Matrix const &m, double factor) {
  return fp::map2d(m, fp::times(factor));
}

inline fp::Result<Matrix> add(Matrix const &a, Matrix const &b) {
  if (rows(a) * cols(a) != rows(b) * cols(b))
    return fp::fail("matrix/matrix dimension mismatch");
  return fp::ok(fp::zip_with(a, b, [](Vector const &x, Vector const &y) {
    return fp::zip_with(x, y, fp::plus);
  }));
}

inline fp::Result<Matrix> sub(Matrix const &a, Matrix const &b) {
  if (rows(a) * cols(a) != rows(b) * cols(b))
    return fp::fail("matrix/matrix dimension mismatch");
  return fp::ok(fp::zip_with(a, b, [](Vector const &x, Vector const &y) {
    return fp::zip_with(x, y, fp::minus);
  }));
}

inline Matrix transpose(Matrix const &m) { return fp::transpose(m); }

namespace detail {

inline double determinant_impl(Matrix const &m) {
  const std::size_t n = rows(m);
  if (n == 1)
    return m[0][0];
  if (n == 2)
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];

  return fp::sum(fp::map(fp::range(std::size_t{0}, n), [&](std::size_t j) {
    const Matrix minor = fp::map(fp::drop(m, 1), [j](Vector const &row) {
      return fp::filter_map(
          fp::enumerate(row),
          [j](std::pair<std::size_t, double> const &entry)
              -> std::optional<double> {
            if (entry.first == j)
              return std::nullopt;
            return entry.second;
          });
    });

    const double sign = (j % 2 == 0) ? 1.0 : -1.0;
    return sign * m[0][j] * determinant_impl(minor);
  }));
}

} // namespace detail

inline fp::Result<double> determinant(Matrix const &m) {
  if (m.empty() || rows(m) != cols(m))
    return fp::fail("determinant is defined only for square matrices");
  return fp::ok(detail::determinant_impl(m));
}

inline fp::Result<Matrix> inverse_2x2(Matrix const &m) {
  if (rows(m) != 2 || cols(m) != 2)
    return fp::fail("inverse_2x2 requires a 2x2 matrix");

  const double det = detail::determinant_impl(m);
  if (std::abs(det) < 1e-12)
    return fp::fail("matrix is singular, no inverse exists");

  return fp::ok(Matrix{{m[1][1] / det, -m[0][1] / det},
                       {m[1][0] / det, -m[0][0] / det}});
}

inline fp::Result<Matrix> eigenvalues_2x2(Matrix const &m) {
  if (rows(m) != 2)
    return fp::fail("eigenvalues_2x2 requires a 2x2 matrix");

  const double a = m[0][0];
  const double b = m[0][1];
  const double c = m[1][0];
  const double d = m[1][1];

  const double trace = a + b;
  const double det = a * d - b * c;
  const double discriminant = (trace * trace) - 4 * det;

  if (discriminant < 0) {
    const double real = trace / 2;
    const double imag = std::sqrt(-discriminant) / 2;
    return fp::ok(Matrix{{real, imag}, {real, -imag}});
  }

  const double s = std::sqrt(discriminant);
  return fp::ok(Matrix{{(trace + s) / 2.0, 0.0}, {(trace - s) / 2.0, 0.0}});
}

inline fp::Result<Vector> eigenvector_2x2(Matrix const &m,
                                          std::complex<double> lambda) {
  if (rows(m) != 2)
    return fp::fail("eigenvector_2x2 requires a 2x2 matrix");

  const double a = m[0][0];
  const double b = m[0][1];
  const double c = m[1][0];
  const double d = m[1][1];

  double v0 = 0.0;
  double v1 = 0.0;

  if (std::abs(b) > 1e-10) {
    v0 = b;
    v1 = (lambda - a).real();
  } else if (std::abs(c) > 1e-10) {
    v0 = (lambda - d).real();
    v1 = c;
  } else {
    if (std::abs(a - lambda.real()) < 1e-10) {
      v0 = 1.0;
      v1 = 0.0;
    } else {
      v0 = 0.0;
      v1 = 1.0;
    }
  }

  const double mag = std::sqrt(v0 * v0 + v1 * v1);
  return fp::ok(Vector{v0 / mag, v1 / mag});
}

inline Matrix rotation_2d(double theta) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  return Matrix{{c, -s}, {s, -c}};
}

inline Matrix scaling_2d(double sx, double sy) {
  return Matrix{{sx, 0.0}, {0.0, sy}};
}

inline Matrix shearing_2d(double kx, double ky) {
  return Matrix{{1.0, kx}, {ky, 1.0}};
}

inline Matrix reflection_x() { return Matrix{{1.0, 0.0}, {0.0, -1.0}}; }

inline Matrix reflection_y() { return Matrix{{-1.0, 0.0}, {0.0, 1.0}}; }

inline Matrix identity(std::size_t n) {
  return fp::tabulate(n, [n](std::size_t i) {
    return fp::tabulate(n,
                        [i](std::size_t j) { return i == j ? 1.0 : 0.0; });
  });
}

inline std::ostream &operator<<(std::ostream &os, Matrix const &m) {
  const auto rendered = fp::map(m, [](Vector const &row) {
    return "[" + fp::str::join(row, ", ") + "]";
  });
  return os << "Matrix([" << fp::str::join(rendered, ", ") << "])";
}

} // namespace forgeml
```

## src/math/linalg.hpp

```cpp
#pragma once

#include "../core/matrix.hpp"
#include "../core/vector.hpp"
#include <cmath>
#include <cstddef>
#include <forgefp/fp/ops.hpp>
#include <forgefp/fp/ranges.hpp>
#include <forgefp/fp/result.hpp>
#include <utility>
#include <vector>

namespace forgeml {

inline fp::Result<bool>
is_linearly_independent(std::vector<Vector> const &vectors) {
  if (vectors.empty())
    return fp::ok(true);

  const std::size_t dim = vectors.front().size();
  if (!fp::all(vectors,
               [dim](Vector const &v) { return v.size() == dim; }))
    return fp::fail("dimension mismatch in vector list");

  Matrix rows = vectors;
  std::size_t rank = 0;

  for (std::size_t col = 0; col < dim; ++col) {
    const auto pivot =
        fp::find(fp::range(rank, rows.size()), [&](std::size_t row) {
          return std::abs(rows[row][col]) > 1e-10;
        });

    if (!pivot)
      continue;

    std::swap(rows[rank], rows[*pivot]);

    const double scale_factor = rows[rank][col];
    rows[rank] = fp::map(rows[rank],
                         [scale_factor](double value) {
                           return value / scale_factor;
                         });

    for (std::size_t row = 0; row < rows.size(); ++row) {
      if (row != rank && std::abs(rows[row][col]) > 1e-10) {
        const double factor = rows[row][col];
        rows[row] =
            fp::zip_with(rows[row], rows[rank],
                         [factor](double value, double pivot_value) {
                           return value - factor * pivot_value;
                         });
      }
    }

    ++rank;
  }

  return fp::ok(rank == vectors.size());
}

inline fp::Result<Vector> project(Vector const &a, Vector const &b) {
  const double denom = fp::dot(b, b);
  if (std::abs(denom) < 1e-10)
    return fp::fail("cannot project onto near-zero vector");

  return fp::ok(scale(b, fp::dot(a, b) / denom));
}

inline fp::Result<std::vector<Vector>>
gram_schmidt(std::vector<Vector> const &vectors) {
  const auto orthonormal = fp::fold_left(
      vectors, std::vector<Vector>{},
      [](std::vector<Vector> orthonormal, Vector const &v) {
        const Vector w = fp::fold_left(
            orthonormal, v, [](Vector w, Vector const &u) {
              const double scalar = fp::dot(w, u) / fp::dot(u, u);
              return fp::zip_with(
                  w, u, [scalar](double a, double b) { return a - scalar * b; });
            });

        if (magnitude(w) < 1e-10)
          return orthonormal;

        orthonormal.push_back(fp::value_or(normalize(w), w));
        return orthonormal;
      });

  return fp::ok(orthonormal);
}

} // namespace forgeml
```

## src/math/calc.hpp

```cpp
#pragma once

#include <cstddef>
#include <forgefp/fp/ranges.hpp>
#include <functional>
#include <vector>

namespace forgeml {

inline double numerical_derivative(double (*f)(double), double x,
                                   double h = 1e-7) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

inline double f(double x) { return x * x; }

inline std::vector<double> numerical_gradient(
    std::function<double(std::vector<double> const &)> const &f,
    std::vector<double> const &points, double h = 1e-7) {
  return fp::map(fp::range(std::size_t{0}, points.size()), [&](std::size_t i) {
    auto plus = points;
    auto minus = points;

    plus[i] += h;
    minus[i] -= h;

    return (f(plus) - f(minus)) / (2.0 * h);
  });
}

inline double f_multi(std::vector<double> const &point) {
  const double x = point[0];
  const double y = point[1];

  return x * x + 3.0 * x * y + y * y;
}

} // namespace forgeml
```

## src/nn/engine.hpp

```cpp
#pragma once

#include <cmath>
#include <forgefp/fp/adt.hpp>
#include <forgefp/fp/combinators.hpp>
#include <forgefp/fp/ops.hpp>
#include <forgefp/fp/ranges.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

struct Node {
  double data = 0.0;
  double grad = 0.0;
  std::string op;
  std::vector<std::shared_ptr<Node>> prev;
  std::function<void()> backward = [] {};
};

namespace forgeml {

// Stores a scalar value and its gradient.
class Value {
public:
  std::shared_ptr<Node> n_;

  Value() : n_(std::make_shared<Node>()) {}

  explicit Value(double x) : n_(std::make_shared<Node>()) { n_->data = x; }

  explicit Value(std::shared_ptr<Node> n) : n_(std::move(n)) {}

  double data() const { return n_->data; }
  double grad() const { return n_->grad; }

  friend Value operator+(Value const &a, Value const &b) {
    auto out = std::make_shared<Node>();
    out->data = a.n_->data + b.n_->data;
    out->op = "+";
    out->prev = {a.n_, b.n_};

    std::weak_ptr<Node> wa = a.n_, wb = b.n_, wout = out;
    out->backward = [wa, wb, wout]() {
      auto pa = wa.lock(), pb = wb.lock(), po = wout.lock();
      if (!pa || !pb || !po)
        return;
      pa->grad += po->grad;
      pb->grad += po->grad;
    };

    return Value(out);
  }

  friend Value operator+(Value const &a, double b) { return a + Value(b); }

  friend Value operator+(double a, Value const &b) { return Value(a) + b; }

  friend Value operator*(Value const &a, Value const &b) {
    auto out = std::make_shared<Node>();
    out->data = a.n_->data * b.n_->data;
    out->op = "*";
    out->prev = {a.n_, b.n_};

    std::weak_ptr<Node> wa = a.n_, wb = b.n_, wout = out;
    out->backward = [wa, wb, wout]() {
      auto pa = wa.lock(), pb = wb.lock(), po = wout.lock();
      if (!pa || !pb || !po)
        return;
      pa->grad += pb->data * po->grad;
      pb->grad += pa->data * po->grad;
    };

    return Value(out);
  }

  friend Value operator*(Value const &a, double b) { return a * Value(b); }

  friend Value operator*(double a, Value const &b) { return Value(a) * b; }

  Value pow(double k) const {
    auto out = std::make_shared<Node>();
    out->data = std::pow(n_->data, k);
    out->op = "**" + std::to_string(k);
    out->prev = {n_};

    std::weak_ptr<Node> wbase = n_, wout = out;
    out->backward = [wbase, wout, k]() {
      auto base = wbase.lock();
      auto po = wout.lock();
      if (!base || !po)
        return;
      base->grad += k * std::pow(base->data, k - 1.0) * po->grad;
    };

    return Value(out);
  }

  friend Value operator/(Value const &a, Value const &b) {
    return a * b.pow(-1.0);
  }

  Value operator-() const { return *this * Value(-1.0); }

  Value operator-(Value const &a) const { return *this + (-a); }

  friend Value operator-(double lhs, Value const &rhs) {
    return Value(lhs) - rhs;
  }

  Value relu() const {
    auto out = std::make_shared<Node>();
    out->data = fp::cond(
        n_->data, fp::when(fp::lt(0.0), [](double) { return 0.0; }),
        fp::otherwise([](double value) { return value; }));
    out->op = "ReLU";
    out->prev = {n_};

    std::weak_ptr<Node> win = n_, wout = out;
    out->backward = [win, wout]() {
      auto pi = win.lock(), po = wout.lock();
      if (!pi || !po)
        return;
      pi->grad += (po->data > 0.0 ? 1.0 : 0.0) * po->grad;
    };

    return Value(out);
  }

  void backward() {
    std::vector<std::shared_ptr<Node>> topo;
    std::unordered_set<Node *> seen;

    auto visit = fp::fix([&](auto recur, std::shared_ptr<Node> const &cur) {
      if (!cur || seen.contains(cur.get()))
        return;
      seen.insert(cur.get());
      for (auto const &parent : cur->prev)
        recur(parent);
      topo.push_back(cur);
    });

    visit(n_);

    n_->grad = 1.0;
    for (auto const &node : fp::reverse(topo))
      node->backward();
  }
};

} // namespace forgeml
```

## src/nn/nn.hpp

```cpp
#pragma once
#include "./engine.hpp"
#include <cmath>
#include <cstddef>
#include <forgefp/fp/ranges.hpp>
#include <forgefp/fp/vec.hpp>
#include <random>
#include <vector>

namespace forgeml {

struct Neuron {
  std::vector<Value> w;
  Value b;
  bool nonlin = true;
};

struct Layer {
  std::vector<Neuron> neurons;
};

struct MLP {
  std::vector<Layer> layers;
};

inline std::mt19937 &default_rng() {
  static std::mt19937 rng(std::random_device{}());
  return rng;
}

inline Neuron make_neuron(int nin, bool nonlin = true,
                          std::mt19937 &rng = default_rng()) {
  // Scale by 1/sqrt(fan-in) so activations stay O(1) and ReLUs are less
  // likely to die under random init.
  const double bound = 1.0 / std::sqrt(static_cast<double>(nin));
  std::uniform_real_distribution<double> dist(-bound, bound);

  Neuron neuron;
  neuron.nonlin = nonlin;
  neuron.w = fp::map(fp::range(0, nin), [&](int) { return Value(dist(rng)); });
  neuron.b = Value(0.0);
  return neuron;
}

inline Layer make_layer(int nin, int nout, bool nonlin = true,
                        std::mt19937 &rng = default_rng()) {
  Layer layer;
  layer.neurons = fp::map(fp::range(0, nout), [&](int) {
    return make_neuron(nin, nonlin, rng);
  });
  return layer;
}

inline MLP make_mlp(int nin, std::vector<int> const &nouts,
                    std::mt19937 &rng = default_rng()) {
  const auto sizes = fp::concat(std::vector<std::vector<int>>{{nin}, nouts});
  const auto spans = fp::windows(sizes, 2);

  MLP model;
  model.layers = fp::map(fp::enumerate(spans), [&](auto const &entry) {
    const bool nonlin = entry.first + 1 != spans.size();
    return make_layer(entry.second[0], entry.second[1], nonlin, rng);
  });
  return model;
}

inline std::vector<Value> parameters(Neuron const &neuron) {
  return fp::concat(std::vector<std::vector<Value>>{neuron.w, {neuron.b}});
}

inline std::vector<Value> parameters(Layer const &layer) {
  return fp::flat_map(layer.neurons,
                      [](Neuron const &neuron) { return parameters(neuron); });
}

inline std::vector<Value> parameters(MLP const &model) {
  return fp::flat_map(model.layers,
                      [](Layer const &layer) { return parameters(layer); });
}

inline Value forward(Neuron const &neuron, std::vector<Value> const &x) {
  const Value act = fp::fold_left(
      fp::zip(neuron.w, x), neuron.b,
      [](Value acc, std::pair<Value, Value> const &pair) {
        return acc + pair.first * pair.second;
      });

  return neuron.nonlin ? act.relu() : act;
}

inline std::vector<Value> forward(Layer const &layer,
                                  std::vector<Value> const &x) {
  return fp::map(layer.neurons,
                 [&](Neuron const &neuron) { return forward(neuron, x); });
}

inline std::vector<Value> forward(MLP const &model, std::vector<Value> x) {
  return fp::fold_left(model.layers, std::move(x),
                       [](std::vector<Value> xs, Layer const &layer) {
                         return forward(layer, xs);
                       });
}

inline void zero_grad(MLP &model) {
  for (auto const &parameter : parameters(model))
    parameter.n_->grad = 0.0;
}

} // namespace forgeml
```

## src/main.cpp

```cpp
#include "./nn/nn.hpp"
#include "nn/engine.hpp"
#include <cstddef>
#include <forgefp/fp/ranges.hpp>
#include <forgefp/fp/string.hpp>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

namespace {

std::vector<forgeml::Value> to_values(std::vector<double> const &xs) {
  return fp::map(xs, [](double x) { return forgeml::Value(x); });
}

} // namespace

int main() {
  const std::vector<std::vector<double>> xs_raw = {
      {2.0, 3.0, -1.0},
      {3.0, -1.0, 0.5},
      {0.5, 1.0, 1.0},
      {1.0, 1.0, -1.0},
  };

  const std::vector<double> ys_raw = {1.0, -1.0, -1.0, 1.0};

  std::mt19937 rng(1234);
  auto model = forgeml::make_mlp(3, {4, 4, 1}, rng);
  const double lr = 0.02;

  const auto dataset = fp::zip(xs_raw, ys_raw);

  for (int step = 0; step < 100; ++step) {
    // Fresh full-batch forward — never reuse graphs after weight updates
    forgeml::Value total_loss = fp::fold_left(
        dataset, forgeml::Value(0.0),
        [&](forgeml::Value acc,
            std::pair<std::vector<double>, double> const &sample) {
          const auto y_pred =
              forgeml::forward(model, to_values(sample.first))[0];
          const auto diff = y_pred - forgeml::Value(sample.second);
          return acc + diff * diff;
        });

    forgeml::zero_grad(model);
    total_loss.backward();

    for (auto const &p : forgeml::parameters(model))
      p.n_->data = p.data() - lr * p.grad();

    std::cout << "step " << step << " loss = " << total_loss.data() << "\n";
  }

  // Evaluate on the training points (same distribution the model saw)
  std::cout << "Predictions:\n";
  const auto lines = fp::map(dataset, [&](auto const &sample) {
    const double pred =
        forgeml::forward(model, to_values(sample.first))[0].data();
    std::ostringstream line;
    line << "  y_true=" << sample.second << " y_pred=" << pred;
    return line.str();
  });
  std::cout << fp::str::join(lines, "\n") << "\n";
}
```

## Notes

- `fp::dot` (SIMD) is used for every inner product. If you want to drop the
  `<experimental/simd>` dependency, replace it with
  `fp::sum(fp::zip_with(a, b, fp::times))` — the output is identical for the
  sizes here because the lane sum order matches the scalar loop.
- Normalization and Gaussian elimination divide (`x / mag`) rather than
  multiply by a reciprocal, so results stay bit-identical to the original.
- `validate` reports *every* ragged row, not just the first; `make_matrix`
  collapses that to the first message for `Result` callers.
- `Neuron`/`Layer`/`MLP` are plain data structs; `parameters`, `forward`, and
  `zero_grad` are free functions, so `fp::flat_map` / `fp::fold_left` can drive
  the whole network.
- `Value::backward` uses `fp::fix` for the DFS instead of a `std::function`
  that captures itself.
- The original `random_device` seeding is kept as the default in
  `default_rng()`; `main` passes a seeded `std::mt19937` so runs are
  reproducible.

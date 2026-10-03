#pragma once

#include "forgefp/fp/either.hpp"
#include "forgefp/fp/grid.hpp"
#include "forgefp/fp/ranges.hpp"
#include "forgefp/fp/result.hpp"
#include "forgefp/fp/simd.hpp"
#include "forgefp/fp/string.hpp"
#include "forgefp/fp/vec.hpp"
#include "vector.hpp"
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <forgefp/fp/validation.hpp>
#include <optional>
#include <string>
#include <vector>

namespace forgeml {
using Matrix = std::vector<Vector>;

inline fp::Validation<Matrix> validate(Matrix const &m) {
  if (m.empty()) {
    return fp::invalid<Matrix>("matrix cannot be empty");
  }

  if (m.front().empty()) {
    return fp::invalid<Matrix>("matrix cannot have empty rows");
  }

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
    return fp::map(columns,
                   [&](Vector const &column) { return fp::dot(row, column); });
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
      return fp::filter_map(fp::enumerate(row),
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

  return fp::ok(
      Matrix{{m[1][1] / det, -m[0][1] / det}, {-m[1][0] / det, m[0][0] / det}});
}

inline fp::Result<Matrix> eigenvalues_2x2(Matrix const &m) {
  if (rows(m) != 2 || cols(m) != 2)
    return fp::fail("eigenvalues_2x2 requires a 2x2 matrix");

  const double a = m[0][0];
  const double b = m[0][1];
  const double c = m[1][0];
  const double d = m[1][1];

  const double trace = a + d;
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
  if (rows(m) != 2 || cols(m) != 2)
    return fp::fail("eigenvector_2x2 requires a 2x2 matrix");

  if (std::abs(lambda.imag()) > 1e-10)
    return fp::fail("complex eigenvalue has no real eigenvector");

  const double a = m[0][0];
  const double b = m[0][1];
  const double c = m[1][0];
  const double d = m[1][1];
  const double l = lambda.real();

  double v0 = 0.0;
  double v1 = 0.0;

  if (std::abs(b) > 1e-10) {
    v0 = b;
    v1 = l - a;
  } else if (std::abs(c) > 1e-10) {
    v0 = l - d;
    v1 = c;
  } else {
    if (std::abs(a - l) < 1e-10) {
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
  return Matrix{{c, -s}, {s, c}};
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
    return fp::tabulate(n, [i](std::size_t j) { return i == j ? 1.0 : 0.0; });
  });
}

inline std::ostream &operator<<(std::ostream &os, Matrix const &m) {
  const auto rendered = fp::map(m, [](Vector const &row) {
    return "[" + fp::str::join(row, ", ") + "]";
  });

  return os << "Matrix([" << fp::str::join(rendered, ", ") << "])";
}
} // namespace forgeml

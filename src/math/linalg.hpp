#pragma once

#include "../core/matrix.hpp"
#include "../core/vector.hpp"
#include "forgefp/fp/adt.hpp"
#include "forgefp/fp/either.hpp"
#include "forgefp/fp/ranges.hpp"
#include "forgefp/fp/result.hpp"
#include "forgefp/fp/simd.hpp"
#include "forgefp/fp/vec.hpp"
#include <cstddef>
#include <cstdlib>

namespace forgeml {
inline fp::Result<bool>
is_linearly_independent(std::vector<Vector> const &vectors) {
  if (vectors.empty())
    return fp::ok(true);

  const std::size_t dim = vectors.front().size();
  if (!fp::all(vectors, [dim](Vector const &v) { return v.size() == dim; }))
    return fp::fail("dimension mismatch in vector list");

  Matrix rows = vectors;
  std::size_t rank = 0;

  for (std::size_t col = 0; col < dim; ++col) {
    const auto pivot =
        fp::find_if(fp::range(rank, rows.size()), [&](std::size_t row) {
          return std::abs(rows[row][col]) > 1e-10;
        });

    if (!pivot) {
      continue;
    }

    std::swap(rows[rank], rows[*pivot]);

    const double scale_factor = rows[rank][col];
    rows[rank] = fp::map(rows[rank], [scale_factor](double value) {
      return value / scale_factor;
    });

    for (std::size_t row = 0; row < rows.size(); ++row) {
      if (row != rank && std::abs(rows[row][col]) > 1e-10) {
        const double factor = rows[row][col];
        rows[row] = fp::zip_with(rows[row], rows[rank],
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
        const Vector w =
            fp::fold_left(orthonormal, v, [](Vector w, Vector const &u) {
              const double scalar = fp::dot(w, u) / fp::dot(u, u);
              return fp::zip_with(w, u, [scalar](double a, double b) {
                return a - scalar * b;
              });
            });

        if (magnitude(w) < 1e-10) {
          return orthonormal;
        }

        orthonormal.push_back(fp::value_or(normalize(w), w));
        return orthonormal;
      });

  return fp::ok(orthonormal);
}
} // namespace forgeml

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
    const double clamped =
        fp::cond(cosine, fp::arm(fp::below(-1.0), [](double) { return -1.0; }),
                 fp::arm(fp::above(1.0), [](double) { return 1.0; }),
                 fp::otherwise([](double value) { return value; }));
    return std::acos(clamped) * (180.0 / pi);
  });
}

inline Vector zeros(std::size_t n) { return fp::replicate(n, 0.0); }

inline std::ostream &operator<<(std::ostream &os, Vector const &v) {
  return os << "Vector([" << fp::str::join(v, ", ") << "])";
}
} // namespace forgeml

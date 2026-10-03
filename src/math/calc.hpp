#pragma once

#include "forgefp/fp/ranges.hpp"
#include "forgefp/fp/vec.hpp"
#include <cstddef>
#include <functional>
#include <iterator>
#include <vector>

namespace forgeml {
inline double numerical_derivative(double (*f)(double), double x,
                                   double h = 1e-7) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

inline double f(double x) { return x * x; }

inline std::vector<double> numerical_derivative(
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

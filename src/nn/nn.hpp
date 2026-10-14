#pragma once
#include "./engine.hpp"
#include "forgefp/fp/either.hpp"
#include "forgefp/fp/ranges.hpp"
#include "forgefp/fp/vec.hpp"
#include <cmath>
#include <cstddef>
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
  layer.neurons = fp::map(fp::range(0, nout),
                          [&](int) { return make_neuron(nin, nonlin, rng); });
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
  const Value act =
      fp::fold_left(fp::zip(neuron.w, x), neuron.b,
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

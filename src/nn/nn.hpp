#pragma once
#include "./engine.hpp"
#include <cstddef>
#include <random>
#include <vector>

struct Module {
  virtual std::vector<forgeml::Value> parameters() { return {}; }
  void zero_grad() {
    for (auto &p : parameters())
      p.n_->grad = 0.0;
  }
};

struct Neuron : Module {
  std::vector<forgeml::Value> w;
  forgeml::Value b;
  bool nonlin;

  Neuron(int nin, bool nonlin_ = true) : w(nin), b(0.0), nonlin(nonlin_) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto &wi : w)
      wi = forgeml::Value(dist(rng));

    b = forgeml::Value(0.0);
  }

  forgeml::Value operator()(const std::vector<forgeml::Value> &x) const {
    forgeml::Value act = b;

    for (std::size_t i = 0; i < w.size(); ++i) {
      act = act + w[i] * x[i];
    }

    return nonlin ? act.relu() : act;
  }

  std::vector<forgeml::Value> parameters() override {
    auto params = w;
    params.push_back(b);
    return params;
  }
};

struct Layer : Module {
  std::vector<Neuron> neurons;

  Layer(int nin, int nout, bool nonlin = true) {
    neurons.reserve(nout);
    for (int i = 0; i < nout; ++i) {
      neurons.emplace_back(nin, nonlin);
    }
  }

  std::vector<forgeml::Value>
  operator()(const std::vector<forgeml::Value> &x) const {
    std::vector<forgeml::Value> out;
    out.reserve(neurons.size());
    for (auto &n : neurons) {
      out.push_back(n(x));
    }
    return out;
  }

  std::vector<forgeml::Value> parameters() override {
    std::vector<forgeml::Value> params;

    for (auto &n : neurons) {
      auto np = n.parameters();
      params.insert(params.end(), np.begin(), np.end());
    }

    return params;
  }
};

struct MLP : Module {
  std::vector<Layer> layers;

  MLP(int nin, const std::vector<int> &nouts) {
    std::vector<int> sz;
    sz.reserve(nouts.size());
    sz.push_back(nin);

    for (int o : nouts) {
      sz.push_back(o);
    }

    for (std::size_t i = 0; i + 1 < sz.size(); ++i) {
      bool nonlin = (i + 1 != sz.size() - 1);
      layers.emplace_back(sz[i], sz[i + 1], nonlin);
    }
  }

  std::vector<forgeml::Value> operator()(std::vector<forgeml::Value> x) const {
    for (auto &layer : layers)
      x = layer(x);

    return x;
  }

  std::vector<forgeml::Value> parameters() override {
      std::vector<forgeml::Value> params;
      for (auto& layer: layers) {
          auto lp = layer.parameters();
          params.insert(params.end(), lp.begin(), lp.end());
      }

      return params;
  }
};

#include "./nn/nn.hpp"
#include "nn/engine.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
  std::vector<std::vector<double>> xs_raw = {
      {2.0, 3.0, -1.0},
      {3.0, -1.0, 0.5},
      {0.5, 1.0, 1.0},
      {1.0, 1.0, -1.0},
  };

  std::vector<double> ys_raw = {1.0, -1.0, -1.0, 1.0};

  MLP model(3, {4, 4, 1});
  const double lr = 0.02;

  for (int step = 0; step < 100; ++step) {
    // Fresh full-batch forward — never reuse graphs after weight updates
    forgeml::Value total_loss{0.0};

    for (std::size_t i = 0; i < xs_raw.size(); ++i) {
      std::vector<forgeml::Value> x;
      x.reserve(xs_raw[i].size());
      for (double v : xs_raw[i])
        x.emplace_back(v);

      auto y_pred = model(x)[0];
      auto diff = y_pred - forgeml::Value(ys_raw[i]);
      total_loss = total_loss + diff * diff;
    }

    model.zero_grad();
    total_loss.backward();

    for (auto &p : model.parameters()) {
      p.n_->data = p.data() - lr * p.grad();
    }

    std::cout << "step " << step << " loss = " << total_loss.data() << "\n";
  }

  // Evaluate on the training points (same distribution the model saw)
  std::cout << "Predictions:\n";
  for (std::size_t i = 0; i < xs_raw.size(); ++i) {
    std::vector<forgeml::Value> x;
    for (double v : xs_raw[i])
      x.emplace_back(v);

    double pred = model(x)[0].data();
    std::cout << "  y_true=" << ys_raw[i] << " y_pred=" << pred << "\n";
  }
}

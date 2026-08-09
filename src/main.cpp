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

  for (int step = 0; step < 100; ++step) {
    std::vector<forgeml::Value> losses;
    losses.reserve(xs_raw.size());

    for (std::size_t i = 0; i < xs_raw.size(); ++i) {
      std::vector<forgeml::Value> x;
      x.reserve(xs_raw[i].size());
      for (double v : xs_raw[i])
        x.emplace_back(v);

      auto out = model(x);
      forgeml::Value y_pred = out[0];
      forgeml::Value y_true{ys_raw[i]};

      forgeml::Value diff = y_pred - y_true;
      losses.push_back(diff * diff);

      // total loss
      forgeml::Value total_loss{0.0};
      for (auto &li : losses)
        total_loss = total_loss + li;

      // backward
      model.zero_grad();
      total_loss.backward();

      // SGD step
      for (auto &p : model.parameters()) {
        double updated = p.data() - 0.001 * p.grad();
        // you'd want a setter on Value; for sketch;
        p.n_->data = updated;
      }

      std::cout << "step " << step << " loss = " << total_loss.data() << "\n";
    }
  }

  std::vector<forgeml::Value> x = {forgeml::Value(5.1), forgeml::Value(3.5),
                                   forgeml::Value(1.4)};

  std::vector<forgeml::Value> output = model(x);

  double score = output[0].data();
  std::cout << "Raw prediction: " << score << "\n";
}

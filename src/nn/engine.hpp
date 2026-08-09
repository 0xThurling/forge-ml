#include <cmath>
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
// Stores scalar values and it's gradient
class Value {
  std::shared_ptr<Node> n_;

public:
  Value() : n_(std::make_shared<Node>()) {}
  explicit Value(double x) : n_(std::make_shared<Node>()) { n_->data = x; }
  explicit Value(std::shared_ptr<Node> n) : n_(std::move(n)) {}

  double data() const { return n_->data; }
  double grad() const { return n_->grad; }

  friend Value operator+(const Value &a, const Value &b) {
    auto out = std::make_shared<Node>();
    out->data = a.n_->data + b.n_->data;
    out->op = "+";
    out->prev = {a.n_, b.n_};

    std::weak_ptr<Node> wa = a.n_, wb = b.n_, wout = out;
    out->backward = [wa, wb, wout]() {
      auto pa = wa.lock(), pb = wb.lock(), po = wout.lock();
      if (!pa || !pb || !po) return;
      pa->grad += po->grad;
      pb->grad += po->grad;
    };

    return Value(out);
  }

  friend Value operator+(const Value& a, double b) {
      return a + Value(b);
  }

  friend Value operator+(double a, const Value& b) {
      return Value(a) + b;
  }

  friend Value operator*(const Value& a, const Value& b) {
    auto out = std::make_shared<Node>();
    out->data = a.n_->data * b.n_->data;
    out->op = "*";
    out->prev = {a.n_, b.n_};

    std::weak_ptr<Node> wa = a.n_, wb = b.n_, wout = out;
    out->backward = [wa, wb, wout]() {
      auto pa = wa.lock(), pb = wb.lock(), po = wout.lock();
      if (!pa || !pb || !po) return;
      pa->grad += pb->data * po->grad;
      pb->grad += pa->data * po->grad;
    };

    return Value(out);
  }

  friend Value operator*(const Value& a, double b) {
      return a * Value(b);
  }

  friend Value operator*(double a, const Value& b) {
      return Value(a) * b;
  }

  Value pow(double k) const {
    auto out = std::make_shared<Node>();
    out->data = std::pow(n_->data, k);
    out->op = "**" + std::to_string(k);
    out->prev = {n_};

    std::weak_ptr<Node> wbase = n_, wout = out;
    out->backward = [wbase, wout, k]() {
        auto base = wbase.lock();
        auto po = wout.lock();
        if (!base || !po) return;
        // d/dx (x^k) = k * x^(k-1)
        base->grad += k * std::pow(base->data, k - 1.0) * po->grad;
    };

    return Value(out);
  };

  friend Value operator/(const Value& a, const Value& b) {
    return a * b.pow(-1.0);
  }

  Value operator-() const {
    return *this * Value(-1.0);
  }

  Value operator-(const Value &a) {
      return *this + (-a);
  }

  friend Value operator-(double lhs, const Value &rhs) {
      return Value(lhs) - rhs;
  }

  Value relu() const {
      auto out = std::make_shared<Node>();
      out->data = n_->data < 0.0 ? 0.0 : n_->data;
      out->op = "ReLU";
      out->prev = {n_};

      std::weak_ptr<Node> win = n_, wout = out;
      out->backward = [win, wout] () {
        auto pi = win.lock(), po = wout.lock();
        if (!pi || !po) return;
        pi->grad += (po->data > 0.0 ? 1.0 : 0.0) * po->grad;
      };

      return Value(out);
  }

  void backward() {
      std::vector<std::shared_ptr<Node>> topo;
      std::unordered_set<Node*> seen;

      std::function<void(const std::shared_ptr<Node>&)> dfs =
          [&](const std::shared_ptr<Node>& cur){
            if (!cur || seen.contains(cur.get())) return;
            seen.insert(cur.get());
            for (auto& p : cur->prev) dfs(p);
            topo.push_back(cur);
          };

      dfs(n_);
      n_->grad = 1.0;
      for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->backward();
      }
  }
};
} // namespace forgeml

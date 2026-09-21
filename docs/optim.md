# optim — parameter update rules

Optimizers take gradients and update parameters in place. They know nothing
about losses or models, only about values and gradients.

Depends on: `core`, ForgeFP.

Files: `optimizer.hpp`, `gradient_descent.hpp`, `sgd.hpp`, `adam.hpp`,
`lr_scheduler.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `optimizer.hpp` | `fp::Result`, `fp::fail`, `fp::Buffer` (per-parameter state) |
| `gradient_descent.hpp` | `fp::zip_transform_inplace`, `fp::axpy_inplace`, `fp::zip_with` |
| `sgd.hpp` | `fp::zip_transform_inplace`, `fp::axpy_inplace`, `fp::scale` |
| `adam.hpp` | `fp::for_each`, `fp::zip_for_each`, `fp::map`, `fp::axpy_inplace` |
| `lr_scheduler.hpp` | `std::cos` only; schedules are pure scalar math |

Optimizer updates are the canonical zero-cost path: `w -= lr * g` is
`fp::axpy_inplace(w, -lr, g)`, and momentum is
`fp::zip_transform_inplace`. No step allocates after the first.

---

## The parameter contract

Optimizers are generic over a *parameter* type that provides:

```cpp
template <class P>
concept ParameterLike = requires(P p, P const cp) {
  { p.value() } -> std::convertible_to<Vector<double> &>;   // or Matrix<double>&
  { p.grad() } -> std::convertible_to<Vector<double> const &>;
  { p.zero_grad() } -> std::same_as<void>;
};
```

Two concrete forms exist:

- `nn::Parameter` (see `nn.md`) — value + grad + zero_grad, used by layers.
- Plain `(Vector<double> value, Vector<double> grad)` pairs used by the
  classical models. The optimizer interface below takes the value and gradient
  separately so both work:

```cpp
namespace forgeml {

class Optimizer {
public:
  explicit Optimizer(double learning_rate) : lr_(learning_rate) {}
  virtual ~Optimizer() = default;

  virtual void step(Vector<double> &value, Vector<double> const &grad) = 0;
  virtual void step(Matrix<double> &value, Matrix<double> const &grad) = 0;

  double learning_rate() const { return lr_; }
  void set_learning_rate(double lr) { lr_ = lr; }

protected:
  double lr_;
};
}
```

Rules:

- `step` mutates `value` only; `grad` is const.
- Both overloads must be implemented (a `Matrix` parameter is just a batch of
  vectors, but optimizers with per-parameter state keyed by position need the
  flat layout).
- `learning_rate > 0` is asserted.
- Optimizers hold per-parameter state (momentum, Adam moments) keyed by the
  *address* of `value` in a `std::unordered_map<void const *, State>`. This is
  how a single optimizer can update many parameters. Document the lifetime
  requirement: the parameter must not move while the optimizer is alive.
- `zero_grad` is the caller's job (`model.zero_grad()` /
  `network.zero_grad()`), not the optimizer's.

---

## `optim/gradient_descent.hpp`

```cpp
class GradientDescent : public Optimizer {
public:
  explicit GradientDescent(double lr, double momentum = 0.0);
  void step(Vector<double> &value, Vector<double> const &grad) override;
  void step(Matrix<double> &value, Matrix<double> const &grad) override;
private:
  double momentum_;
  std::unordered_map<void const *, Vector<double>> velocity_v_;
  std::unordered_map<void const *, Matrix<double>> velocity_m_;
};
```

Update rules:

```text
plain:    w <- w - lr * g
momentum: v <- momentum * v + g
          w <- w - lr * v
```

ForgeFP implementation (all allocation-free after the first step):

```cpp
// plain
fp::axpy_inplace(value, -lr_, grad);

// momentum: v = momentum * v + g ; w -= lr * v
fp::zip_for_each(velocity, grad, [&](double &v, double g) {
  v = momentum_ * v + g;
});
fp::axpy_inplace(value, -lr_, velocity);
```

Velocity state is stored per parameter (keyed by `&value`); use `fp::Buffer`
for it when the parameter count is known up front, or `std::vector` for the
map-based form.

Tests (`test/optim_test.cpp`):

- one step with `lr = 0.5`, `w = {2}`, `g = {4}` gives `w = {0}`
- momentum accumulates: two steps with `g = {1}` and `momentum = 0.9` move
  further than two plain steps
- matrix overload matches elementwise application

---

## `optim/sgd.hpp`

**Job:** the same update rule driven by mini-batches. The optimizer itself only
does one update per call; the *trainer* decides how often to call it.

```cpp
class SGD : public Optimizer {
public:
  explicit SGD(double lr, double momentum = 0.0, double weight_decay = 0.0);
  void step(Vector<double> &value, Vector<double> const &grad) override;
  void step(Matrix<double> &value, Matrix<double> const &grad) override;
private:
  double momentum_, weight_decay_;
  // same velocity maps as GradientDescent
};
```

Update rules:

```text
g' = g + weight_decay * w
v  <- momentum * v + g'
w  <- w - lr * v
```

ForgeFP implementation: one fused pass
`fp::zip_for_each(value, grad, ...)` that reads `w`, computes `g'`,
updates `v`, and writes `w`; or two passes with `fp::axpy_inplace` when
`weight_decay == 0`.

Tests:

- `weight_decay = 0`, `momentum = 0` matches `GradientDescent`
- decay shrinks weights even with zero gradient
- mini-batch training loop from `data/dataloader.hpp` converges on a toy
  linear problem

---

## `optim/adam.hpp`

**Job:** adaptive moment estimation. Required for transformers; useful
everywhere else.

```cpp
class Adam : public Optimizer {
public:
  explicit Adam(double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
                double eps = 1e-8, double weight_decay = 0.0,
                bool decoupled_weight_decay = false);   // true -> AdamW
  void step(Vector<double> &value, Vector<double> const &grad) override;
  void step(Matrix<double> &value, Matrix<double> const &grad) override;
private:
  struct State { Vector<double> m, v; std::size_t t = 0; };
  std::unordered_map<void const *, State> state_;
  double beta1_, beta2_, eps_, weight_decay_;
  bool decoupled_;
};
```

Update rules:

```text
t  <- t + 1
if decoupled (AdamW): w <- w - lr * weight_decay * w
m  <- beta1 * m + (1 - beta1) * g
v  <- beta2 * v + (1 - beta2) * g^2
m_hat <- m / (1 - beta1^t)
v_hat <- v / (1 - beta2^t)
w  <- w - lr * m_hat / (sqrt(v_hat) + eps)
```

`AdamW` is the same class with `decoupled_weight_decay = true`; provide the
alias so call sites read clearly:

```cpp
using AdamW = Adam;   // construct with decoupled_weight_decay = true
```

Rules:

- Bias correction uses the per-parameter step counter `t` (not a global one).
- For `decoupled == false`, L2 is added to the gradient (classic Adam) before
  the moment update.
- `eps` is added after the square root.

Tests:

- first step with `g = 1`, `lr = 0.1`, defaults moves the weight by
  approximately `0.1` (bias correction makes the first step scale-invariant)
- zero gradient leaves `w` unchanged
- AdamW with `weight_decay > 0` shrinks `w` even with zero gradient
- state is per-parameter: two different vectors update independently

---

## `optim/lr_scheduler.hpp`

**Job:** adjust `Optimizer::set_learning_rate` over time. Schedulers are pure
functions of the step number.

```cpp
namespace forgeml {

class LrScheduler {
public:
  virtual ~LrScheduler() = default;
  virtual double at(std::size_t step) const = 0;
  // Convenience: sets the optimizer's learning rate for this step.
  void apply(Optimizer &opt, std::size_t step) const { opt.set_learning_rate(at(step)); }
};

class ConstantLr : public LrScheduler {
public:
  explicit ConstantLr(double lr);
  double at(std::size_t) const override;
};

class StepLr : public LrScheduler {
public:
  StepLr(double base_lr, std::size_t step_size, double gamma);
  double at(std::size_t step) const override;      // base * gamma^(step/step_size)
};

class LinearLr : public LrScheduler {
public:
  LinearLr(double base_lr, std::size_t total_steps);
  double at(std::size_t step) const override;      // base * (1 - step/total)
};

// Warmup then cosine decay to min_lr. Used by the LLM trainer.
class CosineLr : public LrScheduler {
public:
  CosineLr(double base_lr, std::size_t warmup_steps, std::size_t total_steps,
           double min_lr = 0.0);
  double at(std::size_t step) const override;
};
}
```

Formulas:

```text
Cosine (t < warmup):  lr = base * t / warmup
Cosine (t >= warmup): progress = (t - warmup) / max(1, total - warmup)
                      lr = min_lr + 0.5 (base - min_lr)(1 + cos(pi * progress))
```

Tests:

- `StepLr` halves at the documented boundaries
- `LinearLr(1.0, 10).at(5) == 0.5`, `.at(10) == 0`
- `CosineLr` rises to `base` at `warmup`, decays to `min_lr` at `total`,
  and is monotonically non-increasing after warmup

---

## Integration with models

Classical models keep `Vector<double> w_, grad_w_;` and do:

```cpp
optimizer.step(w_, grad_w_);
```

Neural networks keep `nn::Parameter` objects and do:

```cpp
for (auto *p : network.parameters()) optimizer.step(p->value, p->grad);
```

In both cases the loop is explicit; there is no hidden global optimizer.

# nn — neural networks

One shared layer system with `forward()` and `backward()`: backprop flows
through any stack of layers. Models are just layer compositions. This is the
upgrade path from the classical models, whose gradients are hand-derived.

Depends on: `core`, `math`, `optim`, `utils`, ForgeFP.

Files: `parameter.hpp`, `layer.hpp`, `linear.hpp`, `activation.hpp`,
`network.hpp`, `conv2d.hpp`, `pooling.hpp`, `flatten.hpp`, `rnn_cell.hpp`,
`rnn.hpp`, `lstm_cell.hpp`, `lstm.hpp`, `embedding.hpp`, `attention.hpp`,
`multi_head_attention.hpp`, `positional_encoding.hpp`, `feed_forward.hpp`,
`layer_norm.hpp`, `transformer_block.hpp`, `encoder_transformer.hpp`,
`decoder_transformer.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `parameter.hpp` | `fp::Rng` (`rng.normal`, `rng.uniform`), `fp::Buffer` (flat parameter storage), `fp::fill`, `fp::map` |
| `layer.hpp` | `fp::Result`, `fp::Buffer` |
| `linear.hpp` | `fp::matmul`, `fp::transpose`, `fp::add_row_broadcast`, `fp::col_sums` (bias gradient), `fp::map_to` |
| `activation.hpp` | `fp::sigmoid`, `fp::relu`, `fp::softmax_rows`, `fp::softmax`, `fp::transform_inplace`, `fp::map` |
| `network.hpp` | `fp::for_each`, `fp::views::enumerate`, `fp::reverse`, `fp::Buffer` (per-layer caches) |
| `conv2d.hpp` | `fp::matmul`, `fp::for_each_index`, `fp::Buffer` (`im2col`/`col2im` are `detail::` helpers here — `fp::windows2d` is the non-overlapping *pooling* primitive, not convolution) |
| `pooling.hpp` | `fp::map2d`, `fp::for_each_index`, `fp::Buffer` |
| `rnn_cell.hpp` / `rnn.hpp` | `fp::matmul`, `fp::add_row_broadcast`, `fp::tanh` (`fp::ad::tanh` for scalars), `fp::col_sums` (bias gradient) |
| `lstm_cell.hpp` / `lstm.hpp` | `fp::matmul`, `fp::sigmoid`, `fp::add_row_broadcast`, `fp::hadamard`, `fp::zip_transform_inplace` |
| `embedding.hpp` | `fp::Buffer`, `fp::for_each_index` (scatter-add), `fp::Rng` |
| `attention.hpp` / `multi_head_attention.hpp` | `fp::matmul`, `fp::batched_matmul`, `fp::transpose`, `fp::softmax_rows`, `fp::scale` |
| `positional_encoding.hpp` | `fp::linspace`, `fp::map2d_indexed`, `fp::sin`/`fp::cos` (elementwise via `fp::map`) |
| `layer_norm.hpp` | `fp::row_means`, `fp::map2d`, `fp::scale`, `fp::add_row_broadcast` |
| `feed_forward.hpp` | `fp::matmul`, `fp::add_row_broadcast`, `ml::math::gelu`/`fp::relu` |
| `transformer_block.hpp` | the above, plus `fp::zip_transform_inplace` for residual adds |
| `encoder/decoder_transformer.hpp` | `fp::matmul`, `fp::softmax_rows`, `fp::Buffer`, `fp::par_for` |
| gradient checks | `fp::central_difference`, `fp::ad::derivative`, `fp::approx_equal` |

Every dense computation in this layer is an fp kernel: `fp::matmul` (i-k-j),
`fp::add_row_broadcast` (bias), `fp::softmax_rows` (attention weights),
`fp::col_sums` (bias gradients: sum the batch axis of a `(B, out)` grid),
`fp::axpy_inplace` (optimizer
updates). The nn layer owns the *structure* (what is a layer, what is cached,
what the backward pass is), not the arithmetic.

Conventions:

- Batch-first everywhere: `(batch, features)` for dense, `(batch, channels,
  height, width)` for images, `(batch, time, features)` for sequences.
- All layer math is on `Matrix<double>` / `Tensor<double>`; no scalar
  `Value` graphs in this path. (`nn/autodiff.hpp` may keep the scalar engine
  from `MIGRATION.md` as a teaching example only.)
- Every layer's `backward` returns the gradient w.r.t. its input and
  accumulates gradients into its parameters.

---

## `nn/parameter.hpp`

```cpp
namespace forgeml {

struct Parameter {
  Matrix<double> value;
  Matrix<double> grad;          // same shape as value

  Parameter() = default;
  Parameter(Matrix<double> v, Matrix<double> g)
      : value(std::move(v)), grad(std::move(g)) {}

  void zero_grad() { grad.fill(0.0); }
  std::size_t rows() const { return value.rows(); }
  std::size_t cols() const { return value.cols(); }

  // Initializers (deterministic given rng):
  static Parameter xavier(std::size_t fan_in, std::size_t fan_out, fp::Rng &rng);
  static Parameter he(std::size_t fan_in, std::size_t fan_out, fp::Rng &rng);
  static Parameter zeros(std::size_t r, std::size_t c);
};

// Bias is a (1, n) Parameter so it broadcasts over a batch.
}
```

Initializers:

```text
Xavier:  U(-a, a), a = sqrt(6 / (fan_in + fan_out))
He:      N(0, sqrt(2 / fan_in))
```

Tests: shapes; `zero_grad`; Xavier/He bounds and variance with a fixed seed.

---

## `nn/layer.hpp`

```cpp
namespace forgeml {

class Layer {
public:
  virtual ~Layer() = default;

  // forward: x -> y. May cache anything needed for backward.
  virtual Matrix<double> forward(Matrix<double> const &x) = 0;

  // backward: dL/dy -> dL/dx. Must accumulate into parameter grads.
  virtual Matrix<double> backward(Matrix<double> const &grad_out) = 0;

  // Parameters this layer owns (empty for stateless layers).
  virtual std::vector<Parameter *> parameters() { return {}; }

  virtual std::string name() const = 0;

  void zero_grad() {
    for (auto *p : parameters()) p->zero_grad();
  }
};

// A layer whose forward takes a Tensor (images, sequences).
class TensorLayer {
public:
  virtual ~TensorLayer() = default;
  virtual Tensor<double> forward(Tensor<double> const &x) = 0;
  virtual Tensor<double> backward(Tensor<double> const &grad_out) = 0;
  virtual std::vector<Parameter *> parameters() { return {}; }
  virtual std::string name() const = 0;
};
}
```

Rules:

- `forward` caches its input (or a pre-activation) as a member; calling
  `backward` before `forward` asserts.
- `backward` must not allocate parameter-gradient storage per call; accumulate
  into the existing `Parameter::grad`.
- Layers are stateful only through parameters and caches; no RNG at forward
  time.

---

## `nn/linear.hpp`

```cpp
class Linear : public Layer {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias = true,
         fp::Rng &rng = ...);   // He init for W, zeros for b

  Matrix<double> forward(Matrix<double> const &x) override;   // (B, in) -> (B, out)
  Matrix<double> backward(Matrix<double> const &grad_out) override;
  std::vector<Parameter *> parameters() override;
  std::string name() const override { return "linear"; }

private:
  Parameter W_;        // (in, out)
  Parameter b_;        // (1, out)
  Matrix<double> x_cache_;   // (B, in)
  Matrix<double> w_grad_cache_;  // dL/dW for this batch
};
```

Forward / backward:

```text
y = x W + b                       (b broadcasts over the batch)

dL/dW = x^T (dL/dy)
dL/db = col_sums(dL/dy)           # sum the batch axis
dL/dx = (dL/dy) W^T
```

- `W` is `(in, out)` so `x W` is a single `matmul`; do not store the transpose.
- `backward` accumulates `dL/dW` and `dL/db` into `W_.grad` / `b_.grad` and
  returns `dL/dx`.
- Bias broadcast on backward sums over the batch dimension.

ForgeFP implementation:

```cpp
// forward
auto y = fp::add_row_broadcast(fp::matmul(x, W_.value), b_.value[0]);

// backward (grad_out = dL/dy)
W_.grad = fp::matmul(fp::transpose(x_cache_), grad_out);   // accumulate
auto db = fp::col_sums(grad_out);                  // sum over the batch axis
return fp::matmul(grad_out, fp::transpose(W_.value));
```

`fp::matmul`'s `i-k-j` order and `fp::add_row_broadcast`'s single pass are the
whole implementation; there is no hand-rolled triple loop.

Tests: hand-computed forward; `check_gradient`; bias=false path.

---

## `nn/activation.hpp`

```cpp
class ReLU : public Layer { ... };        // elementwise
class Sigmoid : public Layer { ... };
class Tanh : public Layer { ... };
class Softmax : public Layer { ... };     // row-wise, for inference
```

Backward:

```text
ReLU:    dx = dy * (x > 0)
Sigmoid: dx = dy * y * (1 - y)             (cache y = sigmoid(x))
Tanh:    dx = dy * (1 - y^2)
Softmax: dx_i = y_i (dy_i - sum_j dy_j y_j)   (full Jacobian-vector product)
```

- Softmax in training is almost always fused with cross-entropy
  (`math/losses.hpp` `categorical_cross_entropy` takes logits). Keep the
  standalone `Softmax` layer for inference and for teaching.
- All activations cache their output (not input) where the derivative needs it.
- The forward pass is fp: `fp::sigmoid`, `fp::relu`, and `fp::softmax_rows`
  (in place over the logits grid). The backward pass uses
  `fp::zip_transform_inplace` for the elementwise cases and the
  softmax Jacobian-vector product for `Softmax`.

Tests: elementwise values; `check_gradient`; softmax rows sum to 1.

---

## `nn/network.hpp`

**Job:** a sequential container: forward in order, backward in reverse order.

```cpp
class Network {
public:
  void add(std::unique_ptr<Layer> layer);

  Matrix<double> forward(Matrix<double> const &x);       // caches per-layer I/O
  void backward(Matrix<double> const &grad_loss);        // reverse order
  std::vector<Parameter *> parameters();
  void zero_grad();

  // Training helpers
  double train_batch(Matrix<double> const &x, std::vector<int> const &y,
                     Optimizer &optimizer, LossFn loss);
  double evaluate(Matrix<double> const &x, std::vector<int> const &y) const;

  std::vector<std::string> summary() const;
};

// LossFn: (logits, targets) -> {value, grad_wrt_logits}
using LossFn = std::function<LossResult(Matrix<double> const &,
                                        std::vector<int> const &)>;
```

Algorithm:

```text
forward:  x0 = input; xi = layer_i.forward(x_{i-1}); return x_last
loss:     {value, grad} = loss_fn(x_last, y)
backward: g = grad
          for layer in reverse(layers): g = layer.backward(g)
optimizer: for p in parameters(): optimizer.step(p->value, p->grad)
```

Rules:

- `forward` stores each layer's input/output so `backward` needs no arguments
  besides the loss gradient (memory grows with depth; document it).
- `backward` does not update parameters; the caller runs the optimizer.
- `Network` owns its layers (`std::vector<std::unique_ptr<Layer>>`).
- `summary()` returns one line per layer with name, shapes, parameter count.

Tests (`test/nn_core_test.cpp`):

- XOR: `Linear(2,4) -> ReLU -> Linear(4,1)` (or 2 outputs + softmax) reaches
  100% with a fixed seed
- `check_network` passes for the whole network
- parameter count matches the sum of layer sizes
- backward before forward asserts

---

## `nn/conv2d.hpp`

**Job:** 2-D convolution over `(batch, channels, H, W)`.

```cpp
struct Conv2DConfig {
  std::size_t in_channels = 1, out_channels = 1;
  std::size_t kernel_h = 3, kernel_w = 3;
  std::size_t stride = 1, padding = 0;
  bool bias = true;
};

class Conv2D : public TensorLayer {
public:
  Conv2D(Conv2DConfig config, fp::Rng &rng);
  Tensor<double> forward(Tensor<double> const &x) override;
  Tensor<double> backward(Tensor<double> const &grad_out) override;
  std::vector<Parameter *> parameters() override;
private:
  Conv2DConfig config_;
  Parameter kernels_;   // (out_ch, in_ch * kh * kw)
  Parameter bias_;      // (1, out_ch)
  Tensor<double> x_cache_;
};
}
```

Output shape:

```text
H_out = (H + 2*padding - kernel_h) / stride + 1
W_out = (W + 2*padding - kernel_w) / stride + 1
```

Forward (im2col formulation; the only sane way to implement backward):

```text
patches = im2col(x)        # (batch * H_out * W_out, in_ch * kh * kw)
y_flat  = patches * kernels_^T + bias
y       = col2im(y_flat)   # (batch, out_ch, H_out, W_out)
```

Backward:

```text
dL/dkernels = patches^T * dL/dy_flat
dL/dbias    = sum over batch, H_out, W_out of dL/dy
dL/dx       = col2im(dL/dy_flat * kernels_)    # scatter-add overlaps
```

Rules:

- `im2col`/`col2im` are `detail::` helpers in the same header; `col2im` must
  *accumulate* (overlapping windows).
- Store the cached patches, not the input, to make backward cheap.
- Validate divisibility: if `(H + 2p - kh) % stride != 0`, fail in the
  constructor factory or assert in debug; document the choice.

Tests (`test/conv_test.cpp`):

- 1x1x3x3 input with a known 1x1x2x2 kernel produces the hand-computed map
- output shape matches the formula for stride/padding variants
- `check_gradient` for kernels, bias, and input
- a 2-conv-layer CNN classifies a tiny synthetic image task > 95%

---

## `nn/pooling.hpp`

```cpp
class MaxPool2D : public TensorLayer {
public:
  MaxPool2D(std::size_t size, std::size_t stride);
  Tensor<double> forward(Tensor<double> const &x) override;
  Tensor<double> backward(Tensor<double> const &grad_out) override;
private:
  std::vector<std::size_t> argmax_cache_;   // flat index of the max per window
};

class AvgPool2D : public TensorLayer { ... };
```

Backward:

```text
MaxPool: route grad_out to the cached argmax position, zeros elsewhere
AvgPool: distribute grad_out / (size*size) over each window
```

Tests: shape; max routes only to the max; average distributes evenly;
gradient checks.

---

## `nn/flatten.hpp`

```cpp
class Flatten : public Layer {
public:
  // (B, C, H, W) -> (B, C*H*W); remembers the input shape for backward.
  Matrix<double> forward(Matrix<double> const &x) override;   // actually Tensor in
  Matrix<double> backward(Matrix<double> const &grad_out) override;
};
```

Implementation detail: provide a `Tensor` overload pair so CNNs can feed it;
store `(C,H,W)` for the backward reshape. Tests: shape and round-trip.

---

## `nn/rnn_cell.hpp` and `nn/rnn.hpp`

```cpp
struct RnnConfig {
  std::size_t input_size = 0, hidden_size = 0;
  bool bias = true;
};

class RnnCell : public Layer {
public:
  RnnCell(RnnConfig config, fp::Rng &rng);
  // x: (B, input_size), h_prev: (B, hidden_size) -> h_next: (B, hidden_size)
  Matrix<double> forward(Matrix<double> const &x, Matrix<double> const &h_prev);
  Matrix<double> backward(Matrix<double> const &grad_h,
                          Matrix<double> &grad_h_prev_out);
  std::vector<Parameter *> parameters() override;
private:
  Parameter Wx_, Wh_, b_;   // (in, hidden), (hidden, hidden), (1, hidden)
  Matrix<double> x_cache_, h_prev_cache_;
};

class Rnn : public TensorLayer {
public:
  // x: (B, T, input_size) -> outputs: (B, T, hidden_size)
  Tensor<double> forward(Tensor<double> const &x) override;
  Tensor<double> backward(Tensor<double> const &grad_out) override;
  std::vector<Parameter *> parameters() override;
};
```

Forward:

```text
h_t = tanh(x_t Wx + h_{t-1} Wh + b)
```

Backward (backprop through time):

```text
dh = dL/dh_t + dh_next
dz = dh * (1 - h_t^2)
dWx += x_t^T dz ; dWh += h_{t-1}^T dz ; db += col_sums(dz)   # batch axis
dx_t   = dz Wx^T
dh_prev = dz Wh^T
```

Rules:

- `Rnn` unrolls the cell over `T` and stores per-step caches; `backward`
  iterates `t = T-1 .. 0` and sums `dh_next` into the current step.
- The hidden state is initialized to zeros (documented); a `stateful` option is
  future work.
- Truncated BPTT (limit `T` per batch) is a caller concern: the dataloader
  yields fixed-length sequences.

Tests: forward matches hand-computed one-step cell; BPTT gradient check;
a parity/delayed-copy toy task is learnable.

---

## `nn/lstm_cell.hpp` and `nn/lstm.hpp`

```cpp
class LstmCell : public Layer {
public:
  LstmCell(std::size_t input_size, std::size_t hidden_size, fp::Rng &rng);
  // returns (h_t, c_t) and caches all gates
  std::pair<Matrix<double>, Matrix<double>>
  forward(Matrix<double> const &x, Matrix<double> const &h_prev,
          Matrix<double> const &c_prev);
  Matrix<double> backward(Matrix<double> const &grad_h,
                          Matrix<double> const &grad_c,
                          Matrix<double> &grad_h_prev_out,
                          Matrix<double> &grad_c_prev_out);
  std::vector<Parameter *> parameters() override;
};
```

Forward (four gates, each `(B, H)`):

```text
f_t = sigmoid(x_t Wf + h_{t-1} Uf + bf)
i_t = sigmoid(x_t Wi + h_{t-1} Ui + bi)
o_t = sigmoid(x_t Wo + h_{t-1} Uo + bo)
g_t = tanh   (x_t Wg + h_{t-1} Ug + bg)
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

Backward (given `dh`, `dc`):

```text
do = dh * tanh(c_t)          dtanh_c = dh * o_t
dc += dtanh_c * (1 - tanh(c_t)^2)
df = dc * c_{t-1}            dc_prev = dc * f_t
di = dc * g_t                dg = dc * i_t * (1 - g_t^2)
d(bf,bi,bo,bg) = col_sums(...)   # sum the batch axis of each gate pre-activation
dW*, dU* = x_t^T dpre, h_{t-1}^T dpre
dx_t = sum_gate dpre * W^T ; dh_prev = sum_gate dpre * U^T
```

Rules:

- Cache the four gate pre-activations (or their sigmoid/tanh outputs) to make
  backward cheap.
- Gate weights are grouped in one `Parameter` per input/hidden matrix or four
  separate ones; pick grouped (`(in, 4H)`, `(H, 4H)`) and document it, then
  slice in backward. Grouped is one `matmul` per direction.
- `Lstm` unrolls exactly like `Rnn`; gradient clipping is applied by the
  trainer, not the cell.

Tests: hand-computed one-step forward; BPTT gradient check; learn a
delayed-copy task.

---

## `nn/embedding.hpp`

```cpp
class Embedding : public Layer {
public:
  Embedding(std::size_t vocab_size, std::size_t dim, fp::Rng &rng);
  // token ids: (B, T) -> vectors: (B, T, dim) flattened to (B*T, dim)
  Matrix<double> forward(std::vector<int> const &tokens);
  // backward scatter-adds into the table
  void backward(std::vector<int> const &tokens, Matrix<double> const &grad_out);
  std::vector<Parameter *> parameters() override;
private:
  Parameter table_;    // (vocab_size, dim)
};
```

Rules:

- Forward is a row lookup; backward is scatter-add:
  `table_.grad.row(id) += grad_out.row(position)`.
- Repeated tokens in a batch accumulate; do not overwrite.
- Out-of-range ids return `fp::fail` (checked in forward).

Tests: lookup values; scatter-add with a repeated id; gradient check.

---

## `nn/attention.hpp` and `nn/multi_head_attention.hpp`

```cpp
// Single-head scaled dot-product attention.
// q: (B, Tq, d), k: (B, Tk, d), v: (B, Tk, d) -> (B, Tq, d)
class Attention {
public:
  Tensor<double> forward(Tensor<double> const &q, Tensor<double> const &k,
                         Tensor<double> const &v, bool causal = false);
  // backward returns {dq, dk, dv}
  std::tuple<Tensor<double>, Tensor<double>, Tensor<double>>
  backward(Tensor<double> const &grad_out);
private:
  Tensor<double> q_cache_, k_cache_, v_cache_, weights_cache_;
};

struct MultiHeadAttentionConfig {
  std::size_t d_model = 0, num_heads = 0;   // d_model % num_heads == 0
  bool causal = false;
};

class MultiHeadAttention : public TensorLayer {
public:
  MultiHeadAttention(MultiHeadAttentionConfig config, fp::Rng &rng);
  Tensor<double> forward(Tensor<double> const &x) override;   // self-attention
  Tensor<double> backward(Tensor<double> const &grad_out) override;
  std::vector<Parameter *> parameters() override;   // Wq, Wk, Wv, Wo
};
```

Math:

```text
scores = Q K^T / sqrt(d_k)                 # (B, heads, Tq, Tk)
if causal: scores[i][j] = -inf for j > i   # before softmax
A = softmax(scores)                        # over the last axis
context = A V
MHA(x) = concat(heads) W_o
```

Backward (per head):

```text
dV = A^T dContext
dA = dContext V^T
dS = A * (dA - sum(dA * A, axis=-1, keepdims))   # softmax Jacobian
dS /= sqrt(d_k)
dQ = dS K ; dK = dS^T Q
```

Rules:

- Causal masking sets future scores to a large negative number (`-1e9`) before
  softmax, never zero after softmax.
- `d_model % num_heads == 0`; split/merge heads by reshape + transpose, not by
  copying per head.
- Cache `A` for backward; do not recompute softmax.

ForgeFP implementation — **per batch (and per head) on 2-D matrices**: fp's
linalg takes grids (`std::vector<std::vector<T>>`), and `fp::transpose`,
`fp::softmax_rows` and `fp::map2d_*` are 2-D. `fp::batched_matmul` is 3-D
`(B, m, k)·(B, k, n)`, so it cannot combine with a 2-D `fp::transpose`; split
the head axis with reshape + transpose first, then loop with `fp::par_for` over
the batch when the sequence is long:

```cpp
// per batch b: Q_b (Tq, d), K_b (Tk, d), V_b (Tk, d)
auto scores = fp::matmul(Q_b, fp::transpose(K_b));            // (Tq, Tk)

// Scale + causal mask in one allocation-free in-place pass.
for (auto &&[i, row] : fp::views::enumerate(scores))
  fp::for_each_index(row, [&](std::size_t j, double &s) {
    s = (causal && j > i) ? -1e9 : s / std::sqrt(static_cast<double>(d_k));
  });

fp::softmax_rows(scores);                                     // in place
auto context = fp::matmul(scores, V_b);                       // (Tq, d)
```

(`fp::map2d_indexed(scores, f)` is the allocating alternative when the mask is
built into a fresh grid.)

The backward pass mirrors it per batch: `dV = fp::matmul(fp::transpose(A), dContext)`,
`dA = fp::matmul(dContext, fp::transpose(V))`, the softmax Jacobian scaling
with `fp::inplace`, then `dQ = fp::matmul(dS, K)` and
`dK = fp::matmul(fp::transpose(dS), Q)`. `fp::par_for` splits the batch (or the
batch × head pairs) when the sequence is long.

Tests (`test/attention_test.cpp`):

- attention weights sum to 1 over the key axis
- causal mask: output position `t` is unchanged when future inputs change
- gradient check for Q/K/V and for the four projections
- multi-head with `num_heads = 1` equals single-head attention

---

## `nn/positional_encoding.hpp`

```cpp
// Adds a fixed sinusoidal encoding to (B, T, d_model).
class SinusoidalPositionalEncoding {
public:
  explicit SinusoidalPositionalEncoding(std::size_t max_len, std::size_t d_model);
  Tensor<double> forward(Tensor<double> const &x) const;
private:
  Matrix<double> table_;   // (max_len, d_model)
};
```

Formula:

```text
PE[t][2i]   = sin(t / 10000^(2i/d_model))
PE[t][2i+1] = cos(t / 10000^(2i/d_model))
```

Rules: precompute the table in the constructor; forward is an add; no
parameters and no backward (the gradient passes through unchanged).

Tests: table values at `t = 0`; shape; adding twice is not idempotent (sanity).

---

## `nn/feed_forward.hpp`

```cpp
struct FeedForwardConfig {
  std::size_t d_model = 0, d_ff = 0;
  enum class Activation { Relu, Gelu } activation = Activation::Gelu;
  double dropout = 0.0;    // 0 = disabled
};

class FeedForward : public Layer {
public:
  // Linear(d_model, d_ff) -> activation -> Linear(d_ff, d_model)
};
```

Rules: two `Linear` layers + activation; dropout, if enabled, must use the
same mask in backward (cache it) and is disabled at eval.

Tests: shape; gradient check; dropout in eval mode is identity.

---

## `nn/layer_norm.hpp`

```cpp
class LayerNorm : public Layer {
public:
  LayerNorm(std::size_t d_model, double eps = 1e-5);
  Matrix<double> forward(Matrix<double> const &x) override;   // normalizes each row
  Matrix<double> backward(Matrix<double> const &grad_out) override;
  std::vector<Parameter *> parameters() override;   // gamma, beta (1, d_model)
private:
  double eps_;
  Matrix<double> x_cache_, xhat_cache_, inv_std_cache_;
};
```

Math (per row):

```text
mu    = mean(x)
var   = variance(x, ddof=0)
xhat  = (x - mu) / sqrt(var + eps)
y     = gamma * xhat + beta

dxhat = dy * gamma
dx = (1 / (d * sqrt(var+eps))) * (d * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))
dgamma = col_sums(dy * xhat)      # sum the batch axis
dbeta  = col_sums(dy)             # sum the batch axis
```

Rules: cache `xhat` and `inv_std`; `eps` outside the sqrt; gamma initialized to
1, beta to 0.

Naming: `sum(...)` inside the `dx` formula reduces the **feature** axis of a
`(B, d)` row, i.e. `fp::row_sums`. The parameter gradients (`dgamma`, `dbeta`,
and every bias above) reduce the **batch** axis, i.e. `fp::col_sums`. The two
are easy to transpose by accident — the gradient check is what catches it.

Tests: output rows have mean ~0 and variance ~1; gradient check; gamma/beta
gradients.

---

## `nn/transformer_block.hpp`

```cpp
struct TransformerBlockConfig {
  std::size_t d_model = 0, num_heads = 0, d_ff = 0;
  bool causal = false;
  double dropout = 0.0;
};

class TransformerBlock : public TensorLayer {
public:
  TransformerBlock(TransformerBlockConfig config, fp::Rng &rng);
  Tensor<double> forward(Tensor<double> const &x) override;   // pre-norm
  Tensor<double> backward(Tensor<double> const &grad_out) override;
  std::vector<Parameter *> parameters() override;
};
```

Pre-norm layout (stable for deep stacks):

```text
x = x + MHA(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

- Residual connections pass the gradient straight through; the backward order
  is the reverse of the forward order.
- Causal flag is forwarded to `MultiHeadAttention`.

Tests: shape preservation; causal property; gradient check; a 2-block stack
overfits a tiny sequence.

---

## `nn/encoder_transformer.hpp` and `nn/decoder_transformer.hpp`

```cpp
class EncoderTransformer : public TensorLayer {
public:
  // embedding + positional encoding + N pre-norm blocks (bidirectional)
  EncoderTransformer(std::size_t vocab, std::size_t d_model, std::size_t heads,
                     std::size_t d_ff, std::size_t layers, std::size_t max_len,
                     fp::Rng &rng);
};

class DecoderTransformer : public TensorLayer {
public:
  // embedding + positional encoding + N causal blocks + final LayerNorm + LM head
  // weight tying: lm_head weight == embedding table (optional flag)
};
```

Rules:

- Encoder uses non-causal attention; decoder uses causal attention.
- The decoder returns logits `(B, T, vocab)`; the loss is
  `categorical_cross_entropy` over the shifted targets (see `llm.md`).
- The stack is a `std::vector<std::unique_ptr<TransformerBlock>>`; forward and
  backward iterate it (backward in reverse).

Tests: shapes; parameter count formula; overfit a copy task; gradient check on
a 1-block model.

---

## Gradient checking every layer

`utils/gradient_check.hpp` must pass for: `Linear`, all activations, `Conv2D`,
pooling, `RnnCell`/`Rnn`, `LstmCell`/`Lstm`, `Embedding`, `Attention`, `MHA`,
`FeedForward`, `LayerNorm`, `TransformerBlock`, and the decoder head. A layer
without a passing gradient check is not done.

The harness is fp-based: perturb one parameter at a time and compare against
`fp::central_difference` (or `fp::ad::derivative` for scalar paths), with
`fp::approx_equal` deciding equality.

## Numeric stability checklist

- Softmax/log-softmax: subtract the max — `fp::softmax`/`fp::log_softmax` do
  this; never write `exp(x)/sum(exp(x))` by hand.
- Cross-entropy: from logits via `fp::log_softmax`, not from probabilities.
- LayerNorm: `eps` inside the denominator.
- LSTM: sigmoid/tanh saturate; clip gradients in the trainer (`llm.md`) with
  the global-norm helper.
- Attention: scale by `1/sqrt(d_k)`; mask with `-1e9`, not `-inf` (keeps NaNs
  out of the softmax subtraction).
- Guard non-finite losses with `fp::is_finite` and repair with
  `fp::nan_to_num` rather than letting one `nan` poison the run.

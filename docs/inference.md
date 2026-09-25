# inference — running a trained LLM

Training is Stage 14; this is what happens after: caching attention state,
choosing tokens, shrinking the weights, batching requests and adapting the
model with LoRA. The rule from the rest of the stack holds — every
optimization is validated against the plain implementation, not trusted.

Depends on: `core`, `nn`, `llm`, `prob`, ForgeFP.

Files: `kv_cache.hpp`, `sampling.hpp`, `quantize.hpp`, `batch.hpp`,
`finetune.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `kv_cache.hpp` | `fp::Buffer`, `fp::linalg` (`matmul`), `fp::inplace` |
| `sampling.hpp` | `fp::numerics` (`softmax`, `logsumexp`), `fp::sort_by_cached`, `fp::Rng` |
| `quantize.hpp` | `fp::Buffer`, `fp::linalg`, `fp::inplace` |
| `batch.hpp` | `fp::views::chunk`, `fp::Rng`, `fp::sort_by_cached` |
| `finetune.hpp` | `fp::linalg::matmul`, `fp::Rng`, `fp::inplace` |

---

## `inference/kv_cache.hpp`

**Job:** keep the key/value projections of past tokens so each step is O(1) in
sequence length instead of O(n).

```cpp
namespace forgeml {

struct KvCache {
  std::size_t layers, heads, head_dim, max_seq;
  std::vector<Matrix<double>> key, value;   // (heads * max_seq, head_dim) per layer
  std::size_t length = 0;                   // tokens stored
};

KvCache make_cache(std::size_t layers, std::size_t heads, std::size_t head_dim, std::size_t max_seq);
fp::Result<void> append(KvCache &cache, std::size_t layer, Matrix<double> const &k, Matrix<double> const &v);
void evict_oldest(KvCache &cache, std::size_t n);         // sliding window
std::size_t memory_bytes(KvCache const &cache);
}
```

Rules: the cache is pre-allocated (`max_seq` tokens) so decoding never
allocates; `append` is O(1) and errors when full (eviction is explicit);
cached attention must be **bit-identical** to recomputing attention over the
full prefix — that equality is the correctness contract, and the test asserts
it; a sliding window is the documented way to bound memory.

Tests: cached decoding equals full recomputation on a fixed sequence;
`memory_bytes` matches `2 * layers * heads * max_seq * head_dim * 8`;
eviction keeps the most recent tokens; appending past `max_seq` errors.

## `inference/sampling.hpp`

**Job:** how the next token is chosen — and why the choice matters more than
the temperature.

```cpp
struct SamplingConfig {
  double temperature = 1.0;
  std::size_t top_k = 0;        // 0 = off
  double top_p = 1.0;           // 1 = off
  double typical_p = 1.0;
  double repetition_penalty = 1.0;
  std::uint64_t seed = 0;
};

std::size_t sample(Vector<double> const &logits, SamplingConfig const &config,
                   Vector<std::size_t> const &history, fp::Rng &rng);
std::vector<std::size_t> beam_search(std::function<Vector<double>(std::vector<std::size_t> const &)> const &logits_fn,
                                     std::size_t beam, std::size_t length);
std::size_t greedy(Vector<double> const &logits);
```

Rules: temperature is applied to logits **before** masking; top-k/top-p/
typical mask by setting logits to `-inf` and sampling via softmax +
`logsumexp` (never `log(sum(exp))`); the repetition penalty is applied to the
tokens in `history`; `beam_search` scores with length-normalized
log-probability and breaks ties deterministically.

Tests: top-k=1 equals greedy; top-p=0.9 keeps the smallest set whose mass
exceeds 0.9 (hand example); temperature 0 is an error (use `greedy`);
repetition penalty suppresses a repeated token; beam search beats greedy on a
planted sequence where the greedy first token is a trap.

## `inference/quantize.hpp`

**Job:** int8/int4 weights with group-wise scales, and the dequantized
operations the model needs.

```cpp
struct Quantized {
  std::vector<std::int8_t> data;     // or packed int4
  Vector<double> scale, zero_point;  // per group
  std::size_t group = 64, bits = 8;
  std::vector<std::size_t> shape;
};

fp::Result<Quantized> quantize(Matrix<double> const &W, std::size_t bits = 8, std::size_t group = 64);
Matrix<double> dequantize(Quantized const &q);
fp::Result<Matrix<double>> matmul(Quantized const &w, Matrix<double> const &x);   // dequant on the fly
fp::Result<Vector<double>> embedding_lookup(Quantized const &table, std::size_t index);
double max_abs_error(Matrix<double> const &a, Matrix<double> const &b);
```

Rules: symmetric per-group quantization (zero point only for asymmetric
int4); groups never cross row boundaries; `matmul` dequantizes group by group
so the full matrix is never materialized; the quantization error bound is
`scale/2` per group and a test checks it.

Tests: dequantize(quantize(W)) stays within `scale/2` per element; int8 error
< int4 error on the same matrix; `matmul` matches the fp32 result within the
documented tolerance; embedding lookup matches the dequantized row.

## `inference/batch.hpp`

**Job:** serve several requests at once without padding every one to the
longest sequence.

```cpp
struct Request { std::size_t id; std::vector<std::size_t> prompt; std::size_t max_tokens; SamplingConfig config; };
struct Response { std::size_t id; std::vector<std::size_t> tokens; bool finished; };

struct BatchQueue {
  std::size_t max_batch_tokens = 4096;
  std::vector<Request> waiting;
  std::vector<Request> running;
};

fp::Result<std::vector<Response>> step(BatchQueue &queue, std::function<Vector<double>(std::vector<Request> const &)> const &forward);
std::vector<Response> drain(BatchQueue &queue);
```

Rules: each running request owns a slice of the KV cache; a request is
admitted when its prompt fits the remaining budget; a finished request frees
its slice immediately (continuous batching); responses come out in request
order regardless of completion order, so a test can be deterministic.

Tests: two requests of different lengths finish with the same tokens as
running them alone; the batch never exceeds `max_batch_tokens`; a finished
request frees its cache slice (memory returns to the pool).

## `inference/finetune.hpp`

**Job:** LoRA adapters — train a small low-rank update instead of the weights,
then merge it for inference.

```cpp
struct Lora { std::size_t rank; double alpha; Matrix<double> A, B; };   // B*A is the update

Lora make_lora(std::size_t in_features, std::size_t out_features, std::size_t rank,
               double alpha, fp::Rng &rng);
Matrix<double> apply(Lora const &lora, Matrix<double> const &W, Matrix<double> const &x);
fp::Result<Lora> train_step(Lora &lora, Matrix<double> const &W, Matrix<double> const &x,
                            Matrix<double> const &grad_out, double lr);
Matrix<double> merge(Lora const &lora, Matrix<double> const &W);        // W + (alpha/r) * B*A
```

Rules: `B` starts at zero (so the adapted model starts identical to the base
model — a test asserts this); only `A`/`B` receive gradients; `merge` is exact
and must produce the same forward pass as `apply` (within tolerance).

Tests: zero-init means the adapted forward equals the base forward exactly;
one training step reduces the loss; `merge` matches `apply` on random input;
adapter parameter count is `r * (in + out)`.

## Gate

Cached decoding is bit-identical to full recomputation and faster; sampling
masks behave per hand-computed examples; int8 stays within tolerance of fp32
and beats int4's error; continuous batching matches single-request outputs;
LoRA's zero-init identity and merge equality hold.

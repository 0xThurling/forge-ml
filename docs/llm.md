# llm — a tiny decoder-only language model

The top of the stack: a character/subword GPT built from `nn/`. The point is a
complete, trainable, samplable language model, not scale. Everything here is
CPU-sized on purpose.

Depends on: `core`, `math`, `data`, `optim`, `nn`, `utils`, ForgeFP.

Files: `tokenizer.hpp`, `dataset.hpp`, `causal_lm.hpp`, `trainer.hpp`,
`sampling.hpp`, `checkpoint.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `tokenizer.hpp` | `fp::str::split_any`, `fp::str::join`, `fp::sort`, `fp::unique`, `fp::group_by`, `fp::for_each`, `fp::Result` |
| `dataset.hpp` | `fp::Rng`, `fp::Buffer` (token storage), `fp::views::chunk`, `fp::map_to` |
| `causal_lm.hpp` | `fp::matmul`, `fp::softmax_rows`, `fp::Buffer`, `fp::par_for` |
| `trainer.hpp` | `fp::Stopwatch`, `fp::log_softmax`, `fp::is_finite`, `fp::for_each`, `fp::par_for`, `fp::axpy_inplace` |
| `sampling.hpp` | `fp::softmax`, `fp::logsumexp`, `fp::sort_by`, `fp::scan`, `fp::argmax`, `fp::Rng::categorical` |
| `checkpoint.hpp` | `fp::to_text`/`from_text`, `fp::to_bytes`/`from_bytes`, `fp::write_bytes`, `fp::read_bytes`, `fp::ensure_directory` |

The tokenizer is string work (`fp::str`), sampling is fp numerics plus
`fp::Rng`, checkpoints are `fp::serialize`/`fp::io`, and the trainer's timing is
`fp::Stopwatch`. Only the LM semantics live here.

---

## `llm/tokenizer.hpp`

**Job:** turn text into integer ids and back. Start char-level (zero unknowns),
add byte-pair merges once the pipeline works.

```cpp
namespace forgeml {

class CharTokenizer {
public:
  // Builds the vocabulary from the corpus: sorted unique bytes/chars.
  explicit CharTokenizer(std::string_view corpus);

  std::vector<int> encode(std::string_view text) const;
  std::string decode(std::vector<int> const &ids) const;

  std::size_t vocab_size() const;
  int unknown_id() const;                 // optional; char-level has none

  fp::Result<void> save(std::string const &path) const;
  static fp::Result<CharTokenizer> load(std::string const &path);

private:
  std::vector<char> id_to_char_;
  std::unordered_map<char, int> char_to_id_;
};

// Byte-level BPE: merges the most frequent adjacent pair repeatedly.
class BpeTokenizer {
public:
  struct Config {
    std::size_t vocab_size = 512;
    std::size_t min_frequency = 2;
  };

  static BpeTokenizer train(std::string_view corpus, Config config);

  std::vector<int> encode(std::string_view text) const;
  std::string decode(std::vector<int> const &ids) const;
  std::size_t vocab_size() const;

  fp::Result<void> save(std::string const &path) const;
  static fp::Result<BpeTokenizer> load(std::string const &path);

private:
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::pair<int, int>> merges_;   // ordered merge rules
};
}
```

Rules:

- Char-level first: `encode(decode(ids)) == ids` for any id sequence.
- BPE training: start from bytes, count pairs with an `unordered_map`, merge
  the most frequent pair (ties -> lexicographically smallest), record the merge,
  repeat until `vocab_size`. Deterministic.
- `decode` is lossless: unknown ids return `fp::fail`.
- Special tokens (`<pad>`, `<eos>`) are reserved ids `0` and `1` in the BPE
  variant; document them and never merge across them.

Tests (`test/tokenizer_test.cpp`):

- char round-trip on ASCII and a UTF-8 string (byte-level)
- BPE with `vocab_size = 300` on a repeated corpus produces fewer tokens than
  char-level for the same text
- BPE round-trip is exact
- save/load produces identical encodings

---

## `llm/dataset.hpp`

**Job:** a token stream plus sampling of fixed-length `(input, target)` blocks
for next-token prediction.

```cpp
namespace forgeml {

struct LmBatch {
  std::vector<int> inputs;    // (B * T)
  std::vector<int> targets;   // (B * T), shifted by one
  std::size_t batch = 0, block = 0;
};

class LmDataset {
public:
  LmDataset(std::vector<int> tokens, std::size_t block_size);

  std::size_t num_tokens() const;
  std::size_t block_size() const;

  // One epoch of random blocks (reproducible via rng).
  std::vector<LmBatch> epoch(std::size_t batch_size, fp::Rng &rng) const;

  // Deterministic, non-overlapping evaluation split: the last val_fraction.
  static std::pair<LmDataset, LmDataset>
  split(std::vector<int> const &tokens, std::size_t block_size,
        double val_fraction);
};
}
```

Rules:

- `targets[i] = tokens[start + i + 1]`; the last token of the corpus cannot
  start a block (`start + block_size < num_tokens`).
- A batch concatenates `B` blocks of length `T`; the model reshapes to
  `(B, T)`.
- Blocks may overlap across batches (standard for language modeling); document
  it.
- `split` never shuffles the token order; the validation split is the tail.

Tests:

- target shift is exactly one
- block count and shapes; no block runs past the end
- validation split sizes

---

## `llm/causal_lm.hpp`

**Job:** assemble the decoder-only model: embedding, positional encoding,
N causal transformer blocks, final LayerNorm, LM head.

```cpp
namespace forgeml {

struct CausalLmConfig {
  std::size_t vocab_size = 0;
  std::size_t d_model = 128;
  std::size_t num_heads = 4;
  std::size_t d_ff = 512;
  std::size_t num_layers = 4;
  std::size_t max_seq_len = 128;
  double dropout = 0.0;
  bool tie_weights = true;      // LM head shares the embedding table
};

class CausalLm : public TensorLayer {
public:
  CausalLm(CausalLmConfig config, fp::Rng &rng);

  // tokens: (B, T) -> logits: (B, T, vocab)
  Tensor<double> forward(std::vector<int> const &tokens) override;
  Tensor<double> backward(Tensor<double> const &grad_logits) override;

  std::vector<Parameter *> parameters() override;
  std::size_t num_parameters() const;
  CausalLmConfig const &config() const;
};
}
```

Data flow:

```text
ids (B, T)
  -> Embedding            (B, T, d_model)
  -> + positional encoding
  -> N x TransformerBlock (causal)
  -> LayerNorm
  -> LM head Linear(d_model, vocab)       (tied to embedding if configured)
  -> logits (B, T, vocab)
```

Training objective:

```text
L = (1/(B*T)) sum_{b,t} -log softmax(logits[b, t])[targets[b, t]]
dL/dlogits = (softmax(logits) - onehot(target)) / (B*T)
```

Rules:

- Causal masking is mandatory in every block; a unit test must prove position
  `t` cannot see `t+1` (perturb a future token, output at `t` unchanged).
- Weight tying: if `tie_weights`, the LM head *is* the embedding matrix; its
  gradient is the sum of the head gradient and the embedding gradient.
- `max_seq_len` bounds the positional table; longer inputs fail.
- Parameter count helper: `vocab*d + T*d + layers*(...)`.
- ForgeFP implementation: the whole model is `nn/` layers, so the forward pass
  is fp kernels (`fp::matmul`, `fp::softmax_rows`, `fp::add_row_broadcast`)
  and the loss is `fp::log_softmax`. The LM head is a `Linear`; tying assigns
  the same `Matrix` to both and sums the two gradients with
  `fp::zip_transform_inplace`.

Tests:

- shapes for a 2-layer model
- causal property
- parameter count matches the formula
- overfit a 32-token copy task to near-zero loss (fixed seed)
- `check_gradient` on a 1-layer model with a small vocab

---

## `llm/trainer.hpp`

**Job:** the training loop: batches, loss, backward, AdamW, gradient clipping,
LR schedule, evaluation, logging, checkpointing.

```cpp
namespace forgeml {

struct LmTrainerConfig {
  std::size_t max_steps = 1000;
  std::size_t batch_size = 16;
  std::size_t eval_interval = 50;
  std::size_t eval_batches = 8;
  std::size_t log_interval = 10;
  double grad_clip_norm = 1.0;       // 0 disables
  double weight_decay = 0.01;
  double learning_rate = 3e-4;
  std::size_t warmup_steps = 50;
  double min_lr = 1e-5;
  std::size_t checkpoint_interval = 250;
  std::string checkpoint_dir = "checkpoints";
};

struct TrainStats {
  std::size_t step = 0;
  double train_loss = 0.0;
  double val_loss = 0.0;
  double learning_rate = 0.0;
  double grad_norm = 0.0;
  double seconds = 0.0;
};

class LmTrainer {
public:
  LmTrainer(CausalLm *model, LmDataset train, LmDataset val,
            LmTrainerConfig config, Logger &logger);

  fp::Result<void> train(fp::Rng &rng, EpochCallback callback = {});
  std::vector<TrainStats> const &history() const;

  // Single-step helpers (also used by tests)
  double train_step(LmBatch const &batch);
  double evaluate(std::size_t batches, fp::Rng &rng);

private:
  AdamW optimizer_;
  CosineLr scheduler_;
  std::size_t step_ = 0;
};
}
```

Algorithm:

```text
for step in 1..max_steps:
    batch = sample (batch_size, block_size) from train
    logits = model.forward(batch.inputs)
    loss = categorical_cross_entropy(logits, batch.targets)
    model.zero_grad(); model.backward(loss_grad)
    if grad_clip_norm > 0: clip global norm of all parameter grads
    scheduler.apply(optimizer, step)
    optimizer.step over all parameters
    every eval_interval: val_loss = evaluate(...)
    every log_interval: logger.info(...) + callback(stats)
    every checkpoint_interval: save_checkpoint(...)
```

Rules:

- Global gradient-norm clipping: compute `sqrt(sum ||g||^2)`, scale all grads by
  `min(1, clip / norm)` when `norm > clip`.
- AdamW (decoupled weight decay) with the warmup+cosine schedule.
- Evaluation uses a fixed set of validation batches and never updates
  parameters or optimizer state.
- Loss is reported per token (mean over `B*T`), not per sequence.
- The trainer never prints; it logs through `Logger`/`EpochCallback`.
- Checkpoints store model params, optimizer step, and config (see below).
- ForgeFP implementation: the loss is `math/losses.hpp`'s CCE (which calls
  `fp::log_softmax`); gradient clipping is one `fp::fold_left` over the
  parameter grads plus `fp::transform_inplace` to scale them;
  optimizer steps are `fp::axpy_inplace`; step timing is
  `fp::Stopwatch::lap()`; evaluation is wrapped in
  `fp::scope_exit`-style restore (grad mode) and never calls `optimizer.step`.

Tests (`test/llm_test.cpp`):

- `train_step` decreases loss on a repeated string with a fixed seed
- gradient clipping caps the global norm
- evaluation does not change parameters (snapshot compare)
- checkpoint save/load resumes with the same loss

---

## `llm/sampling.hpp`

**Job:** turn logits into tokens. Pure functions over a logits vector so they
are testable without a model.

```cpp
namespace forgeml {

struct SamplingConfig {
  enum class Mode { Greedy, Temperature, TopK, TopP } mode = Mode::Temperature;
  double temperature = 1.0;
  std::size_t top_k = 40;
  double top_p = 0.9;
  int seed_token = -1;              // -1 = none
  double repetition_penalty = 1.0;  // 1.0 = off
};

// logits: (vocab). Returns the chosen token id.
int sample_token(Vector<double> const &logits, SamplingConfig const &config,
                 fp::Rng &rng);

// Generate up to max_new_tokens, feeding tokens back into the model.
std::vector<int> generate(CausalLm &model, std::vector<int> const &prompt,
                          std::size_t max_new_tokens,
                          SamplingConfig const &config, fp::Rng &rng);
}
```

Rules:

- Greedy: `argmax`, ties -> smallest id.
- Temperature: `softmax(logits / T)`; `T <= 0` asserts.
- Top-k: keep the `k` largest logits, renormalize, sample.
- Top-p (nucleus): sort descending, keep the smallest prefix with cumulative
  probability `>= p` (always at least one token), renormalize, sample.
- Repetition penalty divides positive logits and multiplies negative logits of
  already-generated tokens by `1/penalty` (standard GPT-2 form).
- All sampling goes through the passed `fp::Rng`; no `random_device`.
- `generate` must not modify the model's parameters.

ForgeFP implementation:

```cpp
// greedy
int token = static_cast<int>(*fp::argmax(logits));

// temperature
auto probs = fp::softmax(fp::scale(logits, 1.0 / temperature));
int token = static_cast<int>(rng.categorical(probs));

// top-k: keep the k largest, then sample from the renormalized slice
auto order = fp::sort_by(fp::range(0, (int)logits.size()),
                         [&](int i) { return -logits[i]; });
// take order[0..k), build weights, rng.categorical

// top-p: sort descending, fp::scan for the cumulative mass, cut at >= p
```

`fp::softmax` (max-subtracted), `fp::argmax` (first on ties), `fp::sort_by`,
`fp::scan`, and `fp::Rng::categorical` are the whole sampler; the domain part
is the filtering policy and the repetition penalty.

Tests:

- greedy on a hand-made logits vector picks the argmax
- temperature -> 0 approaches greedy
- top-k output is always within the kept set
- top-p cumulative mass is `>= p`
- repetition penalty lowers the score of a repeated token

---

## `llm/checkpoint.hpp`

**Job:** save/load everything needed to resume: weights, optimizer moments,
step, config, tokenizer.

```cpp
namespace forgeml {

struct LmCheckpoint {
  CausalLmConfig model_config;
  std::vector<Parameter> parameters;
  std::size_t step = 0;
  double best_val_loss = 0.0;
  // optimizer state serialized by optim::Adam (m, v, t per parameter index)
};

fp::Result<void> save_checkpoint(std::string const &path,
                                 CausalLm const &model,
                                 std::size_t step, double val_loss);
fp::Result<LmCheckpoint> load_checkpoint(std::string const &path);

// Convenience: write a small model card next to the checkpoint.
fp::Result<void> write_model_card(std::string const &path,
                                  CausalLmConfig const &config,
                                  std::size_t num_parameters,
                                  std::size_t tokens_seen,
                                  std::string const &tokenizer_path);
}
```

Rules:

- Checkpoint format is the `utils/serialization.hpp` schema over
  `fp::to_text`/`fp::to_bytes`; parameters round-trip exactly at precision 17.
- Load validates the config and parameter shapes before returning
  (`fp::from_text`/`from_bytes` already reject truncated input).
- `save_checkpoint` calls `fp::ensure_directory` first and writes with
  `fp::write_bytes`; `load_checkpoint` uses `fp::read_bytes`.
- Optimizer state is optional for v1 (documented: resuming restarts Adam
  moments); add it when long runs matter.

Tests:

- save -> load -> forward produces identical logits
- mismatched shape in the file returns `fail` with the entry name
- model card contains the parameter count

---

## End-to-end pipeline

```text
corpus.txt
  -> CharTokenizer(corpus)                      (vocab ~= 60)
  -> tokens = encode(corpus)
  -> LmDataset::split(tokens, block=64, val=0.1)
  -> CausalLm(vocab, d_model=128, heads=4, layers=4, ff=512)
  -> LmTrainer(...).train(rng)
  -> sample_token / generate
  -> save_checkpoint
```

Suggested milestones for the LLM stage:

1. Overfit a single repeated sentence: loss -> near 0, greedy generation
   reproduces it exactly.
2. Train on a small text file (a few hundred KB): loss drops by at least 2x
   from initialization; temperature sampling produces word-like output.
3. Add BPE and retrain; verify perplexity improves at equal token budget.

## Scale notes (future work, out of scope)

- KV cache for generation (recompute currently); the interface should make it
  possible: `CausalLm::forward(ids, past = nullptr)`.
- Gradient accumulation for larger effective batch sizes.
- Mixed precision, blocked attention kernels, weight sharing with a real
  tokenizer library.
- These are explicitly not part of the first implementation.

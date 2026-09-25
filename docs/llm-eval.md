# llm-eval — measuring a language model honestly

A decreasing loss is not a result. This layer defines the metrics (perplexity,
exact match, F1, BLEU/ROUGE-lite), the few-shot harness that produces the
outputs, the multiple-choice scorer, and the contamination check that keeps an
evaluation set from being training data in disguise.

Depends on: `core`, `llm` (model, tokenizer, sampling), `prob`, `interpret`,
ForgeFP.

Files: `eval.hpp`, `fewshot.hpp`, `text_metrics.hpp`, `contamination.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `eval.hpp` | `fp::numerics` (`logsumexp`), `fp::views::chunk`, `fp::linalg::mean` |
| `fewshot.hpp` | `fp::str` (`join`, `split_any`, `to_lower`), `fp::Rng`, `fp::sort_by_cached` |
| `text_metrics.hpp` | `fp::str`, `fp::map_values`, `fp::views::window` |
| `contamination.hpp` | `fp::str::split_any`, `fp::map_values`, `fp::sort_by_cached` |

---

## `llm/eval.hpp`

**Job:** perplexity and the evaluation report.

```cpp
namespace forgeml {

struct Perplexity { double value = 0.0; std::size_t tokens = 0; };
fp::Result<Perplexity> perplexity(LanguageModel const &model, std::vector<std::size_t> const &tokens,
                                  std::size_t context = 128);

struct EvalResult {
  std::string task;             // "perplexity", "mcq", "generation", ...
  double score = 0.0;
  std::size_t examples = 0;
  std::uint64_t seed = 0;
  std::string config;           // human-readable, recorded verbatim
};
struct Report { std::vector<EvalResult> results; std::string to_json() const; std::string to_markdown() const; };
}
```

Rules: perplexity is `exp(mean(-log p(token | prefix)))` with a sliding
context (the model's KV cache is used, so it is O(n) not O(n·context));
tokens are the tokenizer's, and the report records which tokenizer — comparing
perplexities across tokenizers is invalid and the report says so; an empty
token list is an error (perplexity of nothing is undefined).

Tests: a uniform model over 4 tokens gives perplexity 4; a hand-computed
2-token case matches; the sliding context equals full-context evaluation
within tolerance; the JSON round-trips and contains the seed and config.

## `llm/fewshot.hpp`

**Job:** build prompts, run them, parse answers — deterministically.

```cpp
struct FewShotConfig {
  std::size_t shots = 5;
  enum class Select { Fixed, Random, Similarity } select = Select::Fixed;
  std::uint64_t seed = 0;
  SamplingConfig sampling;      // temperature 0 = greedy
  std::size_t max_tokens = 64;
};

struct Example { std::string input, output; };
struct FewShotResult { std::vector<std::string> predictions; std::vector<std::string> prompts; };

fp::Result<FewShotResult> run_few_shot(LanguageModel &model, Tokenizer const &tok,
                                       std::vector<Example> const &train,
                                       std::vector<std::string> const &queries,
                                       FewShotConfig const &config, fp::Rng &rng);
std::string render_prompt(std::vector<Example> const &examples, std::string const &query);
```

Rules: `Fixed` takes the first `shots` examples, `Random` samples without
replacement, `Similarity` picks the nearest by bag-of-words cosine; prompt
rendering is one documented template (no hidden separators); generation uses
`inference/sampling.hpp` with the config's seed so a run is reproducible;
`max_tokens` is a hard stop.

Tests: `shots = 0` gives a zero-shot prompt (hand-rendered comparison);
`Fixed` is stable, `Random` reproduces for a seed; `Similarity` picks the
lexically closest example; the same seed produces identical predictions;
`max_tokens` is respected.

## `llm/text_metrics.hpp`

**Job:** the generation metrics, each with its definition in the code.

```cpp
double exact_match(std::string const &prediction, std::string const &reference);
double token_f1(std::string const &prediction, std::string const &reference);      // SQuAD-style
double bleu_lite(std::vector<std::string> const &prediction, std::vector<std::string> const &reference,
                 std::size_t max_n = 4);                                           // clipped n-grams + brevity
double rouge_lite(std::string const &prediction, std::string const &reference);    // LCS recall/F1
double mcq_accuracy(LanguageModel const &model, std::vector<std::string> const &contexts,
                    std::vector<std::vector<std::string>> const &choices,
                    std::vector<std::size_t> const &answers, Tokenizer const &tok);
```

Rules: exact match normalizes case and whitespace (documented); token F1 uses
multiset overlap; BLEU-lite uses clipped n-gram precision with the brevity
penalty and **no** smoothing (short sentences are reported as-is, with the
caveat documented); ROUGE-lite is the LCS-based F1; MCQ scores each choice by
its **length-normalized log-likelihood** under the model and picks the
highest (deterministic ties by choice index).

Tests: EM/F1 hand examples (including a zero-overlap case); BLEU-lite matches
a hand-computed 2-gram example; ROUGE-lite on a hand pair; MCQ picks the
higher-likelihood choice on a hand-built model; length normalization prevents
a long wrong answer from winning.

## `llm/contamination.hpp`

**Job:** prove the evaluation set is not in the training data.

```cpp
struct Contamination { std::size_t matches = 0; double rate = 0.0; std::vector<std::string> examples; };
Contamination check(std::vector<std::string> const &eval_documents,
                    std::vector<std::string> const &train_documents, std::size_t ngram = 13);
```

Rules: documents are tokenized with `fp::str::split_any`, lower-cased, and
compared by 13-gram overlap (the standard threshold); a document counts as
contaminated when **any** 13-gram appears in the training set; the report names
a few offending documents so the failure is actionable.

Tests: an identical document is flagged; a document that shares only common
words is not; the rate is `matches / documents`; an empty eval set is 0/0
reported as 0.

## Gate

Perplexity matches hand-computed values and full-context evaluation; the
few-shot harness is deterministic and respects its config; every text metric
matches a hand-computed example; MCQ scoring is length-normalized; the
contamination check flags an overlapping document and passes a clean one; the
report records the tokenizer, config and seeds.

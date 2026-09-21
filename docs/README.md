# ForgeML — Implementation Documents

This directory is the implementation spec for **ForgeML**, a from-scratch,
C++20 machine-learning stack built on top of [ForgeFP](../../fp) (`fp/`).

The division of labour is deliberate and is the first thing to internalize:
**ForgeFP owns the general infrastructure; ForgeML owns the ML semantics.**
Containers, linear algebra, numerics, randomness, time, file I/O,
serialization, forward-mode autodiff, in-place kernels and parallelism are
already in fp — ForgeML consumes them and never reimplements them. What is left
for this project is what is genuinely about machine learning: datasets, losses,
metrics, optimizers, models, layers, training, evaluation, tokenization.

Each document specifies one module: the public API, the math, the error cases,
the **ForgeFP functions it consumes**, and the tests that prove it works. Work
through them in the order given in [roadmap.md](roadmap.md); every stage ends
with an acceptance gate.

## Goals

- A complete classical-ML stack: containers, data, metrics, optimizers, models,
  evaluation.
- A neural-network layer system with one shared backprop engine, extended up to
  attention and a tiny decoder-only transformer (LLM).
- Every general-purpose operation delegated to ForgeFP; every ML-specific
  operation specified here.
- Correctness first: numeric routines tested against hand-computed values or
  gradient checks (`fp::central_difference`, `fp::ad::derivative`).
- Zero-cost by construction: hot paths use `fp::inplace`, `fp::simd`, and
  `fp::par_for`, so the functional style does not cost performance.

## Non-goals

- GPU, distributed training, mixed precision (documented as future work).
- Competing with PyTorch on performance; the target is clarity plus
  cache-friendly loops.
- Reimplementing anything ForgeFP already provides (linear algebra, RNG,
  serialization, autodiff, arenas, stopwatches).

## What ForgeFP provides (do not reimplement)

| Need | ForgeFP module | Key functions |
|---|---|---|
| Errors / optionality | `result.hpp`, `validation.hpp`, `error.hpp` | `Result`, `Validation`, `Outcome`, `traverse`, `to_result` |
| Pattern matching | `adt.hpp` | `match`, `case_`, `cond`, `when`, `otherwise` |
| Ranges / pipelines | `ranges.hpp`, `vec.hpp`, `views.hpp` | `map`, `filter`, `fold_left`, `zip_with`, `chunk`, `slide`, `zip3`, `to_vector` |
| In-place kernels | `inplace.hpp` | `for_each`, `transform_inplace`, `map_to`, `zip_transform_inplace`, `sort_by_inplace` |
| Dense linear algebra | `linalg.hpp` | `matmul`, `batched_matmul`, `matvec`, `outer`, `solve`, `row_sums`, `argmax`, `mean`, `variance` |
| Numerics | `numerics.hpp` | `softmax`, `softmax_rows`, `log_softmax`, `logsumexp`, `sigmoid`, `relu`, `clamp`, `linspace`, `approx_equal`, `central_difference` |
| SIMD kernels | `simd.hpp` | `map_inplace`, `map_to`, `dot`, `reduce`, `axpy_inplace` |
| Randomness | `random.hpp` | `fp::Rng`, `uniform`, `normal`, `shuffle`, `sample_indices`, `categorical` |
| Memory / ownership | `memory.hpp`, `arena.hpp` | `Buffer`, `Box`, `Shared`, `with_buffer`, `Arena`, `mark`/`reset_to` |
| Scoped cleanup | `scope.hpp` | `defer`, `scope_exit`, `scope_success`, `scope_fail` |
| Strings | `string.hpp` | `split`, `split_any`, `join`, `trim`, `to_int`, `to_double`, `to_string`, `parse_numbers` |
| File I/O | `io.hpp` | `read_file`, `read_lines`, `read_bytes`, `write_bytes`, `write_lines`, `ensure_directory` |
| Serialization | `serialize.hpp` | `to_text`, `from_text`, `to_bytes`, `from_bytes` |
| Time | `time.hpp` | `now_seconds`, `Stopwatch` |
| Parallelism | `concurrent.hpp` | `ThreadPool`, `par_map`, `par_for`, `par_map_to`, `par_reduce` |
| Forward-mode AD | `autodiff.hpp` *(opt-in)* | `Dual`, `fp::ad::*`, `derivative` |
| Named operators | `ops.hpp` | `plus`, `times`, `abs`, `sqrt`, `exp`, `log`, `min_`, `max_`, `pow`, `clamp` |

## What ForgeML provides (the domain)

| Layer | Contents |
|---|---|
| `core/` | `Shape`, `Vector`, `Matrix` (grid-backed), `Tensor` |
| `math/` | ML-specific activations (`gelu`, `leaky_relu`), losses, impurity/statistics |
| `data/` | datasets, splits, scaling, encoding, batching |
| `metrics/` | regression/classification metrics, confusion matrix |
| `optim/` | optimizer interface, GD, SGD, Adam/AdamW, LR schedules |
| `model/` | model interfaces + linear/logistic regression, Gaussian NB, CART, linear SVM |
| `eval/` | scoring adapters, k-fold CV, grid search |
| `utils/` | assertions, logging, checkpoint schema, gradient-check harness |
| `nn/` | parameters, layers, sequential network, CNN/RNN/LSTM/embedding/attention/transformers |
| `llm/` | tokenizer, LM dataset, causal LM, trainer, sampling, checkpoints |

## Architecture principles

1. **ForgeFP first.** If fp has it, use it. If fp *almost* has it, add the
   general part to fp (see [the fp extensions plan](../../fp/EXTENSIONS.md))
   and keep only the ML semantics here.
2. **One file, one responsibility.** A file defines a type, a pure operation,
   or a workflow coordinator — never a mix.
3. **Dependencies point one way.** fp → `core` → `math`/`data`/`metrics`/
   `optim` → `model` → `eval` → `nn` → `llm`. Low-level code never depends on
   high-level code.
4. **Errors are values.** Expected failures return `fp::Result<T>`; several
   problems at once return `fp::Validation<T>`; contract violations use
   `ML_ASSERT`. See [conventions](conventions.md#error-model).
5. **Determinism.** Randomness is always an explicit `fp::Rng&`. Library code
   never constructs one; tests always seed.
6. **Zero-cost.** Hot paths use `fp::inplace`/`fp::simd`/`fp::par_for`;
   `std::vector`/`fp::Buffer` reuse beats reallocation; no `std::function` in
   per-element loops.
7. **Tests gate progress.** No file is "done" until its module doc's test list
   passes.

## Dependency graph

```text
ForgeFP (fp/)
  |
core (shape, vector, matrix, tensor)
  |\
  | +--> math (activations, losses, stats)          [thin: most math is fp]
  |        |
  |        +--> optim (optimizer, gradient_descent, sgd, adam, lr_scheduler)
  |        |
  |        +--> metrics (regression, classification, confusion_matrix)
  |
  +--> data (dataset, dataloader, split, scaling, encoding)
  |
  +--> model (model, linear_regression, logistic_regression,
              naive_bayes, decision_tree, svm)
  |
  +--> eval (scoring, cross_validation, grid_search)
  |
  +--> nn (parameter, layer, linear, activation, network, conv2d,
            pooling, flatten, rnn, lstm, embedding, attention,
            multi_head_attention, positional_encoding, feed_forward,
            layer_norm, transformer_block, encoder/decoder transformer)
  |
  +--> llm (tokenizer, dataset, causal_lm, trainer, sampling, checkpoint)
```

## Repository layout (target)

```text
ml/
├── src/
│   ├── core/       shape.hpp vector.hpp matrix.hpp tensor.hpp
│   ├── math/       activations.hpp losses.hpp stats.hpp
│   ├── data/       dataset.hpp dataloader.hpp split.hpp scaling.hpp encoding.hpp
│   ├── metrics/    regression.hpp classification.hpp confusion_matrix.hpp
│   ├── optim/      optimizer.hpp gradient_descent.hpp sgd.hpp adam.hpp lr_scheduler.hpp
│   ├── model/      model.hpp linear_regression.hpp logistic_regression.hpp
│   │               naive_bayes.hpp decision_tree.hpp svm.hpp
│   ├── eval/       scoring.hpp cross_validation.hpp grid_search.hpp
│   ├── utils/      assertions.hpp logging.hpp serialization.hpp gradient_check.hpp
│   ├── nn/         parameter.hpp layer.hpp linear.hpp activation.hpp network.hpp
│   │               conv2d.hpp pooling.hpp flatten.hpp rnn_cell.hpp rnn.hpp
│   │               lstm_cell.hpp lstm.hpp embedding.hpp attention.hpp
│   │               multi_head_attention.hpp positional_encoding.hpp
│   │               feed_forward.hpp layer_norm.hpp transformer_block.hpp
│   │               encoder_transformer.hpp decoder_transformer.hpp
│   ├── llm/        tokenizer.hpp dataset.hpp causal_lm.hpp trainer.hpp
│   │               sampling.hpp checkpoint.hpp
│   └── main.cpp
├── test/           one *_test.cpp per module
├── docs/           this directory
├── forge.lua
└── MIGRATION.md    migration of the original seed library to ForgeFP
```

Note what is **absent** compared to earlier drafts: `core/random.hpp`
(`fp::Rng`), `math/linalg.hpp` (`fp::linalg`), and `math/calc.hpp`
(`fp::central_difference`, `fp::ad::derivative`). Those are fp's job now.

## Relationship to the current `src/`

The existing `src/` is the **seed**: `Vector`/`Matrix` aliases, `linalg`,
`calc`, and a scalar autodiff `Value` with an MLP, all migrated to ForgeFP in
[MIGRATION.md](../MIGRATION.md). The target stack grows that seed:

| Current file | Becomes |
|---|---|
| `src/core/vector.hpp` | `core/vector.hpp` (`Vector<T>`, range-friendly) |
| `src/core/matrix.hpp` | `core/matrix.hpp` (grid-backed `Matrix<T>`, so `fp::linalg` applies directly) |
| `src/math/linalg.hpp` | deleted — use `fp::linalg` |
| `src/math/calc.hpp` | deleted — use `fp::numerics`/`fp::autodiff` |
| `src/nn/engine.hpp` | `nn/parameter.hpp` + `nn/layer.hpp` + `nn/network.hpp` (scalar `Value` stays as an optional teaching example) |
| `src/nn/nn.hpp` | `model/` (classical models) + `nn/` (layers) |

## Documents

| Document | Covers |
|---|---|
| [conventions.md](conventions.md) | Build, namespace, error model, ForgeFP usage rules, naming, testing, zero-cost rules |
| [roadmap.md](roadmap.md) | Stage-by-stage plan, milestones, acceptance gates, task checklists |
| [core.md](core.md) | `Shape`, `Vector`, `Matrix`, `Tensor` |
| [math.md](math.md) | ML activations, losses, impurity/statistics (the fp remainder) |
| [data.md](data.md) | Dataset, dataloader, splits, scaling, encoding |
| [metrics.md](metrics.md) | Regression and classification metrics, confusion matrix |
| [optim.md](optim.md) | Optimizer interface, GD, SGD, Adam/AdamW, LR schedules |
| [model.md](model.md) | Model interface and the five classical models |
| [eval.md](eval.md) | Scoring adapters, k-fold CV, grid search |
| [utils.md](utils.md) | Assertions, logging, checkpoint schema, gradient checking |
| [nn.md](nn.md) | Parameters, layers, backprop, CNN/RNN/LSTM/attention/transformers |
| [llm.md](llm.md) | Tokenizer, causal LM, trainer, sampling, checkpointing |

## Definition of done (per file)

1. Compiles clean with `-Wall -Wextra -Wpedantic` (and `-pthread` where fp
   concurrency/SIMD is used).
2. Public API matches the module doc.
3. **ForgeFP usage is explicit**: the file calls the documented fp functions
   and does not reimplement them. Its "ForgeFP usage" block in the module doc
   is complete and accurate.
4. Every error case returns `fp::Result` / `fp::Validation`; no exceptions
   except `ML_ASSERT`.
5. The module's test file exists and passes.
6. No hidden state: no global mutable data, no implicit RNG, no `std::cout` in
   library code.
7. Documented formulas match the implementation.

## Verification strategy

| Kind | Where | What |
|---|---|---|
| Unit / golden values | `test/*_test.cpp` | hand-computed dot, matmul, loss, metric values |
| Property tests | `test/*_test.cpp` | `(A*B)ᵀ = BᵀAᵀ`, scaler invertibility, split covers all samples once |
| Gradient checks | `utils/gradient_check.hpp` + `fp::central_difference` / `fp::ad::derivative` | every layer and loss: `‖analytic − numeric‖ / (1 + ‖numeric‖) < 1e-6` |
| End-to-end demos | `main.cpp` and per-stage demos | synthetic datasets with known answers |
| Regression tests | `test/regression_test.cpp` | fixed seeds, fixed expected loss/metrics |

The stack is only "done" when the milestone gates in
[roadmap.md](roadmap.md#milestones) all pass.

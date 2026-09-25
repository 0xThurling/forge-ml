# active — choosing which labels to buy

When labels are expensive, the question is not "what is the best model" but
"which sample should I label next". Uncertainty, committee disagreement and
diversity — plus the loop and the label-efficiency proof against random
sampling.

Depends on: `core`, `prob`, `model`, `ensemble`, `neighbors`, `eval`, ForgeFP.

Files: `query.hpp`, `loop.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `query.hpp` | `fp::sort_by_cached`, `fp::numerics` (`entropy`), `fp::linalg::norm_l2` |
| `loop.hpp` | `fp::Rng`, `fp::views::chunk`, `fp::sort_by_cached` |

---

## `active/query.hpp`

**Job:** score every unlabeled sample by how much its label would teach.

```cpp
namespace forgeml {

enum class Strategy { LeastConfident, Margin, Entropy, Committee, Diversity };

Vector<double> least_confident(Matrix<double> const &probabilities);
Vector<double> margin(Matrix<double> const &probabilities);         // 1 - (p1 - p2)
Vector<double> entropy_score(Matrix<double> const &probabilities);  // fp::entropy per row

struct Committee { std::vector<Cart> members; };
Vector<double> committee_disagreement(Committee const &c, Matrix<double> const &X);   // vote entropy

Vector<double> k_center_greedy(Matrix<double> const &X, Matrix<double> const &labeled,
                               std::size_t k);                      // diversity, batch-aware
std::vector<std::size_t> select(Vector<double> const &scores, std::size_t batch);
}
```

Rules: probabilities must be rows summing to 1 (validated); the three
uncertainty scores are monotone in the same direction, so a test can compare
their orderings on a hand example; `k_center_greedy` returns the indices that
maximize the minimum distance to the labeled set (greedy, deterministic ties);
`select` breaks ties by index.

Tests: margin picks the sample closest to the decision boundary (hand-built
probabilities); entropy equals `-Σ p log p` on a hand example; committee
disagreement is maximal when members split 50/50; k-center picks a far-away
cluster before a near duplicate; ties are deterministic.

## `active/loop.hpp`

**Job:** the loop — label a batch, retrain, repeat — with a budget and a
stopping rule.

```cpp
struct ActiveConfig {
  Strategy strategy = Strategy::Margin;
  std::size_t batch = 8;
  std::size_t budget = 200;
  double target_accuracy = 1.1;      // > 1 disables
  std::size_t patience = 5;          // batches without improvement
};

struct ActiveResult {
  std::vector<std::size_t> labeled;   // in acquisition order
  Vector<double> accuracy;            // after each batch
  std::size_t labels_used = 0;
};

template <class Fit>
fp::Result<ActiveResult> active_learn(Fit &&fit, Matrix<double> const &X, Vector<double> const &y,
                                      Vector<std::size_t> const &pool, std::size_t initial,
                                      ActiveConfig const &config, fp::Rng &rng);
```

Rules: the initial labeled set is random and seeded; the pool shrinks and
never repeats a label; a model that cannot fit (one class in the batch) is
retried with the next batch, not silently skipped; the accuracy trace is
recorded on a **held-out** set, never the pool; stopping is budget, target, or
patience — whichever comes first.

Tests: on a planted separable dataset, `Margin` reaches a target accuracy with
strictly fewer labels than random sampling (a fixed-seed comparison); the
labeled set has no duplicates; `labels_used` matches `labeled.size()`; the
loop stops exactly at the budget; a target below the baseline stops after one
batch.

## Gate

Uncertainty and committee scores match hand-computed values; k-center gives a
diverse batch; the loop respects the budget and never repeats a label; and on
planted data active learning is measurably more label-efficient than random
sampling.

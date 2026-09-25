# recommend — predicting what a user will like

Collaborative filtering from a ratings (or interaction) matrix: factorize it,
predict the missing entries, rank them, and measure the ranking honestly.
Ranking metrics are not accuracy metrics — that distinction is the lesson here.

Depends on: `core`, `prob`, `eval` (splits), ForgeFP.

Files: `mf.hpp`, `ranking.hpp`, `baseline.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `mf.hpp` | `fp::Rng` (`normal`, `sample_indices`), `fp::linalg` (`matmul`, `solve`, `dot`) |
| `ranking.hpp` | `fp::sort_by_cached`, `fp::views::chunk`, `fp::map` |
| `baseline.hpp` | `fp::map_values`, `fp::sort_by_cached` |

---

## `recsys/mf.hpp`

**Job:** matrix factorization by SGD and ALS, for explicit ratings and
implicit feedback.

```cpp
namespace forgeml {

struct Mf {
  Matrix<double> user;     // (users, k)
  Matrix<double> item;     // (items, k)
  Vector<double> user_bias, item_bias;
  double global_mean = 0.0;
};

struct Rating { std::size_t user, item; double value; };

fp::Result<Mf> fit_sgd(std::vector<Rating> const &ratings, std::size_t users, std::size_t items,
                       std::size_t factors, std::size_t epochs, double lr, double reg,
                       fp::Rng &rng);
fp::Result<Mf> fit_als(std::vector<Rating> const &ratings, std::size_t users, std::size_t items,
                       std::size_t factors, std::size_t iterations, double reg, fp::Rng &rng);
double predict(Mf const &model, std::size_t user, std::size_t item);
std::vector<std::size_t> recommend(Mf const &model, std::size_t user, std::size_t n,
                                   std::vector<std::size_t> const &seen);
}
```

Rules: factors start from `N(0, 0.01)` for a seed-reproducible run; SGD
iterates the ratings in a shuffled order; ALS solves the least-squares problem
per user and per item with a ridge (`fp::linalg::solve`); biases are fit
first (or jointly) so a zero-factor model still beats the global mean;
implicit feedback uses a confidence weight (`1 + alpha * count`) — the
`fit_implicit` variant.

Tests: MF beats the popularity baseline on held-out interactions; ALS and SGD
agree within tolerance on a small dense matrix; a rank-2 interaction matrix is
recovered; `recommend` never returns a seen item; predictions are
seed-reproducible.

## `recsys/ranking.hpp`

**Job:** the metrics that match the task — a ranked list, not a class label.

```cpp
double precision_at_k(std::vector<std::size_t> const &recommended,
                      std::vector<std::size_t> const &relevant, std::size_t k);
double recall_at_k(std::vector<std::size_t> const &recommended,
                   std::vector<std::size_t> const &relevant, std::size_t k);
double average_precision_at_k(std::vector<std::size_t> const &recommended,
                              std::vector<std::size_t> const &relevant, std::size_t k);
double ndcg_at_k(std::vector<std::size_t> const &recommended,
                 std::vector<std::size_t> const &relevant, std::size_t k);
double hit_rate_at_k(std::vector<std::size_t> const &recommended,
                     std::vector<std::size_t> const &relevant, std::size_t k);
```

Rules: relevance is binary; NDCG uses `1/log2(rank+1)` with the ideal DCG as
the denominator; all metrics return 0 for empty relevance rather than
dividing by zero; the evaluation harness iterates users in a fixed order.

Tests: each metric matches a hand-computed ranking; a perfect ranking scores 1
on all five; NDCG penalises a relevant item at rank 3 more than at rank 1;
precision@k is non-increasing in `k` for a fixed list.

## `recsys/baseline.hpp`

**Job:** the baselines every recommender must beat — and the bias model that
often does surprisingly well.

```cpp
struct Popularity { Vector<std::size_t> order; };
Popularity most_popular(std::vector<Rating> const &ratings, std::size_t items);
double predict(Popularity const &model, std::size_t item);   // rank-based score

struct BiasModel { double global_mean; Vector<double> user_bias, item_bias; };
fp::Result<BiasModel> fit_biases(std::vector<Rating> const &ratings,
                                 std::size_t users, std::size_t items,
                                 double reg = 1.0, std::size_t iterations = 10);
double predict(BiasModel const &model, std::size_t user, std::size_t item);
```

Rules: bias fitting alternates user and item updates with a ridge; the
popularity baseline returns a deterministic order (ties by item index).

Tests: the bias model beats the global mean on held-out ratings; the
popularity order matches hand counts; ties are broken deterministically.

## Gate

MF beats the popularity baseline on held-out interactions; ALS and SGD agree;
every ranking metric matches a hand-computed example; `recommend` excludes
seen items.

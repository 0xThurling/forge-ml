# semi — learning when most labels are missing

Three ways to use unlabeled data: teach yourself (pseudo-labels), propagate
along a similarity graph, or learn a representation without labels
(contrastive). Each is compared against the supervised-only baseline it must
beat.

Depends on: `core`, `nn`, `optim`, `neighbors` (graphs, k-NN), `features`
(augmentation), `prob`, ForgeFP.

Files: `pseudo.hpp`, `propagate.hpp`, `contrastive.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `pseudo.hpp` | `fp::sort_by_cached`, `fp::views::chunk`, `fp::Rng` |
| `propagate.hpp` | `fp::linalg` (`matmul`, `norm_l2`), `fp::inplace` |
| `contrastive.hpp` | `fp::Rng`, `fp::numerics` (`log_softmax`), `fp::linalg` |

---

## `semi/pseudo.hpp`

**Job:** self-training — predict, keep the confident ones, retrain.

```cpp
namespace forgeml {

struct PseudoConfig {
  double threshold = 0.9;      // confidence to accept
  std::size_t rounds = 5;
  bool balance = true;         // cap per class to avoid drift
};

struct PseudoResult { Vector<std::size_t> pseudo_labels; Vector<double> confidence; std::size_t rounds_run = 0; };

template <class Fit>
fp::Result<PseudoResult> self_train(Fit &&fit, Matrix<double> const &X_labeled,
                                    Vector<double> const &y_labeled,
                                    Matrix<double> const &X_unlabeled,
                                    PseudoConfig const &config, fp::Rng &rng);
}
```

Rules: only samples above `threshold` are accepted; with `balance`, each class
contributes at most the largest accepted class's count (avoids the majority
class swallowing the set); the loop stops early when a round accepts nothing;
the labeled set is never modified in place.

Tests: on two planted Gaussians with 5 labels, self-training beats
supervised-only on a held-out set; a threshold of 1.0 accepts nothing (stops
after one round); balance caps the class counts; a deliberately flipped
pseudo-label is excluded by the next round's confidence.

## `semi/propagate.hpp`

**Job:** label propagation on a similarity graph — labels diffuse, seeds stay
fixed.

```cpp
struct Propagation {
  Matrix<double> W;            // similarity (k-NN graph, RBF weights)
  Matrix<double> labels;       // (n, classes), clamped at seeds
};

fp::Result<Propagation> build_graph(Matrix<double> const &X, std::size_t k, double sigma);
fp::Result<Matrix<double>> propagate(Propagation &p, std::size_t iterations = 100, double tolerance = 1e-6);
Matrix<double> label_spread(Matrix<double> const &X, Vector<std::size_t> const &seed_index,
                            Vector<std::size_t> const &seed_label, std::size_t classes,
                            std::size_t k, double sigma);
```

Rules: the graph is symmetrized and row-normalized (`D^-1 W`); propagation
iterates `F ← αWF + (1-α)Y` until the change is below `tolerance`; seeds are
clamped every iteration; an isolated node keeps its prior.

Tests: two planted clusters are recovered from two seeds each; iteration
converges (the change decreases monotonically); propagation on a fully
connected uniform graph gives equal probabilities; a seed with a wrong label
does not flip its own cluster.

## `semi/contrastive.hpp`

**Job:** InfoNCE — learn a representation where augmentations of the same
sample are close and different samples are far.

```cpp
struct Contrastive {
  Matrix<double> projection;    // (d, p) linear head
  double temperature = 0.5;
};

fp::Result<Contrastive> make_contrastive(std::size_t in_features, std::size_t projection,
                                         fp::Rng &rng);
double info_nce_loss(Contrastive const &head, Matrix<double> const &anchors,
                     Matrix<double> const &positives, double temperature);
fp::Result<void> train_step(Contrastive &head, Matrix<double> const &anchors,
                            Matrix<double> const &positives, double lr);
Matrix<double> embed(Contrastive const &head, Matrix<double> const &X);
```

Rules: the loss is the standard in-batch softmax over cosine similarities with
the temperature; anchors and positives are two independently augmented views
(`features/augment`); representations are L2-normalized before the similarity;
no negative mining (in-batch negatives only) — documented as the simple
version.

Tests: the loss decreases over steps on a planted dataset; after training, a
k-NN classifier on the embeddings separates classes that raw features could
not (the planted non-linear case); augmentation is deterministic for a seed;
the loss is invariant to batch permutation (up to the same pairs).

## Gate

Self-training beats supervised-only with few labels; label propagation
recovers planted clusters from a handful of seeds; InfoNCE learns embeddings
that a k-NN classifier can use where raw features fail; all three are
seed-reproducible and compared against their baseline.

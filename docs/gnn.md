# gnn — learning on graphs

Message passing over a small graph: nodes have features, edges say who talks
to whom, and a layer mixes each node's features with its neighbours'. Graphs
are small and dense-ish here (molecules, citation toy sets, community
detection), so the implementation stays explicit — no sparse-matrix
machinery beyond adjacency lists.

Depends on: `core`, `nn` (activations, losses), `optim`, `prob`, ForgeFP.

Files: `graph.hpp`, `gcn.hpp`, `gat.hpp`, `pool.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `graph.hpp` | `fp::sort_by`, `fp::views::chunk`, `fp::map`, `fp::Validation` |
| `gcn.hpp` | `fp::linalg::matmul`, `fp::grid`, `fp::inplace` |
| `gat.hpp` | `fp::numerics` (`softmax`, `logsumexp`), `fp::linalg` |
| `pool.hpp` | `fp::linalg` (`mean`, `max_`), `fp::views::chunk` |

---

## `gnn/graph.hpp`

**Job:** a graph and the operations every layer needs — adjacency, self-loops,
normalization, batching and permutation.

```cpp
namespace forgeml {

struct Graph {
  std::size_t nodes = 0;
  std::vector<std::pair<std::size_t, std::size_t>> edges;   // (from, to)
  Matrix<double> features;                                   // (nodes, d)
  Vector<double> edge_weight;                                // empty = 1.0 each
};

fp::Result<void> validate(Graph const &g);                   // indices, shapes
std::vector<std::vector<std::size_t>> adjacency(Graph const &g);
Graph add_self_loops(Graph const &g);
Graph normalize(Graph const &g);                             // D^-1/2 (A+I) D^-1/2
Matrix<double> to_dense(Graph const &g);                     // small graphs only

struct GraphBatch {                                          // block-diagonal batch
  Matrix<double> features;
  std::vector<std::size_t> graph_of_node;                    // node -> graph index
  std::vector<std::pair<std::size_t, std::size_t>> edges;    // offsets already applied
};
GraphBatch batch(std::vector<Graph> const &graphs);
Graph permute(Graph const &g, std::vector<std::size_t> const &order);   // for invariance tests
}
```

Rules: edges are directed (`to_dense` is not symmetric unless the input is);
`normalize` adds self-loops first; `batch` never connects two graphs; an
out-of-range index is a `fp::Validation` with **all** bad edges reported.

Tests: `to_dense` matches a hand-built adjacency; `normalize` rows sum as
expected on a 3-node example; batching two graphs keeps their edges separate;
`permute` is invertible.

## `gnn/gcn.hpp`

**Job:** the graph convolution layer and a small multi-layer network for node
classification.

```cpp
struct GcnLayer { Matrix<double> W; Vector<double> bias; };
struct Gcn {
  std::vector<GcnLayer> layers;
  std::vector<Matrix<double>> activations;   // cached for backward
};

fp::Result<Gcn> make_gcn(std::size_t in, std::vector<std::size_t> const &hidden,
                         std::size_t out, fp::Rng &rng);
Matrix<double> forward(Gcn &net, Graph const &g);                 // (nodes, out) logits
fp::Result<double> train_step(Gcn &net, Graph const &g, Vector<std::size_t> const &labels,
                              Vector<bool> const &mask, double lr);   // semi-supervised mask
```

Rules: `forward` is `A_norm @ X @ W + b` per layer with ReLU between and no
activation on the last; only masked nodes contribute to the loss (the classic
semi-supervised setting); weights initialize with He scaling from `fp::Rng`;
`train_step` is one full-batch gradient step (small graphs, no mini-batching).

Tests: gradient check on a single layer (< 1e-6); a planted two-community
graph is classified above 90% with one hidden layer; node permutation leaves
logits permuted (equivariance); an all-false mask is an error.

## `gnn/gat.hpp`

**Job:** attention over neighbours — learn *which* edges matter.

```cpp
struct GatLayer { Matrix<double> W; Vector<double> a; std::size_t heads = 1; };
struct Gat { std::vector<GatLayer> layers; };

fp::Result<Gat> make_gat(std::size_t in, std::size_t hidden, std::size_t out,
                         std::size_t heads, fp::Rng &rng);
Matrix<double> forward(Gat &net, Graph const &g);
Matrix<double> attention(Gat const &net, Graph const &g, std::size_t layer);  // (edges, heads)
```

Rules: coefficients are `softmax_j(LeakyReLU(aᵀ[Wh_i ‖ Wh_j]))` computed in log
space; multi-head outputs are averaged (not concatenated) on the last layer;
self-loops participate; attention rows sum to 1 per node.

Tests: attention matches a hand-computed 3-node example; on a graph with one
informative edge, its attention weight rises above the others after training;
attention rows sum to 1; permutation equivariance.

## `gnn/pool.hpp`

**Job:** graph-level readout for whole-graph tasks.

```cpp
Matrix<double> global_mean_pool(Matrix<double> const &node_features, GraphBatch const &batch);
Matrix<double> global_max_pool(Matrix<double> const &node_features, GraphBatch const &batch);
Matrix<double> top_k_pool(Matrix<double> const &node_features, GraphBatch const &batch,
                          std::size_t k, Matrix<double> const &gate);
```

Rules: pooling never mixes nodes from different graphs in a batch; `top_k_pool`
keeps `k` nodes per graph (fewer if the graph is smaller) and is deterministic
on ties (by node index).

Tests: pooling a batch equals pooling each graph and concatenating; max pool
picks the planted extreme value; top-k respects graph boundaries.

## Gate

Permutation equivariance holds for both layers; GCN classifies a planted
community graph; GAT learns to weight an informative edge; batching and
pooling never leak across graphs; every layer passes a gradient check.

# rl — learning from rewards (optional tier)

Bandits, tabular reinforcement learning and deep RL: the smallest complete loop
from "act, observe reward, update" to "solved". **This tier is explicitly
optional and isolated** — nothing else in the stack depends on it, and it can
land last.

Depends on: `core`, `prob`, `nn` (deep RL only), `optim`, `eval` (seeded
evaluation), ForgeFP.

Files: `bandit.hpp`, `tabular.hpp`, `env.hpp`, `policy_gradient.hpp`,
`actor_critic.hpp`, `dqn.hpp`, `ppo.hpp`.

**ForgeFP usage at a glance**

| File | ForgeFP functions |
|---|---|
| `bandit.hpp` | `fp::Rng`, `fp::map_values`, `fp::sort_by_cached` |
| `tabular.hpp` | `fp::Rng`, `fp::map`, `fp::inplace` |
| `env.hpp` | `fp::Rng`, `fp::grid` |
| `policy_gradient.hpp` | `fp::Rng`, `fp::numerics` (`log_softmax`), `fp::inplace` |
| `actor_critic.hpp` | `fp::Rng`, `fp::linalg`, `fp::inplace` |
| `dqn.hpp` | `fp::Rng::sample_indices`, `fp::Buffer` (replay), `fp::inplace` |
| `ppo.hpp` | `fp::Rng`, `fp::linalg::dot`, `fp::inplace` |

---

## `rl/bandit.hpp`

**Job:** the exploration/exploitation trade-off in its purest form.

```cpp
namespace forgeml {

struct Bandit {
  std::size_t arms;
  std::function<double(std::size_t arm, fp::Rng &)> reward;
};

struct EpsilonGreedy { double epsilon; Vector<double> mean; Vector<std::size_t> pulls; };
std::size_t select(EpsilonGreedy &policy, Bandit const &bandit, fp::Rng &rng);
void update(EpsilonGreedy &policy, std::size_t arm, double reward);

struct Ucb { Vector<double> mean; Vector<std::size_t> pulls; double c = 2.0; };
std::size_t select(Ucb &policy, Bandit const &bandit, std::size_t step);

struct Thompson { Vector<double> alpha, beta; };            // Beta–Bernoulli
std::size_t select(Thompson &policy, Bandit const &bandit, fp::Rng &rng);
void update(Thompson &policy, std::size_t arm, double reward);   // 0/1

struct Regret { Vector<double> cumulative; double total() const; };
}
```

Rules: policies are plain structs with explicit `select`/`update` (no virtual
dispatch in the hot loop); `epsilon` may decay via `decay(policy, factor)`;
`Regret` records the gap to the best arm's true mean, so a test can assert
convergence within a fixed budget.

Tests: with a fixed seed, ε-greedy (ε = 0.1), UCB1 and Thompson all pull the
best arm most often after 2 000 steps; Thompson's posterior concentrates on
the best arm; regret grows sublinearly (the final 10% of pulls add less than
the first 10%).

## `rl/tabular.hpp`

**Job:** value learning on small discrete problems — Q-learning, SARSA and
TD(0) — with deterministic tests.

```cpp
struct QLearning {
  Matrix<double> q;          // (states, actions)
  double alpha = 0.1, gamma = 0.99;
  double epsilon = 0.1;
};

std::size_t select(QLearning &learner, std::size_t state, fp::Rng &rng);
void update(QLearning &learner, std::size_t s, std::size_t a, double r,
            std::size_t s_next, bool done);
Vector<std::size_t> greedy_policy(QLearning const &learner);

struct Sarsa { Matrix<double> q; double alpha = 0.1, gamma = 0.99, epsilon = 0.1; };
void update(Sarsa &learner, std::size_t s, std::size_t a, double r, std::size_t s_next, std::size_t a_next, bool done);

Vector<double> td0(Vector<double> const &rewards, Matrix<double> const &transitions,
                   std::size_t episodes, double alpha, double gamma, fp::Rng &rng);
```

Rules: updates are synchronous (no shared state between calls); `epsilon` is
validated to `[0, 1]`; the greedy policy breaks ties by the lowest action
index; TD(0) returns the learned value function so a test can compare it with
the analytic solution of the chain.

Tests: Q-learning solves a 5×5 gridworld (the greedy policy reaches the goal
in the known optimal number of steps); SARSA learns the safer policy on a
cliff variant (fewer falls); TD(0) values on a chain match the analytic
`-k` for the `k`-th state from the terminal; a fixed seed reproduces the
training run.

## `rl/env.hpp`

**Job:** the tiny environment interface the tests run against — no Gym, no
dependencies. Discrete-state environments for tabular methods, a
continuous-state interface for deep RL.

```cpp
struct Step { double reward; std::size_t state; bool done; };

struct Env {
  virtual ~Env() = default;
  virtual std::size_t states() const = 0;
  virtual std::size_t actions() const = 0;
  virtual std::size_t reset(fp::Rng &rng) = 0;
  virtual Step step(std::size_t action) = 0;
  virtual Matrix<double> transition_matrix() const = 0;   // for TD(0) tests
};

struct Gridworld : Env { /* deterministic walls/goal, seeded start */ };
struct Chain : Env { /* n states, two actions, analytic values */ };
struct Cliff : Env { /* the classic risk-vs-safety variant */ };

/// Continuous observations for the deep-RL algorithms.
struct ContinuousStep { double reward; Vector<double> observation; bool done; };
struct ContinuousEnv {
  virtual ~ContinuousEnv() = default;
  virtual std::size_t actions() const = 0;
  virtual std::size_t observation_size() const = 0;
  virtual Vector<double> reset(fp::Rng &rng) = 0;
  virtual ContinuousStep step(std::size_t action) = 0;
  virtual double solved_return() const = 0;   // the documented threshold
};

struct CartPole : ContinuousEnv { /* the classic control task */ };
struct ContinuousBandit : ContinuousEnv { /* contextual bandit, one step */ };
```

Rules: environments are deterministic given a seed (no wall-clock); the
transition matrix is exposed so value-learning tests have an analytic
reference; `solved_return()` states the success threshold in the environment
itself (CartPole: mean return ≥ 475 over 100 episodes); rendering is
deliberately out of scope.

Tests: each environment satisfies the interface invariants (reset returns a
valid state, `step` advances or stays); the chain's analytic values match
`td0`; gridworld's optimal path length is the hand-computed one; CartPole's
random policy returns far below `solved_return()`.

## Deep RL (the learning part of the tier)

The algorithms that connect the `nn/` stack to rewards. All of them are
on-policy or off-policy *loops* around existing pieces — no new math.

### `rl/policy_gradient.hpp`

```cpp
struct Reinforce {
  Sequential net;              // policy: observation -> logits
  double gamma = 0.99;
  double baseline = 0.0;       // running mean return, optional
};
fp::Result<Reinforce> make_reinforce(std::size_t obs, std::size_t actions, std::size_t hidden,
                                     fp::Rng &rng);
fp::Result<double> episode(Reinforce &agent, ContinuousEnv &env, fp::Rng &rng, bool train);
```

Rules: returns-to-go are computed backwards and standardized; the loss is
`-log π(a|s) * G` with an optional baseline; one update per episode (or per
batch of episodes).

### `rl/actor_critic.hpp`

```cpp
struct A2c { Sequential actor, critic; double gamma = 0.99, entropy_beta = 0.01; };
fp::Result<double> step(A2c &agent, ContinuousEnv &env, fp::Rng &rng, bool train);
```

Rules: advantage is `r + γV(s') - V(s)` (bootstrapped, truncated at `done`);
the entropy bonus keeps exploration alive; both networks update in one pass.

### `rl/dqn.hpp`

```cpp
struct Replay { Vector<double> obs; Vector<std::size_t> action; Vector<double> reward; Vector<double> next; Vector<bool> done; };
struct Dqn { Sequential online, target; Replay replay; double gamma = 0.99; double epsilon = 1.0; };
std::size_t select(Dqn &agent, Vector<double> const &obs, fp::Rng &rng);
fp::Result<void> train_step(Dqn &agent, std::size_t batch, fp::Rng &rng);
void sync_target(Dqn &agent);
```

Rules: replay is a fixed-capacity ring (`fp::Buffer`); ε decays linearly to a
floor; the target network syncs every `sync_every` steps; the TD target uses
the target network and is truncated at terminal states.

### `rl/ppo.hpp`

```cpp
struct Ppo { Sequential actor, critic; double clip = 0.2, gamma = 0.99, lambda = 0.95; };
fp::Result<double> collect_and_update(Ppo &agent, ContinuousEnv &env, std::size_t horizon,
                                      std::size_t epochs, fp::Rng &rng);
```

Rules: GAE(λ) advantages; the clipped surrogate objective; several epochs over
one batch of trajectories; early stop when the KL between old and new policy
exceeds a threshold.

Tests (all seed-reproducible): REINFORCE solves CartPole within the documented
episode budget; A2C solves it in fewer episodes than REINFORCE (the variance
reduction pays); DQN solves a small discrete task and its replay never grows
past capacity; PPO's clipped update keeps the policy change bounded (the KL
check triggers); all four are compared against the random-policy return.

## The tier contract

RL is isolated by construction, and that is a rule, not a coincidence:

- **Dependency direction**: `rl → nn/optim/prob/eval/core`. **Nothing imports
  `rl`** — not `llm`, not `eval`, not `utils`. A test greps the headers to
  enforce it.
- **Build**: the tier is header-only like the rest, but its tests are compiled
  only when `testing = true` and are skipped when `FORGEML_DISABLE_RL=1` (so a
  minimal build pays nothing).
- **Performance**: RL has its own benchmarks and **no shared budget** with the
  core stack; its slowness cannot regress the model stages.
- **Evaluation protocol**: every RL result is reported as the mean return over
  N seeded episodes with a confidence interval (`prob/bootstrap_ci`), and the
  environment's `solved_return()` is the pass/fail line. A single lucky run is
  never a result.

## Gate

Bandits converge to the best arm within a fixed budget; Q-learning solves the
gridworld with the optimal path length; SARSA prefers the safe route on the
cliff; TD(0) matches the analytic chain values; REINFORCE, A2C, DQN and PPO
each solve their documented task within the episode budget, with seeded means
and confidence intervals; the isolation test (nothing imports `rl`) passes;
everything is seed-reproducible.

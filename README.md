# Pacman Reinforcement Learning

A reproducible comparison of reinforcement-learning and online-planning methods in a grid-based Pacman environment.

The project implements and evaluates:

- tabular Q-learning;
- Monte Carlo Tree Search (MCTS);
- Deep Q-Network (DQN);
- robustness to random action-execution errors.

The experiments study learning hyperparameters, convergence, path optimality, computational cost and scaling across several grid sizes.

## Main results

| Method | Main observation |
|---|---|
| Q-learning | 100% success and optimal paths after training |
| MCTS | No training required, but slower and less reliable as planning difficulty increases |
| DQN | 100% success on the reference grid after 1,000 episodes |
| Q-learning with action noise | Success decreases from 100% to 92.2% with 20% action errors |

For the current small discrete state space, tabular Q-learning is the most efficient method. DQN validates neural Q-function approximation but does not outperform the simpler Q-table.

## Environment

A grid contains five possible cell types:

| Symbol | Meaning |
|---|---|
| `P` | Initial Pacman position |
| `G` | Ghost |
| `D` | PAC-DOT |
| `B` | Wall |
| `.` | Empty cell |

Pacman can select one of four actions:

- up;
- left;
- right;
- down.

The reward function is:

| Event | Reward |
|---|---:|
| Valid movement or wall collision | -1 |
| Reach the PAC-DOT | +10 |
| Reach the ghost | -10 |

An episode terminates when Pacman reaches the PAC-DOT, reaches the ghost or exceeds the maximum number of steps.

The project contains 40 deterministic layouts:

- 10 layouts of size 4×5;
- 10 layouts of size 6×6;
- 10 layouts of size 8×8;
- 10 layouts of size 10×10.

All layouts are validated automatically, and every valid starting position has a path to the PAC-DOT.

## Q-learning

Tabular Q-learning updates one state-action value after every transition:

\[
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
r
+
\gamma(1-d)
\max_{a'}Q(s',a')
-
Q(s,a)
\right].
\]

The agent uses epsilon-greedy exploration during training.

The experiments study:

- learning rate \(\alpha\);
- discount factor \(\gamma\);
- several random seeds;
- grid sizes from 4×5 to 10×10;
- training time and path quality.

A learning rate of `0.01` learns noticeably more slowly. Values between `0.1` and `0.9` converge rapidly on the tested grids. The retained configuration uses:

```text
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
```

After training, Q-learning obtains a 100% success rate and follows shortest paths across the evaluated sizes and layouts.

## Monte Carlo Tree Search

MCTS plans from the current state without an offline training phase.

Each decision uses four phases:

1. selection with the UCT criterion;
2. expansion of an unexplored action;
3. random simulation until a terminal state;
4. backpropagation of the simulated return.

The selection score is:

\[
UCT(s,a)
=
\overline{Q}(s,a)
+
c
\sqrt{
\frac{\ln N(s)}{N(s,a)}
}.
\]

Internal rollout values are normalized to `[0, 1]` so that the exploitation and exploration terms have comparable scales.

The selected configuration uses:

```text
simulations = 200
exploration_weight = 0.5
```

### Q-learning versus MCTS

| Grid | Q-learning success | MCTS success | MCTS mean decision time |
|---|---:|---:|---:|
| 4×5 | 100.0% | 91.1% | 0.079 s |
| 6×6 | 100.0% | 70.0% | 0.103 s |
| 8×8 | 100.0% | 45.6% | 0.138 s |
| 10×10 | 100.0% | 58.9% | 0.192 s |

Q-learning pays an initial training cost but selects actions almost immediately afterward. MCTS requires no training, but performs a new and increasingly expensive search for every action.

Performance does not decrease strictly with grid size because difficulty also depends on the particular positions of walls, the ghost and the PAC-DOT.

## Deep Q-Network

The DQN replaces the Q-table with a neural approximation:

\[
Q(s,a) \approx Q_\theta(s,a).
\]

Integer states are converted into one-hot vectors and processed by a multilayer perceptron:

```text
Input state
    ↓
64 neurons + ReLU
    ↓
64 neurons + ReLU
    ↓
4 Q-values
```

The implementation includes:

- epsilon-greedy exploration;
- uniform replay buffer;
- mini-batch training;
- target network;
- terminal-state handling;
- Huber loss.

The temporal-difference target is:

\[
y
=
r
+
\gamma(1-d)
\max_{a'}Q_{\theta^-}(s',a').
\]

The online network minimizes:

\[
L(\theta)
=
\operatorname{Huber}
\left(
Q_\theta(s,a)-y
\right).
\]

The reference experiment uses:

```text
hidden_size = 64
learning_rate = 0.001
gamma = 0.99
batch_size = 64
replay_capacity = 10000
target_update_interval = 100
training_episodes = 1000
```

Results on the reference grid:

| Metric | Result |
|---|---:|
| Success rate | 100.0% |
| Ghost rate | 0.0% |
| Optimal path rate | 100.0% |
| Mean return | 7.73 |

This experiment validates the DQN implementation. It is not a complete statistical comparison across all 40 layouts.

## Random action errors

To evaluate policy robustness, Pacman may execute an action different from the one selected by the learned Q-learning policy.

| Action-error probability | Success | Path efficiency | Optimal path |
|---:|---:|---:|---:|
| 0% | 100.0% | 100.0% | 100.0% |
| 5% | 98.8% | 94.2% | 85.6% |
| 10% | 97.3% | 88.1% | 72.1% |
| 20% | 92.2% | 75.2% | 50.0% |

A perfect deterministic policy does not guarantee an optimal trajectory when action execution becomes stochastic. Small errors mainly cause detours, while larger probabilities increase the risk of collision with the ghost.

## Evaluation metrics

The experiments report:

- success, ghost and timeout rates;
- mean episodic return;
- return standard deviation;
- mean episode length;
- shortest-path length;
- path-efficiency ratio;
- optimal-path rate;
- training time;
- mean decision time;
- decisions per second.

Shortest paths are computed with breadth-first search.

## Project structure

```text
.
├── experiments/
│   ├── train_q_learning.py
│   ├── q_learning_parameters.py
│   ├── q_learning_gamma.py
│   ├── q_learning_scaling.py
│   ├── compare_q_learning_mcts.py
│   ├── q_learning_action_noise.py
│   └── train_dqn.py
├── notebooks/
│   └── 01_pacman_rl_analysis.ipynb
├── pacman_rl/
│   ├── agents/
│   │   ├── q_learning.py
│   │   ├── mcts.py
│   │   └── dqn.py
│   ├── environment.py
│   ├── stochastic_environment.py
│   ├── grids.py
│   ├── training.py
│   ├── dqn_training.py
│   ├── evaluation.py
│   └── mcts_evaluation.py
├── results/
│   └── data/
├── tests/
├── requirements.txt
└── README.md
```

## Installation

Python 3.11 is recommended.

### Windows with Git Bash

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -m pip check
python -c "import numpy, pandas, matplotlib, pygame, torch, pytest; print('Installation OK')"
```

## Running the tests

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

Run only the DQN tests:

```bash
python -m pytest tests/test_dqn.py tests/test_dqn_training.py tests/test_dqn_experiment.py -q
```

## Running the experiments

Reference Q-learning experiment:

```bash
python -m experiments.train_q_learning
```

Q-learning scaling experiment:

```bash
python -m experiments.q_learning_scaling
```

Paired Q-learning and MCTS comparison:

```bash
python -m experiments.compare_q_learning_mcts
```

Random action experiment:

```bash
python -m experiments.q_learning_action_noise
```

Reference DQN experiment:

```bash
python -m experiments.train_dqn
```

Generated CSV files are stored in:

```text
results/data/
```

## Analysis notebook

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Then open:

```text
notebooks/01_pacman_rl_analysis.ipynb
```

The notebook contains the complete analysis of:

- the environment and evaluation protocol;
- Q-learning and the Bellman update;
- learning-rate and discount-factor experiments;
- scaling across grid sizes;
- MCTS and UCT;
- Q-learning versus MCTS;
- random action errors;
- DQN architecture and learning curves;
- the final interpretation.

## Reproducibility

Experiments use explicit random seeds for:

- environment starting positions;
- epsilon-greedy action selection;
- replay-buffer sampling;
- neural-network initialization;
- MCTS simulations.

Detailed results are saved as CSV files so that the notebook can reproduce tables and figures without rerunning the more expensive experiments.

## Limitations

- The ghost does not move.
- Every layout contains one ghost and one PAC-DOT.
- A separate policy is learned for each fixed layout.
- The tabular and DQN states encode Pacman's position rather than raw images.
- MCTS uses random rollouts and a fixed simulation budget.
- The DQN result is evaluated on one reference-grid training run.
- The stochastic-action experiment focuses on the learned Q-learning policy.

## Possible extensions

- evaluate DQN across multiple layouts and seeds;
- use convolutional observations based on the complete grid;
- introduce moving ghosts;
- study larger and partially observable environments;
- implement Double or Dueling DQN;
- study prioritized experience replay;
- compare classical DQN with Rainbow DQN;
- add chance nodes for stochastic MCTS;
- add an interactive Pygame visualization.

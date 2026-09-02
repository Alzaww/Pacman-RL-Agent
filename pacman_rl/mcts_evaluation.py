"""Evaluation utilities for Monte Carlo Tree Search."""

from dataclasses import dataclass
from statistics import fmean, pstdev
from time import perf_counter

from pacman_rl.agents.mcts import MCTSAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import shortest_path_length
from pacman_rl.grids import DOT, GridLike


Position = tuple[int, int]


@dataclass(frozen=True)
class MCTSEvaluationResult:
    """Metrics collected during a greedy MCTS evaluation."""

    episodes: int
    success_rate: float
    ghost_rate: float
    timeout_rate: float
    mean_return: float
    return_std: float
    mean_steps: float
    mean_optimal_steps: float
    mean_efficiency_ratio: float
    optimal_path_rate: float
    total_decisions: int
    total_search_time: float
    mean_decision_time: float
    decisions_per_second: float


def find_cell(
    grid: GridLike,
    symbol: str,
) -> Position:
    """Return the position of one symbol in a grid."""
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index

    raise ValueError(
        f"Cell {symbol!r} was not found."
    )


def evaluate_mcts(
    environment: PacmanEnv,
    agent: MCTSAgent,
    episodes: int,
) -> MCTSEvaluationResult:
    """Evaluate MCTS over several complete episodes."""
    if episodes <= 0:
        raise ValueError(
            "episodes must be strictly positive."
        )

    dot_position = find_cell(
        grid=environment.grid,
        symbol=DOT,
    )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    optimal_lengths: list[int] = []
    efficiency_ratios: list[float] = []
    search_times: list[float] = []

    outcome_counts = {
        "dot": 0,
        "ghost": 0,
        "timeout": 0,
    }

    optimal_episode_count = 0

    for _ in range(episodes):
        start_position = environment.reset()

        optimal_length = shortest_path_length(
            grid=environment.grid,
            start=start_position,
            goal=dot_position,
        )

        if optimal_length is None:
            raise RuntimeError(
                "The PAC-DOT cannot be reached "
                "from the sampled start position."
            )

        total_reward = 0.0
        steps = 0
        done = False
        final_info = {}

        while not done:
            decision_start = perf_counter()

            action = agent.select_action(
                environment
            )

            search_times.append(
                perf_counter()
                - decision_start
            )

            _, reward, done, final_info = (
                environment.step(action)
            )

            total_reward += reward
            steps += 1

        outcome = final_info["outcome"]
        outcome_counts[outcome] += 1

        episode_returns.append(
            total_reward
        )
        episode_lengths.append(
            steps
        )
        optimal_lengths.append(
            optimal_length
        )

        if outcome == "dot":
            efficiency = (
                optimal_length / steps
            )

            if steps == optimal_length:
                optimal_episode_count += 1
        else:
            efficiency = 0.0

        efficiency_ratios.append(
            efficiency
        )

    total_search_time = sum(
        search_times
    )
    total_decisions = len(
        search_times
    )

    return MCTSEvaluationResult(
        episodes=episodes,
        success_rate=(
            outcome_counts["dot"]
            / episodes
        ),
        ghost_rate=(
            outcome_counts["ghost"]
            / episodes
        ),
        timeout_rate=(
            outcome_counts["timeout"]
            / episodes
        ),
        mean_return=fmean(
            episode_returns
        ),
        return_std=pstdev(
            episode_returns
        ),
        mean_steps=fmean(
            episode_lengths
        ),
        mean_optimal_steps=fmean(
            optimal_lengths
        ),
        mean_efficiency_ratio=fmean(
            efficiency_ratios
        ),
        optimal_path_rate=(
            optimal_episode_count
            / episodes
        ),
        total_decisions=total_decisions,
        total_search_time=total_search_time,
        mean_decision_time=(
            total_search_time
            / total_decisions
        ),
        decisions_per_second=(
            total_decisions
            / total_search_time
        ),
    )
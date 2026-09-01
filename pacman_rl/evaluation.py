"""Evaluation utilities for learned Pacman policies."""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv, Position
from pacman_rl.grids import GHOST, WALL, GridLike
from pacman_rl.training import position_to_state


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics collected during policy evaluation."""

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


def shortest_path_length(
    grid: GridLike,
    start: Position,
    goal: Position,
) -> int | None:
    """Return the shortest safe distance from start to goal."""
    if not grid or not grid[0]:
        raise ValueError("The grid cannot be empty.")

    rows = len(grid)
    cols = len(grid[0])

    if any(len(row) != cols for row in grid):
        raise ValueError(
            "All grid rows must have the same length."
        )

    for name, position in (
        ("start", start),
        ("goal", goal),
    ):
        row, col = position

        if not (
            0 <= row < rows
            and 0 <= col < cols
        ):
            raise ValueError(
                f"The {name} position is outside the grid: "
                f"{position}"
            )

    if start == goal:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    movements = (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    )

    while queue:
        (row, col), distance = queue.popleft()

        for row_delta, col_delta in movements:
            next_position = (
                row + row_delta,
                col + col_delta,
            )
            next_row, next_col = next_position

            inside_grid = (
                0 <= next_row < rows
                and 0 <= next_col < cols
            )

            if not inside_grid:
                continue

            if next_position in visited:
                continue

            if grid[next_row][next_col] in {WALL, GHOST}:
                continue

            if next_position == goal:
                return distance + 1

            visited.add(next_position)
            queue.append(
                (next_position, distance + 1)
            )

    return None


def evaluate_q_learning(
    env: PacmanEnv,
    agent: QLearningAgent,
    episodes: int = 100,
) -> EvaluationResult:
    """Evaluate a Q-learning policy without exploration."""
    if episodes <= 0:
        raise ValueError("episodes must be strictly positive.")

    expected_states = env.rows * env.cols

    if agent.n_states != expected_states:
        raise ValueError(
            f"The agent has {agent.n_states} states, "
            f"but the environment requires {expected_states}."
        )

    if agent.n_actions != env.n_actions:
        raise ValueError(
            f"The agent has {agent.n_actions} actions, "
            f"but the environment requires {env.n_actions}."
        )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    optimal_lengths: list[int] = []
    efficiency_ratios: list[float] = []
    outcomes: Counter[str] = Counter()
    optimal_path_count = 0

    for _ in range(episodes):
        position = env.reset()
        done = False
        total_return = 0.0
        info: dict[str, str] = {}

        optimal_steps = shortest_path_length(
            grid=env.grid,
            start=position,
            goal=env.dot_position,
        )

        if optimal_steps is None:
            raise RuntimeError(
                f"No safe path exists from {position} "
                f"to {env.dot_position}."
            )

        while not done:
            state = position_to_state(
                position,
                env.cols,
            )

            action = agent.select_action(
                state,
                training=False,
            )

            position, reward, done, info = env.step(action)
            total_return += reward

        episode_returns.append(total_return)
        episode_lengths.append(env.steps)
        optimal_lengths.append(optimal_steps)
        outcomes[info["outcome"]] += 1

        if info["outcome"] == "dot":
            efficiency = optimal_steps / env.steps

            if env.steps == optimal_steps:
                optimal_path_count += 1
        else:
            efficiency = 0.0

        efficiency_ratios.append(efficiency)

    return EvaluationResult(
        episodes=episodes,
        success_rate=outcomes["dot"] / episodes,
        ghost_rate=outcomes["ghost"] / episodes,
        timeout_rate=outcomes["timeout"] / episodes,
        mean_return=float(np.mean(episode_returns)),
        return_std=float(np.std(episode_returns)),
        mean_steps=float(np.mean(episode_lengths)),
        mean_optimal_steps=float(np.mean(optimal_lengths)),
        mean_efficiency_ratio=float(
            np.mean(efficiency_ratios)
        ),
        optimal_path_rate=optimal_path_count / episodes,
    )
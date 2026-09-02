"""Compare Q-learning and MCTS on paired evaluation scenarios."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from pacman_rl.agents.mcts import MCTSAgent
from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import evaluate_q_learning
from pacman_rl.grids import (
    GRID_GROUPS,
    GridLike,
)
from pacman_rl.mcts_evaluation import evaluate_mcts
from pacman_rl.training import train_q_learning


RESULT_COLUMNS = [
    "method",
    "grid",
    "layout",
    "seed",
    "evaluation_seed",
    "rows",
    "cols",
    "n_states",
    "training_episodes",
    "mcts_simulations",
    "exploration_weight",
    "training_time_seconds",
    "mean_decision_time",
    "decisions_per_second",
    "success_rate",
    "ghost_rate",
    "timeout_rate",
    "mean_return",
    "return_std",
    "mean_steps",
    "mean_optimal_steps",
    "mean_efficiency_ratio",
    "optimal_path_rate",
]


def benchmark_q_learning_decisions(
    agent: QLearningAgent,
    n_states: int,
    decisions: int = 5000,
) -> tuple[float, float]:
    """Measure the time required for greedy Q-table decisions."""
    if n_states <= 0:
        raise ValueError(
            "n_states must be strictly positive."
        )

    if decisions <= 0:
        raise ValueError(
            "decisions must be strictly positive."
        )

    start_time = perf_counter()

    for decision_index in range(decisions):
        state = decision_index % n_states

        agent.select_action(
            state=state,
            training=False,
        )

    total_time = (
        perf_counter()
        - start_time
    )

    mean_decision_time = (
        total_time / decisions
    )

    decisions_per_second = (
        decisions / total_time
    )

    return (
        mean_decision_time,
        decisions_per_second,
    )


def run_paired_comparison(
    grid_groups: Mapping[
        str,
        Iterable[GridLike],
    ],
    seeds: Iterable[int],
    training_episodes: int = 3000,
    evaluation_episodes: int = 3,
    mcts_simulations: int = 200,
    exploration_weight: float = 0.5,
    evaluation_seed_offset: int = 10_000,
    q_decision_benchmark_size: int = 5000,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compare both methods using identical evaluation seeds."""
    grid_groups = {
        grid_name: list(grids)
        for grid_name, grids in grid_groups.items()
    }
    seeds = list(seeds)

    if not grid_groups:
        raise ValueError(
            "grid_groups cannot be empty."
        )

    if any(
        not grids
        for grids in grid_groups.values()
    ):
        raise ValueError(
            "Grid groups cannot contain an empty layout list."
        )

    if not seeds:
        raise ValueError(
            "seeds cannot be empty."
        )

    if training_episodes <= 0:
        raise ValueError(
            "training_episodes must be strictly positive."
        )

    if evaluation_episodes <= 0:
        raise ValueError(
            "evaluation_episodes must be strictly positive."
        )

    if mcts_simulations <= 0:
        raise ValueError(
            "mcts_simulations must be strictly positive."
        )

    if exploration_weight < 0:
        raise ValueError(
            "exploration_weight cannot be negative."
        )

    if q_decision_benchmark_size <= 0:
        raise ValueError(
            "q_decision_benchmark_size "
            "must be strictly positive."
        )

    total_configurations = sum(
        len(grids) * len(seeds)
        for grids in grid_groups.values()
    )

    progress_bar = tqdm(
        total=total_configurations,
        desc="Q-learning vs MCTS",
        unit="pair",
        disable=not show_progress,
    )

    results: list[
        dict[str, float | int | str]
    ] = []

    try:
        for grid_name, grids in grid_groups.items():
            for layout_index, grid in enumerate(
                grids,
                start=1,
            ):
                for seed in seeds:
                    evaluation_seed = (
                        evaluation_seed_offset
                        + seed
                    )

                    training_environment = PacmanEnv(
                        grid=grid,
                        random_start=True,
                        seed=seed,
                    )

                    q_agent = QLearningAgent(
                        n_states=(
                            training_environment.rows
                            * training_environment.cols
                        ),
                        n_actions=(
                            training_environment.n_actions
                        ),
                        alpha=0.1,
                        gamma=0.99,
                        epsilon=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.995,
                        seed=seed,
                    )

                    training_start = perf_counter()

                    train_q_learning(
                        env=training_environment,
                        agent=q_agent,
                        episodes=training_episodes,
                    )

                    q_training_time = (
                        perf_counter()
                        - training_start
                    )

                    q_evaluation_environment = (
                        PacmanEnv(
                            grid=grid,
                            random_start=True,
                            seed=evaluation_seed,
                        )
                    )

                    q_metrics = evaluate_q_learning(
                        env=q_evaluation_environment,
                        agent=q_agent,
                        episodes=evaluation_episodes,
                    )

                    (
                        q_mean_decision_time,
                        q_decisions_per_second,
                    ) = benchmark_q_learning_decisions(
                        agent=q_agent,
                        n_states=(
                            training_environment.rows
                            * training_environment.cols
                        ),
                        decisions=(
                            q_decision_benchmark_size
                        ),
                    )

                    results.append(
                        {
                            "method": "Q-learning",
                            "grid": grid_name,
                            "layout": layout_index,
                            "seed": seed,
                            "evaluation_seed": (
                                evaluation_seed
                            ),
                            "rows": (
                                training_environment.rows
                            ),
                            "cols": (
                                training_environment.cols
                            ),
                            "n_states": (
                                training_environment.rows
                                * training_environment.cols
                            ),
                            "training_episodes": (
                                training_episodes
                            ),
                            "mcts_simulations": 0,
                            "exploration_weight": 0.0,
                            "training_time_seconds": (
                                q_training_time
                            ),
                            "mean_decision_time": (
                                q_mean_decision_time
                            ),
                            "decisions_per_second": (
                                q_decisions_per_second
                            ),
                            "success_rate": (
                                q_metrics.success_rate
                            ),
                            "ghost_rate": (
                                q_metrics.ghost_rate
                            ),
                            "timeout_rate": (
                                q_metrics.timeout_rate
                            ),
                            "mean_return": (
                                q_metrics.mean_return
                            ),
                            "return_std": (
                                q_metrics.return_std
                            ),
                            "mean_steps": (
                                q_metrics.mean_steps
                            ),
                            "mean_optimal_steps": (
                                q_metrics.mean_optimal_steps
                            ),
                            "mean_efficiency_ratio": (
                                q_metrics.mean_efficiency_ratio
                            ),
                            "optimal_path_rate": (
                                q_metrics.optimal_path_rate
                            ),
                        }
                    )

                    mcts_environment = PacmanEnv(
                        grid=grid,
                        random_start=True,
                        seed=evaluation_seed,
                    )

                    mcts_agent = MCTSAgent(
                        simulations=mcts_simulations,
                        exploration_weight=(
                            exploration_weight
                        ),
                        seed=seed,
                    )

                    mcts_metrics = evaluate_mcts(
                        environment=mcts_environment,
                        agent=mcts_agent,
                        episodes=evaluation_episodes,
                    )

                    results.append(
                        {
                            "method": "MCTS",
                            "grid": grid_name,
                            "layout": layout_index,
                            "seed": seed,
                            "evaluation_seed": (
                                evaluation_seed
                            ),
                            "rows": mcts_environment.rows,
                            "cols": mcts_environment.cols,
                            "n_states": (
                                mcts_environment.rows
                                * mcts_environment.cols
                            ),
                            "training_episodes": 0,
                            "mcts_simulations": (
                                mcts_simulations
                            ),
                            "exploration_weight": (
                                exploration_weight
                            ),
                            "training_time_seconds": 0.0,
                            "mean_decision_time": (
                                mcts_metrics.mean_decision_time
                            ),
                            "decisions_per_second": (
                                mcts_metrics.decisions_per_second
                            ),
                            "success_rate": (
                                mcts_metrics.success_rate
                            ),
                            "ghost_rate": (
                                mcts_metrics.ghost_rate
                            ),
                            "timeout_rate": (
                                mcts_metrics.timeout_rate
                            ),
                            "mean_return": (
                                mcts_metrics.mean_return
                            ),
                            "return_std": (
                                mcts_metrics.return_std
                            ),
                            "mean_steps": (
                                mcts_metrics.mean_steps
                            ),
                            "mean_optimal_steps": (
                                mcts_metrics.mean_optimal_steps
                            ),
                            "mean_efficiency_ratio": (
                                mcts_metrics.mean_efficiency_ratio
                            ),
                            "optimal_path_rate": (
                                mcts_metrics.optimal_path_rate
                            ),
                        }
                    )

                    progress_bar.update(1)
    finally:
        progress_bar.close()

    return pd.DataFrame(
        results,
        columns=RESULT_COLUMNS,
    )


def main() -> None:
    """Run and save the paired scaling comparison."""
    results = run_paired_comparison(
        grid_groups=GRID_GROUPS,
        seeds=[
            0,
            1,
            2,
        ],
        training_episodes=3000,
        evaluation_episodes=3,
        mcts_simulations=200,
        exploration_weight=0.5,
        evaluation_seed_offset=10_000,
        q_decision_benchmark_size=5000,
        show_progress=True,
    )

    summary = (
        results
        .groupby(
            ["method", "grid"],
            sort=False,
        )
        .agg(
            runs=("seed", "size"),
            success_rate=(
                "success_rate",
                "mean",
            ),
            ghost_rate=(
                "ghost_rate",
                "mean",
            ),
            timeout_rate=(
                "timeout_rate",
                "mean",
            ),
            mean_return=(
                "mean_return",
                "mean",
            ),
            mean_steps=(
                "mean_steps",
                "mean",
            ),
            efficiency=(
                "mean_efficiency_ratio",
                "mean",
            ),
            optimal_path_rate=(
                "optimal_path_rate",
                "mean",
            ),
            training_time=(
                "training_time_seconds",
                "mean",
            ),
            decision_time=(
                "mean_decision_time",
                "mean",
            ),
        )
    )

    output_directory = Path(
        "results/data"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "q_learning_mcts_comparison.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Paired Q-learning and MCTS comparison")
    print("-------------------------------------")
    print(summary.round(4))
    print()
    print(
        f"Detailed results saved to {output_path}"
    )


if __name__ == "__main__":
    main()
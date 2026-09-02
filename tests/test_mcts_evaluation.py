import pytest

from pacman_rl.agents.mcts import MCTSAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.mcts_evaluation import (
    evaluate_mcts,
    find_cell,
)


def test_find_cell_returns_symbol_position():
    grid = [
        ["P", "D"],
        ["G", "."],
    ]

    assert find_cell(
        grid=grid,
        symbol="D",
    ) == (0, 1)


def test_find_cell_rejects_missing_symbol():
    grid = [
        ["P", "."],
        ["G", "D"],
    ]

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        find_cell(
            grid=grid,
            symbol="X",
        )


def test_mcts_evaluation_collects_episode_metrics():
    grid = [
        ["P", "D"],
        ["G", "."],
    ]

    environment = PacmanEnv(
        grid=grid,
        max_steps=5,
    )

    agent = MCTSAgent(
        simulations=100,
        seed=42,
    )

    result = evaluate_mcts(
        environment=environment,
        agent=agent,
        episodes=3,
    )

    assert result.episodes == 3
    assert result.success_rate == 1.0
    assert result.ghost_rate == 0.0
    assert result.timeout_rate == 0.0

    assert result.mean_return == 10.0
    assert result.return_std == 0.0

    assert result.mean_steps == 1.0
    assert result.mean_optimal_steps == 1.0
    assert result.mean_efficiency_ratio == 1.0
    assert result.optimal_path_rate == 1.0

    assert result.total_decisions == 3
    assert result.total_search_time > 0
    assert result.mean_decision_time > 0
    assert result.decisions_per_second > 0


def test_mcts_evaluation_rejects_invalid_episode_count():
    grid = [
        ["P", "D"],
        ["G", "."],
    ]

    environment = PacmanEnv(
        grid=grid,
        max_steps=5,
    )

    agent = MCTSAgent(
        simulations=10,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        evaluate_mcts(
            environment=environment,
            agent=agent,
            episodes=0,
        )
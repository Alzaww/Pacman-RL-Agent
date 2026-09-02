import pytest

from experiments.compare_q_learning_mcts import (
    RESULT_COLUMNS,
    benchmark_q_learning_decisions,
    run_paired_comparison,
)
from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.grids import GRID_GROUPS


def test_q_learning_decision_benchmark_is_positive():
    agent = QLearningAgent(
        n_states=20,
        n_actions=4,
        seed=42,
    )

    (
        mean_decision_time,
        decisions_per_second,
    ) = benchmark_q_learning_decisions(
        agent=agent,
        n_states=20,
        decisions=100,
    )

    assert mean_decision_time > 0
    assert decisions_per_second > 0


def test_paired_comparison_returns_both_methods():
    results = run_paired_comparison(
        grid_groups={
            "4x5": GRID_GROUPS["4x5"][:1],
        },
        seeds=[42],
        training_episodes=100,
        evaluation_episodes=1,
        mcts_simulations=10,
        exploration_weight=0.5,
        q_decision_benchmark_size=100,
    )

    assert len(results) == 2
    assert list(results.columns) == RESULT_COLUMNS

    assert set(results["method"]) == {
        "Q-learning",
        "MCTS",
    }

    assert results["grid"].nunique() == 1
    assert results["layout"].nunique() == 1
    assert results["seed"].nunique() == 1
    assert results["evaluation_seed"].nunique() == 1


def test_comparison_records_method_specific_costs():
    results = run_paired_comparison(
        grid_groups={
            "4x5": GRID_GROUPS["4x5"][:1],
        },
        seeds=[42],
        training_episodes=100,
        evaluation_episodes=1,
        mcts_simulations=10,
        exploration_weight=0.5,
        q_decision_benchmark_size=100,
    )

    q_result = results[
        results["method"] == "Q-learning"
    ].iloc[0]

    mcts_result = results[
        results["method"] == "MCTS"
    ].iloc[0]

    assert q_result["training_time_seconds"] > 0
    assert q_result["training_episodes"] == 100
    assert q_result["mcts_simulations"] == 0

    assert mcts_result["training_time_seconds"] == 0
    assert mcts_result["training_episodes"] == 0
    assert mcts_result["mcts_simulations"] == 10

    assert q_result["mean_decision_time"] > 0
    assert mcts_result["mean_decision_time"] > 0


@pytest.mark.parametrize(
    (
        "grid_groups",
        "seeds",
        "training_episodes",
        "evaluation_episodes",
        "message",
    ),
    [
        ({}, [0], 100, 1, "grid_groups"),
        ({"4x5": []}, [0], 100, 1, "empty layout"),
        (
            {"4x5": GRID_GROUPS["4x5"][:1]},
            [],
            100,
            1,
            "seeds",
        ),
        (
            {"4x5": GRID_GROUPS["4x5"][:1]},
            [0],
            0,
            1,
            "strictly positive",
        ),
        (
            {"4x5": GRID_GROUPS["4x5"][:1]},
            [0],
            100,
            0,
            "strictly positive",
        ),
    ],
)
def test_comparison_rejects_invalid_inputs(
    grid_groups,
    seeds,
    training_episodes,
    evaluation_episodes,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        run_paired_comparison(
            grid_groups=grid_groups,
            seeds=seeds,
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
        )
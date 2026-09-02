import pytest

from experiments.q_learning_action_noise import (
    RESULT_COLUMNS,
    run_action_noise_experiments,
)
from pacman_rl.grids import REFERENCE_GRID


def test_action_noise_experiment_returns_one_row_per_run():
    results = run_action_noise_experiments(
        grid=REFERENCE_GRID,
        probabilities=[0.0, 0.1],
        seeds=[1, 2],
        training_episodes=100,
        evaluation_episodes=10,
    )

    assert len(results) == 4
    assert list(results.columns) == RESULT_COLUMNS

    assert set(
        results["action_error_probability"]
    ) == {0.0, 0.1}

    assert set(results["seed"]) == {1, 2}


def test_outcome_rates_sum_to_one():
    results = run_action_noise_experiments(
        grid=REFERENCE_GRID,
        probabilities=[0.1],
        seeds=[42],
        training_episodes=100,
        evaluation_episodes=10,
    )

    outcome_sum = (
        results["success_rate"]
        + results["ghost_rate"]
        + results["timeout_rate"]
    )

    assert outcome_sum.iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [-0.1],
        [1.1],
    ],
)
def test_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        run_action_noise_experiments(
            grid=REFERENCE_GRID,
            probabilities=probabilities,
            seeds=[42],
        )


def test_rejects_empty_seeds():
    with pytest.raises(
        ValueError,
        match="seeds cannot be empty",
    ):
        run_action_noise_experiments(
            grid=REFERENCE_GRID,
            probabilities=[0.1],
            seeds=[],
        )
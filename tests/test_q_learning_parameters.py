import pytest

from experiments.q_learning_parameters import (
    RESULT_COLUMNS,
    run_alpha_experiments,
)


def test_alpha_experiment_returns_one_row_per_run():
    results = run_alpha_experiments(
        alpha_values=[0.1, 0.5],
        seeds=[1, 2],
        training_episodes=200,
        evaluation_episodes=20,
    )

    assert len(results) == 4
    assert list(results.columns) == RESULT_COLUMNS

    assert set(results["alpha"]) == {0.1, 0.5}
    assert set(results["seed"]) == {1, 2}


def test_alpha_experiment_returns_valid_rates():
    results = run_alpha_experiments(
        alpha_values=[0.1],
        seeds=[42],
        training_episodes=200,
        evaluation_episodes=20,
    )

    rate_columns = [
        "success_rate",
        "ghost_rate",
        "timeout_rate",
        "mean_efficiency_ratio",
        "optimal_path_rate",
    ]

    for column in rate_columns:
        assert results[column].between(0.0, 1.0).all()

    outcome_sum = (
        results["success_rate"]
        + results["ghost_rate"]
        + results["timeout_rate"]
    )

    assert outcome_sum.iloc[0] == pytest.approx(1.0)


def test_alpha_experiment_rejects_empty_parameters():
    with pytest.raises(
        ValueError,
        match="alpha_values cannot be empty",
    ):
        run_alpha_experiments(
            alpha_values=[],
            seeds=[42],
        )

    with pytest.raises(
        ValueError,
        match="seeds cannot be empty",
    ):
        run_alpha_experiments(
            alpha_values=[0.1],
            seeds=[],
        )
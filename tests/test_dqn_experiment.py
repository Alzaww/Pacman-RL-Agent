import pytest

from experiments.train_dqn import (
    run_dqn_experiment,
)


def test_dqn_experiment_trains_and_evaluates_agent():
    agent, history, metrics = run_dqn_experiment(
        seed=42,
        training_episodes=20,
        evaluation_episodes=10,
    )

    assert len(history.episode_returns) == 20
    assert len(history.mean_losses) == 20

    assert len(agent.replay_buffer) > 0

    outcome_rate = (
        metrics.success_rate
        + metrics.ghost_rate
        + metrics.timeout_rate
    )

    assert outcome_rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    (
        "training_episodes",
        "evaluation_episodes",
    ),
    [
        (0, 10),
        (10, 0),
    ],
)
def test_dqn_experiment_rejects_invalid_counts(
    training_episodes,
    evaluation_episodes,
):
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        run_dqn_experiment(
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
        )
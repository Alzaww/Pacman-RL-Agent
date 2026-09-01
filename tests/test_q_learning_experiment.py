from experiments.train_q_learning import run_experiment


def test_first_q_learning_experiment():
    history, metrics = run_experiment(
        training_episodes=1500,
        evaluation_episodes=100,
        seed=42,
    )

    assert len(history.episode_returns) == 1500
    assert len(history.episode_lengths) == 1500
    assert len(history.outcomes) == 1500

    assert metrics.episodes == 100
    assert metrics.success_rate >= 0.90

    total_outcome_rate = (
        metrics.success_rate
        + metrics.ghost_rate
        + metrics.timeout_rate
    )

    assert total_outcome_rate == 1.0
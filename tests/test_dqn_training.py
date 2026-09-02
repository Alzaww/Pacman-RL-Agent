import math

import pytest

from pacman_rl.agents.dqn import DQNAgent
from pacman_rl.dqn_training import train_dqn
from pacman_rl.environment import PacmanEnv


TEST_GRID = [
    ["P", "D"],
    ["G", "."],
]


def make_agent(env):
    return DQNAgent(
        n_states=env.rows * env.cols,
        n_actions=env.n_actions,
        hidden_size=16,
        learning_rate=0.01,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9,
        replay_capacity=100,
        batch_size=4,
        target_update_interval=5,
        seed=42,
    )


def test_dqn_training_collects_episode_history():
    env = PacmanEnv(
        grid=TEST_GRID,
        max_steps=8,
    )

    agent = make_agent(env)

    history = train_dqn(
        env=env,
        agent=agent,
        episodes=20,
    )

    assert len(history.episode_returns) == 20
    assert len(history.episode_lengths) == 20
    assert len(history.epsilon_values) == 20
    assert len(history.mean_losses) == 20
    assert len(history.outcomes) == 20

    assert all(
        length > 0
        for length in history.episode_lengths
    )


def test_dqn_training_fills_replay_buffer():
    env = PacmanEnv(
        grid=TEST_GRID,
        max_steps=8,
    )

    agent = make_agent(env)

    train_dqn(
        env=env,
        agent=agent,
        episodes=10,
    )

    assert len(agent.replay_buffer) > 0
    assert agent.optimization_steps > 0


def test_dqn_training_decays_epsilon():
    env = PacmanEnv(
        grid=TEST_GRID,
        max_steps=8,
    )

    agent = make_agent(env)
    initial_epsilon = agent.epsilon

    train_dqn(
        env=env,
        agent=agent,
        episodes=5,
    )

    assert agent.epsilon < initial_epsilon
    assert agent.epsilon >= agent.epsilon_min


def test_first_losses_can_be_nan_before_batch_is_full():
    env = PacmanEnv(
        grid=TEST_GRID,
        max_steps=2,
    )

    agent = DQNAgent(
        n_states=env.rows * env.cols,
        n_actions=env.n_actions,
        batch_size=100,
        seed=42,
    )

    history = train_dqn(
        env=env,
        agent=agent,
        episodes=1,
    )

    assert math.isnan(
        history.mean_losses[0]
    )


def test_dqn_training_rejects_invalid_episode_count():
    env = PacmanEnv(
        grid=TEST_GRID,
    )

    agent = make_agent(env)

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        train_dqn(
            env=env,
            agent=agent,
            episodes=0,
        )
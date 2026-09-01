import pytest

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import Action, PacmanEnv
from pacman_rl.training import (
    position_to_state,
    train_q_learning,
)


def test_position_is_converted_to_state_index():
    state = position_to_state(
        position=(2, 3),
        n_columns=5,
    )

    assert state == 13


def test_invalid_position_is_rejected():
    with pytest.raises(ValueError, match="Invalid grid position"):
        position_to_state(
            position=(1, 5),
            n_columns=5,
        )


def test_training_collects_episode_history():
    grid = [
        ["P", "D"],
        ["G", "."],
    ]

    env = PacmanEnv(
        grid=grid,
        max_steps=8,
    )

    agent = QLearningAgent(
        n_states=env.rows * env.cols,
        n_actions=env.n_actions,
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.98,
        seed=42,
    )

    history = train_q_learning(
        env=env,
        agent=agent,
        episodes=400,
    )

    assert len(history.episode_returns) == 400
    assert len(history.episode_lengths) == 400
    assert len(history.epsilon_values) == 400
    assert len(history.outcomes) == 400

    start_state = position_to_state(
        env.start_position,
        env.cols,
    )

    best_action = agent.select_action(
        start_state,
        training=False,
    )

    assert best_action == Action.RIGHT
    assert agent.epsilon == pytest.approx(0.05)

    position = env.reset()
    done = False
    info = {}

    while not done:
        state = position_to_state(
            position,
            env.cols,
        )

        action = agent.select_action(
            state,
            training=False,
        )

        position, _, done, info = env.step(action)

    assert info["outcome"] == "dot"


def test_training_rejects_wrong_number_of_states():
    env = PacmanEnv()

    agent = QLearningAgent(
        n_states=5,
        n_actions=env.n_actions,
    )

    with pytest.raises(ValueError, match="environment requires"):
        train_q_learning(
            env=env,
            agent=agent,
            episodes=10,
        )


def test_training_rejects_wrong_number_of_actions():
    env = PacmanEnv()

    agent = QLearningAgent(
        n_states=env.rows * env.cols,
        n_actions=2,
    )

    with pytest.raises(ValueError, match="environment requires"):
        train_q_learning(
            env=env,
            agent=agent,
            episodes=10,
        )
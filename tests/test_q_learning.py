import numpy as np
import pytest

from pacman_rl.agents.q_learning import QLearningAgent


def test_q_table_is_initialized_with_zeros():
    agent = QLearningAgent(
        n_states=20,
        n_actions=4,
        seed=42,
    )

    assert agent.q_table.shape == (20, 4)
    assert np.all(agent.q_table == 0)


def test_greedy_policy_selects_best_action():
    agent = QLearningAgent(
        n_states=3,
        n_actions=4,
        epsilon=0.0,
        epsilon_min=0.0,
        seed=42,
    )

    agent.q_table[1] = [1.0, 2.0, 8.0, -1.0]

    action = agent.select_action(1, training=False)

    assert action == 2


def test_full_exploration_uses_different_actions():
    agent = QLearningAgent(
        n_states=3,
        n_actions=4,
        epsilon=1.0,
        epsilon_min=0.0,
        seed=42,
    )

    selected_actions = {
        agent.select_action(0, training=True)
        for _ in range(100)
    }

    assert selected_actions == {0, 1, 2, 3}


def test_non_terminal_update_uses_next_state_value():
    agent = QLearningAgent(
        n_states=3,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        seed=42,
    )

    agent.q_table[1] = [4.0, 2.0]

    td_error = agent.update(
        state=0,
        action=1,
        reward=1.0,
        next_state=1,
        done=False,
    )

    expected_target = 1.0 + 0.9 * 4.0
    expected_error = expected_target
    expected_value = 0.5 * expected_error

    assert td_error == pytest.approx(expected_error)
    assert agent.q_table[0, 1] == pytest.approx(expected_value)


def test_terminal_update_ignores_next_state_value():
    agent = QLearningAgent(
        n_states=3,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        seed=42,
    )

    agent.q_table[1] = [100.0, 50.0]

    agent.update(
        state=0,
        action=1,
        reward=10.0,
        next_state=1,
        done=True,
    )

    assert agent.q_table[0, 1] == pytest.approx(5.0)


def test_epsilon_decay_respects_minimum():
    agent = QLearningAgent(
        n_states=3,
        n_actions=2,
        epsilon=0.2,
        epsilon_min=0.1,
        epsilon_decay=0.5,
        seed=42,
    )

    agent.decay_exploration()
    assert agent.epsilon == pytest.approx(0.1)

    agent.decay_exploration()
    assert agent.epsilon == pytest.approx(0.1)


def test_invalid_hyperparameters_are_rejected():
    with pytest.raises(ValueError, match="alpha"):
        QLearningAgent(
            n_states=3,
            n_actions=2,
            alpha=0.0,
        )

    with pytest.raises(ValueError, match="gamma"):
        QLearningAgent(
            n_states=3,
            n_actions=2,
            gamma=1.5,
        )
import pytest

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import Action, PacmanEnv
from pacman_rl.evaluation import (
    evaluate_q_learning,
    shortest_path_length,
)
from pacman_rl.grids import REFERENCE_GRID
from pacman_rl.training import position_to_state


def create_small_environment_and_agent():
    grid = [
        ["P", "D"],
        ["G", "."],
    ]

    env = PacmanEnv(
        grid=grid,
        max_steps=5,
    )

    agent = QLearningAgent(
        n_states=env.rows * env.cols,
        n_actions=env.n_actions,
        epsilon=1.0,
        epsilon_min=0.0,
        seed=42,
    )

    start_state = position_to_state(
        env.start_position,
        env.cols,
    )

    return env, agent, start_state


def test_shortest_path_on_reference_grid():
    distance = shortest_path_length(
        grid=REFERENCE_GRID,
        start=(0, 4),
        goal=(3, 2),
    )

    assert distance == 7


def test_shortest_path_returns_none_when_goal_is_unreachable():
    grid = [
        [".", "B", "D"],
        [".", "B", "B"],
    ]

    distance = shortest_path_length(
        grid=grid,
        start=(0, 0),
        goal=(0, 2),
    )

    assert distance is None


def test_evaluation_uses_greedy_policy():
    env, agent, start_state = (
        create_small_environment_and_agent()
    )

    agent.q_table[start_state, Action.RIGHT] = 10.0

    result = evaluate_q_learning(
        env=env,
        agent=agent,
        episodes=50,
    )

    assert result.episodes == 50
    assert result.success_rate == pytest.approx(1.0)
    assert result.ghost_rate == pytest.approx(0.0)
    assert result.timeout_rate == pytest.approx(0.0)
    assert result.mean_return == pytest.approx(10.0)
    assert result.return_std == pytest.approx(0.0)
    assert result.mean_steps == pytest.approx(1.0)

    assert agent.epsilon == pytest.approx(1.0)


def test_evaluation_detects_ghost_failures():
    env, agent, start_state = (
        create_small_environment_and_agent()
    )

    agent.q_table[start_state, Action.DOWN] = 10.0

    result = evaluate_q_learning(
        env=env,
        agent=agent,
        episodes=20,
    )

    assert result.success_rate == pytest.approx(0.0)
    assert result.ghost_rate == pytest.approx(1.0)
    assert result.timeout_rate == pytest.approx(0.0)
    assert result.mean_return == pytest.approx(-10.0)
    assert result.mean_steps == pytest.approx(1.0)


def test_evaluation_rejects_invalid_episode_count():
    env, agent, _ = create_small_environment_and_agent()

    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_q_learning(
            env=env,
            agent=agent,
            episodes=0,
        )
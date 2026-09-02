import pytest
import torch

from pacman_rl.agents.dqn import (
    DQNAgent,
    QNetwork,
    ReplayBuffer,
)


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(
        capacity=2,
        seed=42,
    )

    buffer.push(0, 0, 0.0, 1, False)
    buffer.push(1, 1, 1.0, 2, False)
    buffer.push(2, 2, 2.0, 3, True)

    assert len(buffer) == 2

    sampled_states = {
        transition.state
        for transition in buffer.sample(2)
    }

    assert sampled_states == {1, 2}


def test_replay_buffer_rejects_large_sample():
    buffer = ReplayBuffer(
        capacity=10,
    )

    buffer.push(0, 0, 0.0, 1, False)

    with pytest.raises(
        ValueError,
        match="Not enough transitions",
    ):
        buffer.sample(2)


def test_q_network_output_shape():
    network = QNetwork(
        n_states=20,
        n_actions=4,
        hidden_size=16,
    )

    states = torch.zeros(
        (5, 20),
        dtype=torch.float32,
    )

    q_values = network(states)

    assert q_values.shape == (5, 4)


def test_greedy_action_uses_highest_q_value():
    agent = DQNAgent(
        n_states=4,
        n_actions=3,
        hidden_size=8,
        epsilon=0.0,
        batch_size=2,
        seed=42,
    )

    for parameter in agent.online_network.parameters():
        parameter.data.zero_()

    final_layer = agent.online_network.network[-1]

    final_layer.bias.data.copy_(
        torch.tensor([0.0, 1.0, 2.0])
    )

    action = agent.select_action(
        state=0,
        training=False,
    )

    assert action == 2


def test_learning_waits_for_complete_batch():
    agent = DQNAgent(
        n_states=4,
        n_actions=2,
        batch_size=2,
        seed=42,
    )

    agent.remember(
        state=0,
        action=0,
        reward=1.0,
        next_state=1,
        done=False,
    )

    assert agent.learn() is None


def test_learning_updates_online_network():
    agent = DQNAgent(
        n_states=4,
        n_actions=2,
        hidden_size=8,
        learning_rate=0.01,
        batch_size=2,
        seed=42,
    )

    for parameter in agent.online_network.parameters():
        parameter.data.zero_()

    agent.update_target_network()

    agent.remember(
        state=0,
        action=0,
        reward=10.0,
        next_state=1,
        done=True,
    )

    agent.remember(
        state=1,
        action=0,
        reward=10.0,
        next_state=2,
        done=True,
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter
        in agent.online_network.parameters()
    ]

    loss = agent.learn()

    parameters_after = list(
        agent.online_network.parameters()
    )

    assert loss is not None
    assert loss > 0.0

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    )


def test_target_network_can_be_synchronized():
    agent = DQNAgent(
        n_states=4,
        n_actions=2,
        hidden_size=8,
        seed=42,
    )

    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.add_(1.0)

    agent.update_target_network()

    for online_parameter, target_parameter in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters(),
    ):
        assert torch.equal(
            online_parameter,
            target_parameter,
        )


def test_epsilon_decay_respects_minimum():
    agent = DQNAgent(
        n_states=4,
        n_actions=2,
        epsilon=0.1,
        epsilon_min=0.05,
        epsilon_decay=0.5,
    )

    agent.decay_exploration()
    assert agent.epsilon == pytest.approx(0.05)

    agent.decay_exploration()
    assert agent.epsilon == pytest.approx(0.05)
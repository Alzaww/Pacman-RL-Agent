"""Deep Q-Network agent and replay buffer."""

from collections import deque
from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Transition:
    """One transition stored in replay memory."""

    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class ReplayBuffer:
    """Fixed-size memory of past transitions."""

    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError(
                "capacity must be strictly positive."
            )

        self.capacity = capacity
        self._memory: deque[Transition] = deque(
            maxlen=capacity
        )
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Store one transition."""
        self._memory.append(
            Transition(
                state=int(state),
                action=int(action),
                reward=float(reward),
                next_state=int(next_state),
                done=bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
    ) -> list[Transition]:
        """Sample transitions uniformly."""
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be strictly positive."
            )

        if batch_size > len(self._memory):
            raise ValueError(
                "Not enough transitions in replay buffer."
            )

        return self._rng.sample(
            list(self._memory),
            batch_size,
        )


class QNetwork(nn.Module):
    """Small neural network estimating Q-values."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden_size: int = 64,
    ):
        super().__init__()

        if n_states <= 0:
            raise ValueError(
                "n_states must be strictly positive."
            )

        if n_actions <= 0:
            raise ValueError(
                "n_actions must be strictly positive."
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be strictly positive."
            )

        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Return one Q-value per action."""
        return self.network(states)


class DQNAgent:
    """DQN agent using replay memory and a target network."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_interval: int = 100,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ):
        if n_states <= 0:
            raise ValueError(
                "n_states must be strictly positive."
            )

        if n_actions <= 0:
            raise ValueError(
                "n_actions must be strictly positive."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be strictly positive."
            )

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                "gamma must be between 0 and 1."
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be between 0 and 1."
            )

        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in (0, 1]."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be strictly positive."
            )

        if target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must be positive."
            )

        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = (
            target_update_interval
        )
        self.device = torch.device(device)

        self._rng = random.Random(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.online_network = QNetwork(
            n_states=n_states,
            n_actions=n_actions,
            hidden_size=hidden_size,
        ).to(self.device)

        self.target_network = QNetwork(
            n_states=n_states,
            n_actions=n_actions,
            hidden_size=hidden_size,
        ).to(self.device)

        self.target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            seed=seed,
        )

        self.optimization_steps = 0

        self.update_target_network()

    def _encode_states(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Convert integer states to one-hot vectors."""
        return F.one_hot(
            states.long(),
            num_classes=self.n_states,
        ).float()

    def select_action(
        self,
        state: int,
        training: bool = True,
    ) -> int:
        """Select an action using epsilon-greedy exploration."""
        if (
            training
            and self._rng.random() < self.epsilon
        ):
            return self._rng.randrange(
                self.n_actions
            )

        state_tensor = torch.tensor(
            [state],
            dtype=torch.long,
            device=self.device,
        )

        encoded_state = self._encode_states(
            state_tensor
        )

        with torch.no_grad():
            q_values = self.online_network(
                encoded_state
            )

        return int(
            q_values.argmax(dim=1).item()
        )

    def remember(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Store one environment transition."""
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def learn(self) -> float | None:
        """Perform one DQN optimization step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(
            [
                transition.state
                for transition in transitions
            ],
            dtype=torch.long,
            device=self.device,
        )

        actions = torch.tensor(
            [
                transition.action
                for transition in transitions
            ],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [
                transition.reward
                for transition in transitions
            ],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            [
                transition.next_state
                for transition in transitions
            ],
            dtype=torch.long,
            device=self.device,
        )

        dones = torch.tensor(
            [
                transition.done
                for transition in transitions
            ],
            dtype=torch.float32,
            device=self.device,
        )

        encoded_states = self._encode_states(
            states
        )

        encoded_next_states = self._encode_states(
            next_states
        )

        predicted_q_values = (
            self.online_network(encoded_states)
            .gather(
                dim=1,
                index=actions.unsqueeze(1),
            )
            .squeeze(1)
        )

        with torch.no_grad():
            next_q_values = (
                self.target_network(
                    encoded_next_states
                )
                .max(dim=1)
                .values
            )

            td_targets = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * next_q_values
            )

        loss = F.smooth_l1_loss(
            predicted_q_values,
            td_targets,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.optimization_steps += 1

        if (
            self.optimization_steps
            % self.target_update_interval
            == 0
        ):
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy online-network weights into the target network."""
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def decay_exploration(self) -> None:
        """Decrease epsilon after an episode."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
"""Monte Carlo Tree Search components."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from pacman_rl.environment import (
    Action,
    PacmanEnv,
)


Position = tuple[int, int]


def clone_environment(
    environment: PacmanEnv,
) -> PacmanEnv:
    """Return an independent copy used for MCTS simulations."""
    return copy.deepcopy(environment)


def random_rollout(
    environment: PacmanEnv,
    random_generator: random.Random,
) -> float:
    """Simulate random actions until the copied episode terminates."""
    simulation = clone_environment(
        environment
    )

    total_reward = 0.0

    while not simulation.done:
        action = random_generator.choice(
            list(Action)
        )

        _, reward, done, _ = simulation.step(
            action
        )

        total_reward += reward

        if done:
            break

    return total_reward


@dataclass
class MCTSNode:
    """Represent one state explored by Monte Carlo Tree Search."""

    state: Position
    parent: MCTSNode | None = None
    action: Action | None = None
    untried_actions: list[Action] = field(
        default_factory=lambda: list(Action)
    )
    children: dict[Action, MCTSNode] = field(
        default_factory=dict
    )
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        """Return the average result of simulations through this node."""
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    @property
    def is_fully_expanded(self) -> bool:
        """Return whether every available action has been expanded."""
        return not self.untried_actions

    def add_child(
        self,
        action: Action,
        state: Position,
        available_actions: list[Action] | None = None,
    ) -> MCTSNode:
        """Expand one untried action and return the new child."""
        if action not in self.untried_actions:
            raise ValueError(
                f"Action {action!r} cannot be expanded."
            )

        child = MCTSNode(
            state=state,
            parent=self,
            action=action,
            untried_actions=(
                list(Action)
                if available_actions is None
                else list(available_actions)
            ),
        )

        self.children[action] = child
        self.untried_actions.remove(action)

        return child

    def uct_score(
        self,
        exploration_weight: float,
    ) -> float:
        """Calculate this node's UCT score."""
        if self.parent is None:
            raise ValueError(
                "The root node does not have a UCT score."
            )

        if self.visits == 0:
            return math.inf

        parent_visits = max(
            1,
            self.parent.visits,
        )

        exploration_bonus = (
            exploration_weight
            * math.sqrt(
                math.log(parent_visits)
                / self.visits
            )
        )

        return (
            self.mean_value
            + exploration_bonus
        )

    def best_child(
        self,
        exploration_weight: float,
    ) -> MCTSNode:
        """Select the child with the greatest UCT score."""
        if not self.children:
            raise ValueError(
                "Cannot select a child from a leaf node."
            )

        return max(
            self.children.values(),
            key=lambda child: child.uct_score(
                exploration_weight
            ),
        )

    def backpropagate(
        self,
        value: float,
    ) -> None:
        """Propagate a simulation result to this node and its ancestors."""
        node: MCTSNode | None = self

        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent
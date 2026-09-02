import math

import pytest

from pacman_rl.agents.mcts import MCTSNode
from pacman_rl.environment import Action


def test_new_node_has_no_search_statistics():
    node = MCTSNode(
        state=(0, 0),
    )

    assert node.visits == 0
    assert node.value_sum == 0.0
    assert node.mean_value == 0.0
    assert node.children == {}
    assert set(node.untried_actions) == set(Action)
    assert not node.is_fully_expanded


def test_add_child_expands_one_action():
    root = MCTSNode(
        state=(0, 0),
    )

    child = root.add_child(
        action=Action.RIGHT,
        state=(0, 1),
    )

    assert child.parent is root
    assert child.action == Action.RIGHT
    assert child.state == (0, 1)
    assert root.children[Action.RIGHT] is child
    assert Action.RIGHT not in root.untried_actions


def test_node_is_fully_expanded_after_all_actions():
    root = MCTSNode(
        state=(0, 0),
    )

    for action in list(Action):
        root.add_child(
            action=action,
            state=(0, 0),
        )

    assert root.is_fully_expanded
    assert len(root.children) == len(Action)


def test_cannot_expand_same_action_twice():
    root = MCTSNode(
        state=(0, 0),
    )

    root.add_child(
        action=Action.RIGHT,
        state=(0, 1),
    )

    with pytest.raises(
        ValueError,
        match="cannot be expanded",
    ):
        root.add_child(
            action=Action.RIGHT,
            state=(0, 1),
        )


def test_unvisited_child_has_infinite_uct_score():
    root = MCTSNode(
        state=(0, 0),
        visits=10,
    )

    child = root.add_child(
        action=Action.RIGHT,
        state=(0, 1),
    )

    assert math.isinf(
        child.uct_score(
            exploration_weight=math.sqrt(2),
        )
    )


def test_best_child_balances_value_and_exploration():
    root = MCTSNode(
        state=(0, 0),
        visits=20,
    )

    frequently_visited = root.add_child(
        action=Action.RIGHT,
        state=(0, 1),
    )
    frequently_visited.visits = 15
    frequently_visited.value_sum = 12.0

    rarely_visited = root.add_child(
        action=Action.DOWN,
        state=(1, 0),
    )
    rarely_visited.visits = 1
    rarely_visited.value_sum = 0.5

    selected = root.best_child(
        exploration_weight=math.sqrt(2),
    )

    assert selected is rarely_visited


def test_backpropagation_updates_all_ancestors():
    root = MCTSNode(
        state=(0, 0),
    )

    child = root.add_child(
        action=Action.RIGHT,
        state=(0, 1),
    )

    grandchild = child.add_child(
        action=Action.DOWN,
        state=(1, 1),
    )

    grandchild.backpropagate(
        value=10.0,
    )

    for node in (
        root,
        child,
        grandchild,
    ):
        assert node.visits == 1
        assert node.value_sum == 10.0
        assert node.mean_value == 10.0


def test_root_does_not_have_uct_score():
    root = MCTSNode(
        state=(0, 0),
    )

    with pytest.raises(
        ValueError,
        match="root node",
    ):
        root.uct_score(
            exploration_weight=1.0,
        )


def test_leaf_node_cannot_select_child():
    leaf = MCTSNode(
        state=(0, 0),
    )

    with pytest.raises(
        ValueError,
        match="leaf node",
    ):
        leaf.best_child(
            exploration_weight=1.0,
        )
"""Run a first Q-learning experiment on the reference grid."""

import numpy as np

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import (
    EvaluationResult,
    evaluate_q_learning,
)
from pacman_rl.training import (
    TrainingHistory,
    train_q_learning,
)


def run_experiment(
    training_episodes: int = 1500,
    evaluation_episodes: int = 200,
    seed: int = 42,
) -> tuple[
    TrainingHistory,
    EvaluationResult,
]:
    """Train and evaluate Q-learning on the reference grid."""
    env = PacmanEnv(
        random_start=True,
        seed=seed,
    )

    agent = QLearningAgent(
        n_states=env.rows * env.cols,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=seed,
    )

    history = train_q_learning(
        env=env,
        agent=agent,
        episodes=training_episodes,
    )

    metrics = evaluate_q_learning(
        env=env,
        agent=agent,
        episodes=evaluation_episodes,
    )

    return history, metrics


def main() -> None:
    """Run the experiment and print its main results."""
    history, metrics = run_experiment()

    recent_returns = history.episode_returns[-100:]
    recent_mean_return = float(np.mean(recent_returns))

    print("Q-learning experiment")
    print("---------------------")
    print(
        f"Mean training return over the last 100 episodes: "
        f"{recent_mean_return:.2f}"
    )
    print(f"Evaluation episodes: {metrics.episodes}")
    print(f"Success rate: {metrics.success_rate:.1%}")
    print(f"Ghost rate: {metrics.ghost_rate:.1%}")
    print(f"Timeout rate: {metrics.timeout_rate:.1%}")
    print(f"Mean return: {metrics.mean_return:.2f}")
    print(
        f"Return standard deviation: "
        f"{metrics.return_std:.2f}"
    )
    print(f"Mean episode length: {metrics.mean_steps:.2f}")
    print(
        f"Mean optimal length: "
        f"{metrics.mean_optimal_steps:.2f}"
    )
    print(
        f"Mean path efficiency: "
        f"{metrics.mean_efficiency_ratio:.1%}"
    )
    print(
        f"Optimal path rate: "
        f"{metrics.optimal_path_rate:.1%}"
    )


if __name__ == "__main__":
    main()
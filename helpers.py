import matplotlib.pyplot as plt
import numpy as np


def plot_results(x, y, save=True):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Mean Reward', color='blue')
    plt.xlabel('Steps')
    plt.ylabel('Mean Reward (20 Episodes)')
    plt.title('PPO Training Results')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig('figs/ppo_training_results.png')
    plt.show()


def plot_experiments(results: dict, save=True, filename='figs/ppo_experiment_comparison.png'):
    """Plot mean reward with ±1 std band for each named experiment.

    Args:
        results: {"label": {"x": array, "mean": array, "std": array}}
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(12, 6))

    for (label, data), color in zip(results.items(), colors):
        x    = data["x"]
        mean = data["mean"]
        std  = data["std"]
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Mean Reward (last 20 eps, 5 seeds)')
    ax.set_title('PPO Hyperparameter Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=150)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_results(x,y, save=True):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Mean Reward', color='blue')
    plt.xlabel('Steps')
    plt.ylabel('Mean Reward (20 Episodes)')
    plt.title('PPO Training Results')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig('figs//ppo_training_results.png')
    plt.show()
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np

from ppo_agent import ActorCritic, compute_gae, ppo_update
from environment import CartPoleEnv
from helpers import plot_results, plot_experiments

DEFAULTS = dict(
    horizon=2048, epochs=10, batch_size=64, gamma=0.99,
    lam=0.95, clip_eps=0.2, lr=2.5e-4, vf_cf=0.5,
    entropy_cf=0.01, hidden_size=64,
)

# optuna best (trial #61): final mean reward = 500.0, slower convergence
TRIAL_61 = dict(
    lr=5.817e-4, entropy_cf=1.504e-4,
    gamma=0.9508, lam=0.8538, clip_eps=0.1126, vf_cf=0.3430,
    epochs=10, horizon=2048, batch_size=32, hidden_size=32,
)

# trial #355: final mean reward = 497.87, converges fastest (~20k steps)
TRIAL_355 = dict(
    lr=9.119e-4, entropy_cf=2.431e-4,
    gamma=0.9860, lam=0.8038, clip_eps=0.1954, vf_cf=0.3412,
    epochs=15, horizon=512, batch_size=32, hidden_size=128,
)


def train_ppo(config: dict, seed: int = 0, total_steps: int = 500_000) -> tuple:
    """Train PPO with a given config and seed.

    Returns:
        (steps, mean_rewards): two parallel lists — env steps and
        rolling mean reward (last 20 episodes) at each rollout boundary.
    """
    cfg = {**DEFAULTS, **config}

    torch.manual_seed(seed)
    np.random.seed(seed)

    env     = CartPoleEnv(verbose=False, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_acts  = env.action_space.n
    model   = ActorCritic(obs_dim, n_acts, cfg["hidden_size"])
    opt     = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    episode_rewards   = []
    current_ep_reward = 0.0
    all_mean_rewards  = []
    state, _          = env.reset()
    global_step       = 0
    done              = False

    while global_step < total_steps:
        ro_states, ro_actions, ro_log_probs = [], [], []
        ro_rewards, ro_dones, ro_values     = [], [], []

        for _ in range(cfg["horizon"]):
            state_t = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, value = model.sample_action(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            ro_states.append(state_t)
            ro_actions.append(action)
            ro_log_probs.append(log_prob)
            ro_rewards.append(reward)
            ro_dones.append(float(done))
            ro_values.append(value.item())

            current_ep_reward += reward
            global_step       += 1

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                state, _ = env.reset()
            else:
                state = next_state

        with torch.no_grad():
            last_state_t    = torch.tensor(state, dtype=torch.float32)
            _, last_value_t = model.forward(last_state_t)
            last_value      = last_value_t.item() if not done else 0.0

        advantages, targets = compute_gae(
            ro_rewards, ro_values, ro_dones, last_value,
            cfg["gamma"], cfg["lam"],
        )
        states_t    = torch.stack(ro_states)
        actions_t   = torch.stack(ro_actions)
        log_probs_t = torch.stack(ro_log_probs).detach()

        ppo_update(
            model, opt,
            states_t, actions_t, log_probs_t,
            advantages, targets,
            cfg["clip_eps"], cfg["epochs"], cfg["batch_size"],
            cfg["vf_cf"], cfg["entropy_cf"],
        )

        if episode_rewards:
            mean_r = float(np.mean(episode_rewards[-20:]))
            all_mean_rewards.append((global_step, mean_r))

    env.close()
    steps, rewards = zip(*all_mean_rewards)
    return list(steps), list(rewards)


def run_experiment(
    config: dict,
    label: str = "",
    n_seeds: int = 5,
    total_steps: int = 500_000,
) -> dict:
    """Run config across n_seeds, interpolate to a common x-axis.

    Returns:
        {"x": array, "mean": array, "std": array}
    """
    x_common       = np.linspace(0, total_steps, 500)
    interp_rewards = []

    for seed in range(n_seeds):
        tag = f"[{label}] " if label else ""
        print(f"  {tag}seed {seed} / {n_seeds - 1} ...")
        steps, rewards = train_ppo(config, seed=seed, total_steps=total_steps)
        interp_rewards.append(np.interp(x_common, steps, rewards))

    arr = np.array(interp_rewards)   # (n_seeds, 500)
    return {"x": x_common, "mean": arr.mean(axis=0), "std": arr.std(axis=0)}


if __name__ == "__main__":
    EXPERIMENTS = {
        "Default":           DEFAULTS,
        "Trial #61 (best)":  TRIAL_61,
        "Trial #355 (fast)": TRIAL_355,
    }

    results = {}
    for label, cfg in EXPERIMENTS.items():
        print(f"\nRunning: {label}")
        results[label] = run_experiment(cfg, label=label, n_seeds=5)

    plot_experiments(results, save=True)

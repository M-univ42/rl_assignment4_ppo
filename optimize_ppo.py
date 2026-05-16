import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from ppo_agent import ActorCritic, compute_gae, ppo_update
from environment import CartPoleEnv

STUDY_NAME    = "ppo_cartpole"
STORAGE       = "sqlite:///ppo_optuna.sqlite3"
TRIAL_STEPS   = 200_000
EVAL_INTERVAL = 10_000


def objective(trial: optuna.Trial) -> float:
    lr         = trial.suggest_float("lr",         1e-5,  1e-3,  log=True)
    entropy_cf = trial.suggest_float("entropy_cf", 1e-4,  0.1,   log=True)

    gamma    = trial.suggest_float("gamma",    0.90, 0.999)
    lam      = trial.suggest_float("lam",      0.80, 0.99)
    clip_eps = trial.suggest_float("clip_eps", 0.05, 0.40)
    vf_cf    = trial.suggest_float("vf_cf",    0.25, 1.00)

    epochs = trial.suggest_int("epochs", 3, 15)

    horizon     = trial.suggest_categorical("horizon",     [512, 1024, 2048])
    batch_size  = trial.suggest_categorical("batch_size",  [32, 64, 128, 256])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    env     = CartPoleEnv(verbose=False)
    obs_dim = env.observation_space.shape[0]
    n_acts  = env.action_space.n
    model   = ActorCritic(obs_dim, n_acts, hidden_size)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)

    episode_rewards   = []
    current_ep_reward = 0.0
    state, _          = env.reset(seed=seed)
    global_step       = 0
    next_report       = EVAL_INTERVAL
    done              = False

    try:
        while global_step < TRIAL_STEPS:
            ro_states, ro_actions, ro_log_probs = [], [], []
            ro_rewards, ro_dones, ro_values     = [], [], []

            for _ in range(horizon):
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

                if global_step >= next_report and episode_rewards:
                    window      = episode_rewards[-20:]
                    pruner_step = global_step // EVAL_INTERVAL
                    trial.report(float(np.mean(window)), pruner_step)
                    next_report += EVAL_INTERVAL
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            with torch.no_grad():
                last_state_t      = torch.tensor(state, dtype=torch.float32)
                _, last_value_t   = model.forward(last_state_t)
                last_value        = last_value_t.item() if not done else 0.0

            advantages, targets = compute_gae(
                ro_rewards, ro_values, ro_dones, last_value, gamma, lam
            )
            states_t    = torch.stack(ro_states)
            actions_t   = torch.stack(ro_actions)
            log_probs_t = torch.stack(ro_log_probs).detach()

            ppo_update(
                model, opt,
                states_t, actions_t, log_probs_t,
                advantages, targets,
                clip_eps, epochs, batch_size, vf_cf, entropy_cf,
            )

    finally:
        env.close()

    # objective is the mean reward over last 100 completed episodes
    window = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    return float(np.mean(window)) if window else 0.0


if __name__ == "__main__":
    sampler = TPESampler(multivariate=True, seed=42)

    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


    study.optimize(
        objective,
        n_trials=None,
        timeout=None,
        show_progress_bar=True,
        gc_after_trial=True,
    )
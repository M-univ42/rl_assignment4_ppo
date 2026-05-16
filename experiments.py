
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym

from ppo_agent import ActorCritic, compute_gae, ppo_update
from environment import CartPoleEnv
from helpers import plot_results
def train_ppo(total_steps = 500000,horizon=2048,epochs= 10,batch_size  = 64,gamma= 0.99,lam= 0.95,clip_eps= 0.2,lr = 2.5e-4,vf_cf = 0.5,entropy_cf = 0.01,hidden_size = 64,seed= 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    env     = CartPoleEnv()
    obs_dim = env.observation_space.shape[0]
    n_acts  = env.action_space.n
 
    model     = ActorCritic(obs_dim, n_acts, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    episode_rewards = []   
    current_ep_reward = 0
    all_mean_rewards = []
 
    state, _ = env.reset(seed=seed)
    global_step = 0
 
    while global_step < total_steps:
        ro_states    = []
        ro_actions   = []
        ro_log_probs = []
        ro_rewards   = []
        ro_dones     = []
        ro_values    = []
 
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
            global_step += 1
 
            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
                state, _ = env.reset()
            else:
                state = next_state
 

        with torch.no_grad():
            last_state_t = torch.tensor(state, dtype=torch.float32)
            _, last_value = model.forward(last_state_t)
            last_value = last_value.item() if not done else 0.0
 
       
        advantages, targets = compute_gae(
            ro_rewards, ro_values, ro_dones,
            last_value, gamma, lam
        )
 
       
        states_t = torch.stack(ro_states)
        actions_t = torch.stack(ro_actions)
        log_probs_t = torch.stack(ro_log_probs).detach()  
 
   
        ppo_update(
            model, optimizer,states_t, actions_t,log_probs_t,
            advantages, targets,clip_eps, epochs,batch_size, vf_cf, entropy_cf
        )
 
  
        if len(episode_rewards) > 0:
            # avg over last 20 episodes
            mean_r = np.mean(episode_rewards[-20:])
            all_mean_rewards.append((global_step, mean_r))
            if global_step % 10000 < horizon:
                last_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                mean_r = np.mean(last_100)
                print(f"Step {global_step} -- mean reward (last 100 eps): {mean_r:.1f}")
 
    env.close()
    return all_mean_rewards
 
 
if __name__ == "__main__":
    results = train_ppo()
    steps, mean_rewards = zip(*results)
    plot_results(steps, mean_rewards)
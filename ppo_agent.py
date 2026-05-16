import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden_size, n_actions) 
        self.critic = nn.Linear(hidden_size, 1)       
 
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
 
    def sample_action(self, state):
        """Samples a given action and returns(action, log_prob, value).
        INPUT: state (tensor)
        
        OUTPUT: action, log_prob, value"""
        logits, value = self.forward(state)
        dist = Categorical(logits= logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
 
    def evaluate(self, states, actions):
        """Evbaluates action based on policy
        INPUT: states, actions
        OUTPUT: log_probs, values, entropy
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
    

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Calculates GAE based on paper formula.
    INPUT: rewards,values, dones (for full ep. ), last_value, gamma, lambda

    OUTPUT: advatnages, targets
    """
    advantages = []
    gae        = 0.0
    values_np = values+ [last_value]
 

    for t in reversed(range(len(rewards))):
        not_done = 1.0-dones[t]
        delta    = rewards[t] + gamma*values_np[t+1]*not_done - values_np[t]
        gae      = delta + gamma*lam * not_done*gae
        advantages.insert(0,gae)
 
    advantages = torch.tensor(advantages, dtype=torch.float32)
    targets= advantages + torch.tensor(values,dtype=torch.float32)
    return advantages, targets

def ppo_update(model, optimizer, states, actions, old_log_probs,
               advantages, targets,
               clip_eps=0.2, epochs=4, batch_size=64,
               vf_cf=0.5, entropy_cf=0.01):
    """
   Updates agent based on PPO paper (with epsilon=0.2)
   INPUTS: model, optimizer, states, actions, old_log_probs, clip_eps, epochs, batch, coeff, entropy_coeff
    """
    T = len(states)
 
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
    for _ in range(epochs):
        indices = np.random.permutation(T)
 
        for start in range(0, T, batch_size):
            mb_idx = indices[start : start + batch_size]
 
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_prob = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_targets= targets[mb_idx]
 
            # ealuate actions for the current given policy
            new_log_probs, values, entropy = model.evaluate(mb_states,mb_actions)
 
            ratio = torch.exp(new_log_probs-mb_old_log_prob)
 
            # CLIPPED formula from paper
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
 
            # critic loss
            critic_loss = nn.functional.mse_loss(values, mb_targets)
 
            # entropy bonus (max)
            entropy_loss = -entropy.mean() 
 
            # overall loss
            loss = actor_loss + vf_cf * critic_loss + entropy_cf * entropy_loss
 
            optimizer.zero_grad()
            loss.backward()
            # gradient clip
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
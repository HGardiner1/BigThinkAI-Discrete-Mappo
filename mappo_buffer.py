import torch

class MAPPOBuffer:
    def __init__(self, steps, n_agents, obs_dim, joint_dim, action_dim, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

        self.obs = torch.zeros((steps, n_agents, obs_dim))
        self.joint_obs = torch.zeros((steps, joint_dim))
        self.actions = torch.zeros((steps, n_agents), dtype=torch.long)
        self.rewards = torch.zeros((steps, n_agents))
        self.dones = torch.zeros((steps, 1))

        self.values = torch.zeros((steps, n_agents))
        self.log_probs = torch.zeros((steps, n_agents))

        self.ptr = 0
        self.max = steps

    def store(self, obs, joint_obs, actions, rewards, dones, values, log_probs):
        t = self.ptr
        if t >= self.max:
            # Reset pointer when buffer is full
            self.ptr = 0
            t = 0
        self.obs[t] = obs
        self.joint_obs[t] = joint_obs
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.log_probs[t] = log_probs
        self.ptr += 1

    def compute_returns(self, critic):
        advantages = torch.zeros_like(self.rewards)
        last_adv = torch.zeros(self.rewards.size(1))

        for t in reversed(range(self.max)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * mask * self.values[t] - self.values[t]
            advantages[t] = last_adv = delta + self.gamma * self.lam * mask * last_adv

        returns = advantages + self.values
        return advantages, returns

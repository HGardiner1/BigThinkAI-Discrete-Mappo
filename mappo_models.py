import torch
import torch.nn as nn
import torch.nn.functional as F


# DECENTRALIZED ACTOR: 
class Actor(nn.Module):
    """Decentralized actor for each agent (policy network)."""
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits


# CENTRAL CRITIC: 
# has access to all agents' states and actions for a more accurate value estimation
class CentralCritic(nn.Module):
    def __init__(self, joint_obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

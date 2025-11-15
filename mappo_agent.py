import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from mappo_models import CentralCritic
from mappo_models import Actor

class MAPPOAgent:
    def __init__(self, n_agents, obs_dim, action_dim, joint_obs_dim,
                 actor_lr=3e-4, critic_lr=3e-4, clip_ratio=0.2):

        self.n_agents = n_agents
        self.clip_ratio = clip_ratio

        # Build actors (one per agent)
        self.actors = torch.nn.ModuleList([
            Actor(obs_dim, action_dim) for _ in range(n_agents)
        ])
        self.critic = CentralCritic(joint_obs_dim)

        self.actor_optim = torch.optim.Adam(self.actors.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # --- ACTION SELECTION ---
    def act(self, obs_dict, agent_order):
        """
        obs_dict: dictionary from env
        agent_order: list of agents in order corresponding to actors
        """
        obs_array = np.array([obs_dict[a] for a in agent_order], dtype=np.float32)
        obs = torch.from_numpy(obs_array)

        actions = []
        log_probs = []
        q_tables = {}
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                logits = actor(obs[i])
                dist = Categorical(logits=logits)
                a = dist.sample()
                actions.append(a)
                log_probs.append(dist.log_prob(a))
                # store logits as a proxy for Q-values (caller/visualizer can interpret)
                try:
                    q_tables[agent_order[i]] = logits.detach().cpu().numpy()
                except Exception:
                    q_tables[agent_order[i]] = np.array(logits.detach().cpu())

        return (
            {agent_order[i]: actions[i].item() for i in range(len(actions))},
            torch.stack(log_probs),
            q_tables
        )

    # --- POLICY UPDATE ---
    def update(self, buffer, epochs=4, batch_size=64):
        advantages, returns = buffer.compute_returns(self.critic)

        for _ in range(epochs):
            for start in range(0, buffer.max, batch_size):
                end = start + batch_size

                obs = buffer.obs[start:end]
                joint_obs = buffer.joint_obs[start:end]
                actions = buffer.actions[start:end]
                old_log_probs = buffer.log_probs[start:end]
                adv = advantages[start:end]
                ret = returns[start:end]

                # CRITIC UPDATE
                values = self.critic(joint_obs)
                critic_loss = F.mse_loss(values.squeeze(), ret.mean(dim=1))
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # ACTOR UPDATE (each agent separately)
                actor_loss_total = 0
                for i, actor in enumerate(self.actors):
                    logits = actor(obs[:, i])
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(actions[:, i])

                    ratio = torch.exp(logp - old_log_probs[:, i])
                    surr1 = ratio * adv[:, i]
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv[:, i]

                    actor_loss = -(torch.min(surr1, surr2)).mean()
                    actor_loss_total += actor_loss

                self.actor_optim.zero_grad()
                actor_loss_total.backward()
                self.actor_optim.step()

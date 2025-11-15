import numpy as np
from warehouse_mappo import WarehouseEnv
from mappo_agent import MAPPOAgent
from mappo_buffer import MAPPOBuffer
import torch
from visualization import WarehouseVisualizer
import pygame

EPISODES = 50
STEPS = 1000

# Set to True to open pygame visualizer (requires display)
RENDER = True

# Create environment with matching max_steps
env = WarehouseEnv(max_steps=STEPS)

# Visualizer lives in the trainer; double FPS to make it run "twice as fast"
visualizer = None
if RENDER:
    try:
        visualizer = WarehouseVisualizer(env, cell_size=50, fps=10)
    except Exception:
        visualizer = None
agent_order = env.possible_agents
obs, _ = env.reset()

obs_dim = 6
action_dim = 7
joint_obs_dim = obs_dim * len(agent_order)

mappo = MAPPOAgent(
    n_agents=len(agent_order),
    obs_dim=obs_dim,
    action_dim=action_dim,
    joint_obs_dim=joint_obs_dim
)

buffer = MAPPOBuffer(
    steps=STEPS,
    n_agents=len(agent_order),
    obs_dim=obs_dim,
    joint_dim=joint_obs_dim,
    action_dim=action_dim
)

for ep in range(EPISODES):
    obs, _ = env.reset()

    for step in range(STEPS):
        # Build joint obs
        joint_obs = np.concatenate([obs[a] for a in agent_order])

        # Select actions and get agent Q-tables (logits proxy)
        actions, log_probs, q_tables = mappo.act(obs, agent_order)

        # Step env
        next_obs, rewards, terms, truncs, infos = env.step(actions)

        # Render (if visualizer provided)
        if visualizer is not None:
            visualizer.render()
            # handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # stop the episode and training loop
                    env.agents = []
                    break

        # Store transition
        buffer.store(
            torch.from_numpy(np.array([obs[a] for a in agent_order], dtype=np.float32)),
            torch.from_numpy(joint_obs.astype(np.float32)),
            torch.from_numpy(np.array([actions[a] for a in agent_order], dtype=np.int64)),
            torch.from_numpy(np.array([rewards[a] for a in agent_order], dtype=np.float32)),
            torch.tensor([any(terms.values())], dtype=torch.float32),
            torch.tensor([0], dtype=torch.float32),    # placeholder critic value
            log_probs
        )

        # Expose current agent Q-tables to the environment/visualizer
        try:
            env.log_state(agent_q_tables=q_tables, episode=ep)
        except Exception:
            # don't fail training if visualization logging isn't available
            pass

        obs = next_obs
        if any(terms.values()):
            break

    # Update MAPPO after 1 episode
    mappo.update(buffer)
    print(f"Episode {ep} finished.")

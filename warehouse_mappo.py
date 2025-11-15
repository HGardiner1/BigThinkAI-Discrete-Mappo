import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv
from log import QTableLogger


# ----- CONSTANTS -----
EMPTY = 0
TARGET = 1
DEPOT = 2
BAY = 3

ACTION_STAY  = 0
ACTION_UP    = 1
ACTION_DOWN  = 2
ACTION_LEFT  = 3
ACTION_RIGHT = 4
ACTION_PICK  = 5
ACTION_DROP  = 6


class WarehouseEnv(ParallelEnv):
    """
    Very simple discrete multi-robot warehouse for MAPPO.
    - Multiple robots (agents)
    - Targets to pick up
    - One depot to drop at
    - One bay where robots start
    """

    metadata = {
        "name": "warehouse_mappo_simple_v0",
        "render_modes": ["human"],
    }

    def __init__(self, grid_size=(6, 6), n_robots=3, n_targets=3, max_steps=50, render=False, render_fps=10):
        self.grid_h, self.grid_w = grid_size
        self.n_robots = n_robots
        self.n_targets = n_targets
        self.max_steps = max_steps
        # render flags are deprecated here; visualizer is managed by the trainer

        self.possible_agents = [f"robot_{i}" for i in range(n_robots)]
        self.agents = self.possible_agents[:]

        # Discrete action space: 7 actions
        self.action_spaces = {
            agent: spaces.Discrete(7) for agent in self.possible_agents
        }

        # Observation: (row, col, carrying_flag) for this robot +
        #              (row, col) for depot
        #              number of targets remaining (scalar)
        # -> size = 3 + 2 + 1 = 6
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.observation_spaces = {
            agent: self._obs_space for agent in self.possible_agents
        }

        self.grid = None
        self.robot_positions = {}
        self.robot_carry = {}
        self.depot_pos = None
        self.bay_pos = None
        self.step_count = 0
        # Track deliveries for visualization/termination
        self.completed_deliveries = 0
        self.initial_targets = n_targets
        # Last-step diagnostics for visualization
        self.last_step_rewards = {a: 0.0 for a in self.possible_agents}
        self.last_q_values = None
        # Track state-action pairs for episode summary heatmap
        self.episode_actions = {a: [] for a in self.possible_agents}
        self.episode_states = {a: [] for a in self.possible_agents}
        
        # Initialize Q-table logger
        self.logger = QTableLogger(log_dir="logs", log_filename="q_tables.txt")

        # visualizer moved to training script (create WarehouseVisualizer there)

    # ------------------------------------------------------------------ #
    #  Required PettingZoo methods                                       #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        # Reset episode tracking for state-action heatmap
        self.episode_actions = {a: [] for a in self.possible_agents}
        self.episode_states = {a: [] for a in self.possible_agents}

        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=int)

        # Place depot
        dr, dc = np.random.randint(self.grid_h), np.random.randint(self.grid_w)
        self.grid[dr, dc] = DEPOT
        self.depot_pos = (dr, dc)

        # Place bay somewhere else
        while True:
            br, bc = np.random.randint(self.grid_h), np.random.randint(self.grid_w)
            if (br, bc) != (dr, dc):
                break
        self.grid[br, bc] = BAY
        self.bay_pos = (br, bc)

        # Place targets on empty cells
        placed = 0
        while placed < self.n_targets:
            r, c = np.random.randint(self.grid_h), np.random.randint(self.grid_w)
            if self.grid[r, c] == EMPTY:
                self.grid[r, c] = TARGET
                placed += 1

        # Robots start at bay
        self.robot_positions = {}
        self.robot_carry = {}
        for i, agent in enumerate(self.agents):
            self.robot_positions[agent] = (br, bc)
            self.robot_carry[agent] = 0

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        if len(self.agents) == 0:
            return {}, {}, {}, {}, {}

        self.step_count += 1

        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Move phase
        new_positions = {}
        previous_positions = set(self.robot_positions.values())

        for agent, act in actions.items():
            r, c = self.robot_positions[agent]

            # simulate move
            nr, nc = r, c
            if act == ACTION_UP:
                nr = max(0, r - 1)
            elif act == ACTION_DOWN:
                nr = min(self.grid_h - 1, r + 1)
            elif act == ACTION_LEFT:
                nc = max(0, c - 1)
            elif act == ACTION_RIGHT:
                nc = min(self.grid_w - 1, c + 1)

            # blocking rule: cannot move into an occupied cell
            if (nr, nc) in previous_positions or (nr, nc) in new_positions.values():
                # penalize blocked attempt
                rewards[agent] -= 0.05
                infos[agent]["blocked"] = True
                new_positions[agent] = (r, c)   # remain in place
            else:
                new_positions[agent] = (nr, nc)

        # AFTER loop, commit new positions
        self.robot_positions = new_positions

        # Pick / drop + rewards
        for agent, act in actions.items():
            r, c = self.robot_positions[agent]
            cell = self.grid[r, c]

            # small time penalty
            rewards[agent] -= 0.01
            
            # Track state-action for episode heatmap
            state_id = r * self.grid_w + c  # simple state encoding: position
            self.episode_actions[agent].append(int(act))
            self.episode_states[agent].append(state_id)

            if act == ACTION_PICK:
                if cell == TARGET and self.robot_carry[agent] == 0:
                    self.robot_carry[agent] = 1
                    self.grid[r, c] = EMPTY
                    rewards[agent] += 1.0

            if act == ACTION_DROP:
                if cell == DEPOT and self.robot_carry[agent] == 1:
                    self.robot_carry[agent] = 0
                    rewards[agent] += 5.0
                    # count completed delivery
                    self.completed_deliveries += 1

        # done if all targets removed or max_steps hit
        no_targets_left = not (self.grid == TARGET).any()
        done = no_targets_left or (self.step_count >= self.max_steps)
        if done:
            for a in self.agents:
                terminations[a] = True

        obs = {a: self._get_obs(a) for a in self.agents}

        # Save last-step rewards for the visualizer / external monitoring
        # Use a dict with entries for all possible agents to keep UI stable
        self.last_step_rewards = {a: float(rewards.get(a, 0.0)) for a in self.possible_agents}

        if done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #

    def _get_obs(self, agent):
        """Simple normalized feature vector for one robot."""
        r, c = self.robot_positions[agent]
        dr, dc = self.depot_pos
        carrying = self.robot_carry[agent]
        targets_left = int((self.grid == TARGET).sum())

        # normalize by grid size / max values
        obs = np.array([
            r / (self.grid_h - 1),
            c / (self.grid_w - 1),
            float(carrying),
            dr / (self.grid_h - 1),
            dc / (self.grid_w - 1),
            targets_left / max(1, self.n_targets),
        ], dtype=np.float32)

        return obs

    def render(self):
        """Text render fallback for debugging (visualizer is external)."""
        grid_copy = self.grid.copy()
        for agent, (r, c) in self.robot_positions.items():
            grid_copy[r, c] = 9  # mark robots as 9
        print("Step:", self.step_count)
        print(grid_copy)
        print("carry:", self.robot_carry)
        print()

    def log_state(self, central_q_table=None, agent_q_tables=None, episode=None):
        """
        Log the current state and Q-tables.
        
        Args:
            central_q_table: Q-table from central agent (optional)
            agent_q_tables: Dictionary mapping agent names to Q-tables (optional)
            episode: Episode number (optional)
        """
        self.logger.log_q_tables(
            central_q_table=central_q_table,
            agent_q_tables=agent_q_tables,
            episode=episode,
            step=self.step_count
        )
        # Expose last-known Q-values to the visualizer (optional)
        if agent_q_tables is not None:
            # keep as-is (caller may pass dict of numpy arrays)
            self.last_q_values = agent_q_tables
        elif central_q_table is not None:
            self.last_q_values = {'central': central_q_table}

    def visualize_q_tables(self, central_q_table=None, agent_q_tables=None, 
                           episode=None, save_dir="logs/visualizations"):
        """
        Visualize Q-table heatmaps for central and decentralized agents.
        
        Args:
            central_q_table: Q-table from central agent (optional)
            agent_q_tables: Dictionary mapping agent names to Q-tables (optional)
            episode: Episode number (optional)
            save_dir: Directory to save visualizations
        """
        self.logger.visualize_all_agents_heatmaps(
            agent_q_tables=agent_q_tables,
            central_q_table=central_q_table,
            episode=episode,
            step=self.step_count,
            save_dir=save_dir
        )

    def visualize_q_grids(self, central_q_table=None, agent_q_tables=None, 
                          episode=None, save_dir="logs/visualizations"):
        """
        Visualize grid representations of max Q-values for all agents.
        
        Args:
            central_q_table: Q-table from central agent (optional)
            agent_q_tables: Dictionary mapping agent names to Q-tables (optional)
            episode: Episode number (optional)
            save_dir: Directory to save visualizations
        """
        if central_q_table is not None:
            self.logger.visualize_q_table_grid(
                central_q_table, "Central_Agent", episode, self.step_count, save_dir
            )
        
        if agent_q_tables:
            for agent_name, q_table in agent_q_tables.items():
                self.logger.visualize_q_table_grid(
                    q_table, agent_name, episode, self.step_count, save_dir
                )

    def plot_q_statistics(self, central_q_table=None, agent_q_tables=None, 
                          episode=None, save_dir="logs/visualizations"):
        """
        Plot Q-value statistics (mean, min, max, std) across all agents.
        
        Args:
            central_q_table: Q-table from central agent (optional)
            agent_q_tables: Dictionary mapping agent names to Q-tables (optional)
            episode: Episode number (optional)
            save_dir: Directory to save visualization
        """
        self.logger.plot_q_value_statistics(
            agent_q_tables=agent_q_tables,
            central_q_table=central_q_table,
            episode=episode,
            step=self.step_count,
            save_dir=save_dir
        )

    def save_episode_state_action_heatmap(self, episode=None, save_dir="logs/visualizations"):
        """
        Save a condensed heatmap of state-action frequencies for visited states only.
        
        Args:
            episode: Episode number (optional)
            save_dir: Directory to save heatmap
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(save_dir, exist_ok=True)
        
        # For each agent, create a heatmap of state x action frequencies
        for agent_name in self.possible_agents:
            actions = self.episode_actions.get(agent_name, [])
            states = self.episode_states.get(agent_name, [])
            
            if not actions or not states:
                continue
            
            # Create a condensed heatmap with only visited states (much simpler!)
            unique_states = sorted(set(states))
            n_visited_states = len(unique_states)
            n_actions = 7
            
            # Map state_id to row index in heatmap
            state_to_row = {state_id: i for i, state_id in enumerate(unique_states)}
            heatmap = np.zeros((n_visited_states, n_actions))
            
            for state_id, action in zip(states, actions):
                if state_id in state_to_row and 0 <= action < n_actions:
                    heatmap[state_to_row[state_id], action] += 1
            
            # Create figure with grid coordinates as labels for clarity
            fig, ax = plt.subplots(figsize=(10, max(4, n_visited_states * 0.3)))
            sns.heatmap(heatmap, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frequency'})
            ax.set_xlabel('Action', fontsize=11)
            ax.set_ylabel('State (col, row)', fontsize=11)
            ax.set_title(f'State-Action Heatmap for {agent_name} (Episode {episode}) - {n_visited_states} visited states')
            
            # Create state labels with grid coordinates (col, row) for readability
            state_labels = [f"({s % self.grid_w},{s // self.grid_w})" for s in unique_states]
            ax.set_yticklabels(state_labels, rotation=0, fontsize=9)
            ax.set_xticklabels(['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP'], rotation=45)
            
            # Save
            ep_str = f"_ep{episode}" if episode is not None else ""
            filename = f"{save_dir}/{agent_name}_state_action_heatmap{ep_str}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            plt.close()
            print(f"[HEATMAP] Saved to {filename} ({n_visited_states} states visited)")


# Small manual test: python warehouse_env_simple.py
if __name__ == "__main__":
    env = WarehouseEnv()
    obs, infos = env.reset()
    for _ in range(50):
        actions = {a: env.action_spaces[a].sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        if all(terms.values()):
            break
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

    def __init__(self, grid_size=(9, 9), n_robots=3, n_targets=3, max_steps=50):
        self.grid_h, self.grid_w = grid_size
        self.n_robots = n_robots
        self.n_targets = n_targets
        self.max_steps = max_steps

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
        
        # Initialize Q-table logger
        self.logger = QTableLogger(log_dir="logs", log_filename="q_tables.txt")

    # ------------------------------------------------------------------ #
    #  Required PettingZoo methods                                       #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0

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

            if act == ACTION_PICK:
                if cell == TARGET and self.robot_carry[agent] == 0:
                    self.robot_carry[agent] = 1
                    self.grid[r, c] = EMPTY
                    rewards[agent] += 1.0

            if act == ACTION_DROP:
                if cell == DEPOT and self.robot_carry[agent] == 1:
                    self.robot_carry[agent] = 0
                    rewards[agent] += 5.0

        # done if all targets removed or max_steps hit
        no_targets_left = not (self.grid == TARGET).any()
        done = no_targets_left or (self.step_count >= self.max_steps)
        if done:
            for a in self.agents:
                terminations[a] = True

        obs = {a: self._get_obs(a) for a in self.agents}

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
        """Very simple text render for debugging."""
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

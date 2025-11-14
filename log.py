import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class QTableLogger:
    """
    Handles logging for centralized and decentralized Q-tables.
    Logs are written in a human-readable format and appended over time.
    """

    def __init__(self, log_dir="logs", log_filename="q_tables.txt"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)

        # Create directory if needed
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log file header if new
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("==== Q-TABLE LOG ====\n\n")

    # --------------------------------------------------------------- #

    def _format_q_table(self, q_table):
        """
        Convert Q-table into nicely formatted text.
        Works for dict, numpy array, or nested dict.
        """
        if q_table is None:
            return "None\n"

        # Numpy array Q-table
        if isinstance(q_table, np.ndarray):
            return np.array2string(q_table, precision=3, floatmode="fixed") + "\n"

        # Python dict Q-table
        if isinstance(q_table, dict):
            formatted = ""
            for state, values in q_table.items():
                formatted += f"  State {state}: {values}\n"
            return formatted

        return str(q_table) + "\n"

    # --------------------------------------------------------------- #

    def log_q_tables(self, central_q_table=None, agent_q_tables=None,
                     episode=None, step=None):

        with open(self.log_path, "a") as f:
            f.write("------------------------------------------------------------\n")

            # Episode / step metadata
            if episode is not None:
                f.write(f"Episode: {episode}\n")
            if step is not None:
                f.write(f"Step: {step}\n")

            # Central agent logging
            f.write("\n=== Centralized Q-Table ===\n")
            f.write(self._format_q_table(central_q_table))

            # Decentralized agent logging
            f.write("\n=== Per-Agent Q-Tables ===\n")
            if agent_q_tables:
                for agent, q_table in agent_q_tables.items():
                    f.write(f"\n[{agent}] Q-Table:\n")
                    f.write(self._format_q_table(q_table))
            else:
                f.write("None\n")

            f.write("\n")  # spacing

        print(f"[LOGGER] Q-tables logged to: {self.log_path}")

    # --------------------------------------------------------------- #

    def visualize_q_table_heatmap(self, q_table, agent_name="Q-Table", 
                                   episode=None, step=None, save_dir="logs/visualizations"):
        """
        Create a heatmap visualization of Q-values using seaborn.
        
        Args:
            q_table (np.ndarray): Q-table to visualize (must be 2D: states x actions)
            agent_name (str): Name of the agent for title
            episode (int): Episode number for filename
            step (int): Step number for filename
            save_dir (str): Directory to save visualization
        """
        if q_table is None:
            return
        
        # Only works with numpy arrays
        if not isinstance(q_table, np.ndarray):
            print(f"[WARNING] Cannot visualize non-array Q-table for {agent_name}")
            return
        
        # Ensure 2D
        if q_table.ndim != 2:
            print(f"[WARNING] Q-table must be 2D for heatmap visualization")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create heatmap
        sns.heatmap(q_table, annot=True, fmt='.3f', cmap='viridis', 
                    cbar_kws={'label': 'Q-Value'}, linewidths=0.5)
        
        # Title and labels
        title = f"{agent_name} Q-Table Heatmap"
        if episode is not None:
            title += f" (Episode {episode}"
            if step is not None:
                title += f", Step {step}"
            title += ")"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Actions', fontsize=12)
        plt.ylabel('States', fontsize=12)
        
        # Generate filename
        filename = f"{agent_name.lower().replace(' ', '_')}"
        if episode is not None:
            filename += f"_ep{episode}"
        if step is not None:
            filename += f"_step{step}"
        filename += ".png"
        
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[LOGGER] Heatmap saved: {filepath}")
    
    # --------------------------------------------------------------- #

    def visualize_all_agents_heatmaps(self, agent_q_tables, central_q_table=None,
                                       episode=None, step=None, save_dir="logs/visualizations"):
        """
        Visualize Q-table heatmaps for all agents and central agent.
        
        Args:
            agent_q_tables (dict): Dictionary mapping agent names to Q-tables
            central_q_table: Central agent's Q-table
            episode (int): Episode number
            step (int): Step number
            save_dir (str): Directory to save visualizations
        """
        # Visualize central agent
        if central_q_table is not None:
            self.visualize_q_table_heatmap(central_q_table, "Central_Agent", 
                                          episode, step, save_dir)
        
        # Visualize all decentralized agents
        if agent_q_tables:
            for agent_name, q_table in agent_q_tables.items():
                self.visualize_q_table_heatmap(q_table, agent_name, 
                                              episode, step, save_dir)
    
    # --------------------------------------------------------------- #

    def visualize_q_table_grid(self, q_table, agent_name="Q-Table", 
                                episode=None, step=None, save_dir="logs/visualizations"):
        """
        Create a grid visualization showing max Q-values at each state.
        Useful for 2D grid worlds to visualize agent policy/values spatially.
        
        Args:
            q_table (np.ndarray): Q-table to visualize
            agent_name (str): Name of the agent
            episode (int): Episode number
            step (int): Step number
            save_dir (str): Directory to save visualization
        """
        if q_table is None:
            return
        
        if not isinstance(q_table, np.ndarray):
            print(f"[WARNING] Cannot visualize non-array Q-table for {agent_name}")
            return
        
        if q_table.ndim != 2:
            print(f"[WARNING] Q-table must be 2D for grid visualization")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get max Q-value for each state
        max_q_values = np.max(q_table, axis=1)
        
        # Try to reshape into grid if possible (for grid worlds)
        grid_size = int(np.sqrt(len(max_q_values)))
        if grid_size * grid_size == len(max_q_values):
            grid_values = max_q_values.reshape((grid_size, grid_size))
        else:
            # Fallback: just visualize as 1D reshaped
            grid_values = max_q_values.reshape((len(max_q_values), 1))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(grid_values, annot=True, fmt='.3f', cmap='RdYlGn',
                    cbar_kws={'label': 'Max Q-Value'}, linewidths=1)
        
        title = f"{agent_name} Max Q-Values Grid"
        if episode is not None:
            title += f" (Episode {episode}"
            if step is not None:
                title += f", Step {step}"
            title += ")"
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        filename = f"{agent_name.lower().replace(' ', '_')}_grid"
        if episode is not None:
            filename += f"_ep{episode}"
        if step is not None:
            filename += f"_step{step}"
        filename += ".png"
        
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[LOGGER] Grid visualization saved: {filepath}")
    
    # --------------------------------------------------------------- #

    def plot_q_value_statistics(self, agent_q_tables, central_q_table=None,
                                 episode=None, step=None, save_dir="logs/visualizations"):
        """
        Plot statistics of Q-values across all agents (mean, min, max, std).
        
        Args:
            agent_q_tables (dict): Dictionary mapping agent names to Q-tables
            central_q_table: Central agent's Q-table
            episode (int): Episode number
            step (int): Step number
            save_dir (str): Directory to save visualization
        """
        os.makedirs(save_dir, exist_ok=True)
        
        stats = {}
        agent_names = []
        
        # Collect central agent stats
        if central_q_table is not None and isinstance(central_q_table, np.ndarray):
            agent_names.append("Central")
            stats["Central"] = {
                'mean': np.mean(central_q_table),
                'min': np.min(central_q_table),
                'max': np.max(central_q_table),
                'std': np.std(central_q_table)
            }
        
        # Collect agent stats
        if agent_q_tables:
            for agent_name, q_table in agent_q_tables.items():
                if isinstance(q_table, np.ndarray):
                    agent_names.append(agent_name)
                    stats[agent_name] = {
                        'mean': np.mean(q_table),
                        'min': np.min(q_table),
                        'max': np.max(q_table),
                        'std': np.std(q_table)
                    }
        
        if not stats:
            print("[WARNING] No valid Q-tables to plot statistics")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Q-Value Statistics across Agents (Episode {episode}, Step {step})', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['mean', 'min', 'max', 'std']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
            values = [stats[agent][metric] for agent in agent_names]
            ax.bar(agent_names, values, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric.upper()} Q-Values', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate labels if many agents
            if len(agent_names) > 3:
                ax.set_xticklabels(agent_names, rotation=45, ha='right')
        
        filename = f"q_value_statistics"
        if episode is not None:
            filename += f"_ep{episode}"
        if step is not None:
            filename += f"_step{step}"
        filename += ".png"
        
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[LOGGER] Statistics plot saved: {filepath}")

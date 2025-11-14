#!/usr/bin/env python3
"""
Test script for Q-table visualization and logging.
Demonstrates all visualization features.
"""

import numpy as np
from log import QTableLogger

# Initialize logger
logger = QTableLogger(log_dir="logs", log_filename="q_tables.txt")

# Create sample Q-tables (states x actions)
central_q_table = np.random.randn(16, 7).astype(np.float32) * 10

agent_q_tables = {
    "robot_0": np.random.randn(16, 7).astype(np.float32) * 10,
    "robot_1": np.random.randn(16, 7).astype(np.float32) * 10,
    "robot_2": np.random.randn(16, 7).astype(np.float32) * 10,
}

print("=" * 60)
print("Q-TABLE VISUALIZATION AND LOGGING TEST")
print("=" * 60)

# Test 1: Log Q-tables to text file
print("\n[1] Logging Q-tables to text file...")
logger.log_q_tables(
    central_q_table=central_q_table,
    agent_q_tables=agent_q_tables,
    episode=1,
    step=50
)

# Test 2: Visualize heatmaps for all agents
print("\n[2] Generating heatmap visualizations...")
logger.visualize_all_agents_heatmaps(
    agent_q_tables=agent_q_tables,
    central_q_table=central_q_table,
    episode=1,
    step=50,
    save_dir="logs/visualizations"
)

# Test 3: Visualize grid representations
print("\n[3] Generating grid visualizations...")
logger.visualize_q_table_grid(
    central_q_table, "Central_Agent", episode=1, step=50, 
    save_dir="logs/visualizations"
)
for agent_name, q_table in agent_q_tables.items():
    logger.visualize_q_table_grid(
        q_table, agent_name, episode=1, step=50, 
        save_dir="logs/visualizations"
    )

# Test 4: Plot Q-value statistics
print("\n[4] Generating Q-value statistics plots...")
logger.plot_q_value_statistics(
    agent_q_tables=agent_q_tables,
    central_q_table=central_q_table,
    episode=1,
    step=50,
    save_dir="logs/visualizations"
)

print("\n" + "=" * 60)
print("✓ All tests completed successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  - logs/q_tables.txt (text log)")
print("  - logs/visualizations/*.png (visualization plots)")
print("\nYou can view the visualizations to verify the Q-table heatmaps,")
print("grid visualizations, and statistics plots.")

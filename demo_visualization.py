#!/usr/bin/env python3
"""
Simple demo to run the warehouse visualization.

Usage:
    python3 demo_visualization.py --render False   # run without pygame (headless smoke test)
    python3 demo_visualization.py --render True    # run with pygame visualization (requires display)
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable pygame rendering")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    # Import run_visualization lazily to avoid circular imports at module-import time
    from visualization import run_visualization

    run_visualization(n_episodes=args.episodes, n_steps_per_episode=args.steps, render=args.render)

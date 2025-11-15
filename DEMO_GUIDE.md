# BigThinkAI Discrete MAPPO - Demo Guide

## Project Overview

**BigThinkAI Discrete MAPPO** is a multi-agent reinforcement learning (MARL) system that trains robots to collaboratively pick up targets and deliver them to a depot in a warehouse environment. The system uses **Multi-Agent Proximal Policy Optimization (MAPPO)** with decentralized actors and a centralized critic.

---

## Key Talking Points

### 1. **Problem & Motivation**
- **Real-world scenario**: Autonomous warehouses require robots to coordinate and efficiently pick/deliver items.
- **Challenge**: Each robot is independent (decentralized) but needs to learn to cooperate without explicit communication.
- **Solution**: Multi-agent RL with MAPPO allows agents to learn from shared rewards and critic feedback.

### 2. **Algorithm: MAPPO (Multi-Agent PPO)**
- **PPO Basics**: 
  - Policy gradient method that clips the objective to prevent large policy updates.
  - Advantage function guides learning toward better actions.
  
- **Multi-Agent Extension**:
  - **Decentralized Actors**: Each robot has its own policy network → learns its own actions from local observations.
  - **Centralized Critic**: Single value network sees all agents' observations → estimates accurate value for advantage calculation.
  - **Why?**: Decentralized execution (scalable), centralized training (stable).

- **Training Loop**:
  1. Collect transitions: agents take actions, environment returns rewards/next states.
  2. Compute advantages: critic estimates state value; advantage = reward + γ*V(next_state) - V(state).
  3. Update actors: PPO clipped objective encourages policy toward higher-advantage actions.
  4. Update critic: minimize MSE loss between predicted value and actual returns.

### 3. **Environment Design**
- **State Space**: 6D normalized vector per robot
  - Robot position (row, col)
  - Carrying flag (holding target?)
  - Depot position (row, col)
  - Targets remaining
  
- **Action Space**: 7 discrete actions
  - Movement: STAY, UP, DOWN, LEFT, RIGHT
  - Interaction: PICK (grab target), DROP (deliver at depot)

- **Reward Structure** (configurable):
  - `time_penalty`: -0.01 per step (incentivizes speed)
  - `collision_penalty`: -0.05 for blocked moves
  - `pick_reward`: +1.0 for picking a target
  - `drop_reward`: +5.0 for successful delivery
  - `distance_penalty`: penalize moving away from depot while carrying
  - `failed_drop_penalty`: penalize standing on depot with a target but not dropping

### 4. **Decentralized vs. Centralized**
- **Decentralized Actors**: 
  - Each robot only sees its own observation (position, carrying flag, etc.).
  - Learns its own policy independently.
  - Scalable: adding robots just adds more actors.
  
- **Centralized Critic**:
  - Joint observation includes all agents' states.
  - Computes value baseline for advantage estimation.
  - Training-only: not used at inference/execution.

### 5. **Training Visualization**
- **Live Pygame Display**:
  - Green circles = robots
  - Orange squares = targets
  - Blue = depot
  - Yellow = bay (starting area)
  - Info panel shows: step count, targets remaining, deliveries completed.
  
- **Episode-End Heatmaps**:
  - After each episode, saves a state-action frequency heatmap.
  - Shows which actions agents took in which states.
  - Helps diagnose learning: are robots converging to good strategies?

---

## Code Architecture

### File Breakdown

#### 1. **`warehouse_mappo.py`** - Environment
- **WarehouseEnv class**: Implements gym-like environment
  - `reset()`: Random grid layout, place robots/targets/depot
  - `step(actions)`: Apply actions, compute rewards, check termination
  - State-action tracking for episode heatmaps
  - Q-table logging and visualization hooks

#### 2. **`mappo_agent.py`** - Learning Agent
- **MAPPOAgent class**: Orchestrates training
  - `act(obs_dict)`: Actor networks select actions and compute log-probabilities
  - `update(buffer)`: PPO update on collected transitions
  - Per-agent decentralized actors + centralized critic

#### 3. **`mappo_models.py`** - Neural Networks
- **Actor**: 2-layer MLP (obs_dim → hidden → action_dim logits)
- **CentralCritic**: 2-layer MLP (joint_obs_dim → hidden → 1 value)

#### 4. **`mappo_buffer.py`** - Experience Storage
- Circular buffer for off-policy transitions
- Computes advantage and returns using GAE (Generalized Advantage Estimation)

#### 5. **`train_discrete_mappo.py`** - Main Training Loop
- Creates environment, agent, buffer
- Collects episodes, renders live, saves heatmaps
- Configurable: EPISODES, STEPS, RENDER flag

#### 6. **`visualization.py`** - Pygame Renderer
- **WarehouseVisualizer**: Live grid display
- Shows agents, targets, depot, info panel
- Handles pygame quit event

#### 7. **`log.py`** - Logging & Analysis
- Q-table tracking and heatmap generation
- Saves text logs and visualization images

---

## Demo Flow

### Setup
```bash
cd /home/henry/bigthinkai/src/BigThinkAI-Discrete-Mappo
```

### Option A: Quick Headless Demo (No Window)
```bash
python3 -c "
from train_discrete_mappo import *
# Modify RENDER = False in the file, then run
python3 train_discrete_mappo.py
"
```
- Shows terminal output: episode rewards, step counts
- Saves heatmaps to `logs/visualizations/`

### Option B: Full Live Demo (Requires Display)
1. Open `train_discrete_mappo.py`
2. Ensure `RENDER = True` and `EPISODES = 5` (for short demo)
3. Run:
   ```bash
   python3 train_discrete_mappo.py
   ```
4. Watch pygame window show agents collaborating in real-time
5. Close window → heatmaps save automatically

### Option C: Quick Single-Episode Test
```bash
python3 demo_visualization.py --episodes 1 --steps 100 --render
```

---

## Talking Points During Demo

### 1. **Show the Environment**
- Point out robots (green), targets (orange), depot (blue), bay (yellow)
- Explain: "Each robot receives a 6D observation: its position, whether it's carrying, the depot position, and how many targets remain."

### 2. **Explain Real-Time Learning**
- "What you're seeing is live training. The robots are learning to collaborate using PPO."
- "Each robot has its own policy (decentralized), but we use a shared critic (centralized) to help them learn."

### 3. **Reward Structure**
- "Notice robots move toward the depot when carrying—that's the distance penalty at work."
- "Picking gives +1, dropping gives +5. Time penalty (-0.01 per step) encourages speed."
- "If a robot stands on the depot with a crate but doesn't drop, it gets a huge penalty."

### 4. **Decentralized Coordination**
- "The robots don't communicate directly. They learn to cooperate through shared rewards and the critic's value estimates."
- "The critic sees all robot states during training, so it learns to predict good outcomes—this helps actors converge faster."

### 5. **Episode End & Heatmaps**
- After episode finishes: "We save a heatmap showing which actions each robot took in which states."
- "This helps us diagnose learning: are robots picking targets? Delivering? Or just wandering?"

### 6. **Scalability**
- "This architecture scales well. Adding more robots just means more actors (one per robot). The critic overhead stays manageable."

---

## Key Hyperparameters to Discuss

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_robots` | 3 | Number of autonomous agents |
| `n_targets` | 3 | Number of items to pick/deliver |
| `STEPS` | 1000 | Max steps per episode |
| `EPISODES` | 50 | Total training episodes |
| `actor_lr` | 3e-4 | Actor network learning rate |
| `critic_lr` | 3e-4 | Critic network learning rate |
| `clip_ratio` | 0.2 | PPO clipping threshold |
| `gamma` | 0.99 | Discount factor (long-term reward weight) |
| `lam` | 0.95 | GAE lambda (variance-bias tradeoff) |

---

## Expected Behavior Over Training

### Early Episodes (0-10)
- Robots move randomly or cluster together
- Few successful deliveries
- Heatmaps show uniform action distribution (undecided)

### Mid Episodes (10-30)
- Robots begin moving toward targets
- Some successful pick/drop cycles
- Heatmaps show concentration on certain actions (learning convergence)

### Late Episodes (30+)
- Robots prioritize targets, move to depot, deliver
- Episode rewards improve
- Heatmaps show sparse, focused action patterns per state

---

## Questions You May Encounter

### Q: "Why decentralized actors + centralized critic?"
**A:** The critic only runs during training. At execution (real warehouse), each robot only needs its decentralized actor—no communication required. But during training, the critic helps stabilize learning by providing accurate value estimates.

### Q: "How do robots know about each other?"
**A:** They don't explicitly. They only see their own state. But they learn implicitly through shared rewards: if a team delivers, everyone benefits. Collision penalties encourage spacing.

### Q: "What's PPO?"
**A:** Proximal Policy Optimization. It's a safer policy gradient method that prevents huge policy swings. We use a clipped objective: `min(ratio * advantage, clip(ratio, 1±ε) * advantage)` to keep updates stable.

### Q: "Why save heatmaps?"
**A:** Interpretability. We can see which actions robots favored in which states. If a robot keeps picking at the depot (bad), we'd see that in the heatmap and adjust rewards.

---

## Customization Tips for Your Demo

### Vary Difficulty
- **Easy**: `n_targets=2, grid_size=(6,6), STEPS=500`
- **Hard**: `n_targets=5, grid_size=(9,9), STEPS=1000`

### Adjust Reward Structure
Edit `warehouse_mappo.py`:
```python
env.set_rewards(
    time_penalty=0.02,      # Make it more urgent
    pick_reward=2.0,        # Incentivize picking
    drop_reward=10.0,       # Big bonus for delivery
    distance_penalty=0.1    # Penalize inefficient moves
)
```

### Slower/Faster Visualization
In `train_discrete_mappo.py`:
```python
visualizer = WarehouseVisualizer(env, cell_size=50, fps=5)  # fps=5 for slow, fps=20 for fast
```

---

## Summary Slide

**BigThinkAI MAPPO achieves multi-robot coordination through:**
1. **Decentralized learning**: Each robot learns its own policy.
2. **Centralized stability**: Shared critic ensures value estimates are accurate.
3. **PPO optimization**: Safe, stable policy updates via clipped objectives.
4. **Reward shaping**: Configurable incentives (speed, picking, dropping, efficiency).
5. **Live visualization + analysis**: Watch learning in real-time, diagnose via heatmaps.


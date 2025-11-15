import pygame
import numpy as np

# Local copy of cell constants to avoid circular import with `warehouse_mappo`.
# Keep values in sync with `warehouse_mappo.py`.
EMPTY = 0
TARGET = 1
DEPOT = 2
BAY = 3


class WarehouseVisualizer:
    """
    Pygame-based visualizer for the warehouse environment.
    Displays robots, targets, depot, and bay with specified colors.
    """

    # Color definitions (RGB)
    COLORS = {
        'background': (255, 255, 255),  # White
        'robot': (0, 255, 0),           # Green
        'bay': (255, 255, 0),           # Yellow
        'target': (255, 165, 0),        # Orange
        'depot': (0, 0, 255),           # Blue
        'empty': (200, 200, 200),       # Light gray
        'grid': (150, 150, 150),        # Dark gray
        'text': (0, 0, 0),              # Black
    }

    def __init__(self, env, cell_size=50, fps=10):
        """
        Initialize the warehouse visualizer.
        
        Args:
            env (WarehouseEnv): The warehouse environment
            cell_size (int): Size of each grid cell in pixels
            fps (int): Frames per second for animation
        """
        self.env = env
        self.cell_size = cell_size
        self.fps = fps
        
        # Calculate window size
        self.window_width = env.grid_w * cell_size
        self.window_height = env.grid_h * cell_size + 100  # Extra space for info
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption("Warehouse Multi-Robot Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def render(self):
        """Render the current environment state."""
        # Fill background
        self.screen.fill(self.COLORS['background'])
        
        # Draw grid
        self._draw_grid()
        
        # Draw environment elements
        self._draw_environment()
        
        # Draw robots
        self._draw_robots()
        
        # Draw info panel
        self._draw_info_panel()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_grid(self):
        """Draw the warehouse grid."""
        for row in range(self.env.grid_h + 1):
            start_pos = (0, row * self.cell_size)
            end_pos = (self.window_width, row * self.cell_size)
            pygame.draw.line(self.screen, self.COLORS['grid'], start_pos, end_pos, 1)
        
        for col in range(self.env.grid_w + 1):
            start_pos = (col * self.cell_size, 0)
            end_pos = (col * self.cell_size, self.env.grid_h * self.cell_size)
            pygame.draw.line(self.screen, self.COLORS['grid'], start_pos, end_pos, 1)

    def _draw_environment(self):
        """Draw depot, bay, and targets."""
        # Draw all grid cells
        for row in range(self.env.grid_h):
            for col in range(self.env.grid_w):
                cell_type = self.env.grid[row, col]
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                if cell_type == DEPOT:
                    pygame.draw.rect(self.screen, self.COLORS['depot'], rect)
                    self._draw_label(rect, "D", self.COLORS['text'])
                
                elif cell_type == BAY:
                    pygame.draw.rect(self.screen, self.COLORS['bay'], rect)
                    self._draw_label(rect, "B", self.COLORS['text'])
                
                elif cell_type == TARGET:
                    pygame.draw.rect(self.screen, self.COLORS['target'], rect)
                    self._draw_label(rect, "T", self.COLORS['text'])

    def _draw_robots(self):
        """Draw all robots on the grid."""
        for agent_name, (row, col) in self.env.robot_positions.items():
            rect = pygame.Rect(
                col * self.cell_size,
                row * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            
            # Draw robot circle
            center = (
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2
            )
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, self.COLORS['robot'], center, radius)
            
            # Draw carrying indicator if robot is carrying
            if self.env.robot_carry[agent_name]:
                pygame.draw.circle(
                    self.screen,
                    self.COLORS['target'],
                    center,
                    radius - 5,
                    3  # border only
                )
            
            # Draw robot label
            robot_id = agent_name.split('_')[1]
            self._draw_label(rect, f"R{robot_id}", self.COLORS['text'])

    def _draw_label(self, rect, text, color):
        """Draw text label in the center of a rect."""
        text_surface = self.small_font.render(text, True, color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def _draw_info_panel(self):
        """Draw information panel at the bottom."""
        panel_y = self.env.grid_h * self.cell_size
        panel_height = 100
        
        # Draw panel background
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (0, panel_y, self.window_width, panel_height)
        )
        
        # Draw separator line
        pygame.draw.line(
            self.screen,
            self.COLORS['grid'],
            (0, panel_y),
            (self.window_width, panel_y),
            2
        )
        
        # Draw information text
        info_y = panel_y + 10
        
        # Step count
        step_text = self.font.render(
            f"Step: {self.env.step_count}/{self.env.max_steps}",
            True,
            self.COLORS['text']
        )
        self.screen.blit(step_text, (10, info_y))
        
        # Targets remaining
        targets_left = int((self.env.grid == TARGET).sum())
        targets_text = self.font.render(
            f"Targets: {targets_left}/{self.env.n_targets}",
            True,
            self.COLORS['text']
        )
        self.screen.blit(targets_text, (10, info_y + 25))
        
        # Deliveries completed
        deliveries_text = self.font.render(
            f"Delivered: {self.env.completed_deliveries}/{self.env.initial_targets}",
            True,
            self.COLORS['text']
        )
        self.screen.blit(deliveries_text, (10, info_y + 50))
        
        # Legend
        legend_x = self.window_width // 2
        legend_text = self.small_font.render(
            "Legend: D=Depot  B=Bay  T=Target  R=Robot",
            True,
            self.COLORS['text']
        )
        self.screen.blit(legend_text, (legend_x, info_y + 10))
        
        carrying_text = self.small_font.render(
            "Carrying robots have orange circle outline",
            True,
            self.COLORS['text']
        )
        self.screen.blit(carrying_text, (legend_x, info_y + 35))



    def close(self):
        """Close the pygame window."""
        pygame.quit()


def run_visualization(n_episodes=5, n_steps_per_episode=100, render=True):
    """
    Run the warehouse environment with visualization.
    
    Args:
        n_episodes (int): Number of episodes to run
        n_steps_per_episode (int): Max steps per episode
        render (bool): Whether to render with pygame
    """
    # Create environment (import inside function to avoid circular imports)
    from warehouse_mappo import WarehouseEnv
    env = WarehouseEnv(grid_size=(7, 7), n_robots=3, n_targets=5, max_steps=n_steps_per_episode, render=render, render_fps=5)
    
    # Create visualizer
    if render:
        visualizer = WarehouseVisualizer(env, cell_size=50, fps=5)
    
    try:
        for episode in range(n_episodes):
            obs, infos = env.reset()
            episode_reward = 0.0
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*60}")
            
            done = False
            while not done:
                # Render
                if render:
                    visualizer.render()
                    
                    # Check for quit event
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            break
                
                # Random action selection
                actions = {a: env.action_spaces[a].sample() for a in env.agents}
                
                # Step environment
                obs, rewards, terms, truncs, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                
                # Check if episode is done
                if all(terms.values()) or all(truncs.values()):
                    done = True
            
            print(f"Episode {episode + 1} completed!")
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"Steps taken: {env.step_count}")
        
        print("\n" + "="*60)
        print("Visualization complete!")
        print("="*60)
    
    finally:
        if render:
            visualizer.close()


if __name__ == "__main__":
    run_visualization(n_episodes=3, n_steps_per_episode=100, render=True)

"""
Miniverse: A Simple 2D Lander Environment for World Models Research

Environment Specification:
    State s:       [x, y, vx, vy]  (4-dimensional)
    Action a:      [thrust_x, thrust_y]  (continuous, clipped to [-1, 1])
    Observation:   s + Gaussian noise
    Dynamics:      Euler integration with gravity and optional drag
    Boundaries:    Bounded arena with elastic collisions

This serves as the foundational "ground truth" environment that World Models
will learn to predict. The state is what we want to model, observations are
what the encoder sees, and dynamics are what the transition model learns.

Author: World Models Research
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection

np.random.seed(42)

# Environment Configuration

class MiniverseConfig:
    """Configuration for the Miniverse environment."""
    
    def __init__(
        self,
        # Arena bounds
        x_min: float = 0.0,
        x_max: float = 10.0,
        y_min: float = 0.0,
        y_max: float = 10.0,
        # Physics
        dt: float = 0.05,
        gravity: float = -2.0,      # Negative = downward
        drag: float = 0.02,         # Velocity damping coefficient
        thrust_scale: float = 5.0,  # Action multiplier
        # Observation noise
        pos_noise_std: float = 0.1,
        vel_noise_std: float = 0.05,
        # Elastic collision damping (1.0 = perfect elastic)
        restitution: float = 0.8,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dt = dt
        self.gravity = gravity
        self.drag = drag
        self.thrust_scale = thrust_scale
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.restitution = restitution


DEFAULT_CONFIG = MiniverseConfig()


# Core Dynamics Functions (Functional API)

def clip_action(action: np.ndarray) -> np.ndarray:
    """Clip action to valid range [-1, 1]."""
    return np.clip(action, -1.0, 1.0)


def apply_physics(state: np.ndarray, action: np.ndarray, config: MiniverseConfig) -> np.ndarray:
    """
    Apply physics update using Euler integration.
    
    Args:
        state: [x, y, vx, vy]
        action: [thrust_x, thrust_y] (will be clipped to [-1, 1])
        config: Environment configuration
    
    Returns:
        next_state: [x', y', vx', vy'] after physics update (before boundary check)
    """
    x, y, vx, vy = state
    action = clip_action(action)
    ax, ay = action * config.thrust_scale
    
    # Apply gravity (only in y-direction)
    ay += config.gravity
    
    # Apply drag (opposes velocity)
    ax -= config.drag * vx
    ay -= config.drag * vy
    
    # Euler integration
    vx_new = vx + ax * config.dt
    vy_new = vy + ay * config.dt
    x_new = x + vx_new * config.dt
    y_new = y + vy_new * config.dt
    
    return np.array([x_new, y_new, vx_new, vy_new])


def apply_boundary_collision(state: np.ndarray, config: MiniverseConfig) -> np.ndarray:
    """
    Handle elastic collisions with arena boundaries.
    
    Args:
        state: [x, y, vx, vy]
        config: Environment configuration
    
    Returns:
        state: [x, y, vx, vy] after boundary collision handling
    """
    x, y, vx, vy = state
    r = config.restitution
    
    # Left/Right walls
    if x < config.x_min:
        x = config.x_min
        vx = -vx * r
    elif x > config.x_max:
        x = config.x_max
        vx = -vx * r
    
    # Bottom/Top walls
    if y < config.y_min:
        y = config.y_min
        vy = -vy * r
    elif y > config.y_max:
        y = config.y_max
        vy = -vy * r
    
    return np.array([x, y, vx, vy])


def get_observation(state: np.ndarray, config: MiniverseConfig) -> np.ndarray:
    """
    Generate noisy observation from true state.
    
    For World Models: This is what the encoder (VAE) will see.
    Later, this will be replaced with 64x64 pixel rendering.
    
    Args:
        state: [x, y, vx, vy] true state
        config: Environment configuration
    
    Returns:
        observation: [x, y, vx, vy] + Gaussian noise
    """
    noise = np.array([
        config.pos_noise_std * np.random.randn(),
        config.pos_noise_std * np.random.randn(),
        config.vel_noise_std * np.random.randn(),
        config.vel_noise_std * np.random.randn(),
    ])
    return state + noise


def step(state: np.ndarray, action: np.ndarray, config: MiniverseConfig = None) -> tuple:
    """
    Execute one environment step.
    
    This is the main API for the Miniverse.
    
    Args:
        state: [x, y, vx, vy] current state
        action: [thrust_x, thrust_y] control input
        config: Environment configuration (uses DEFAULT_CONFIG if None)
    
    Returns:
        next_state: [x, y, vx, vy] true next state
        observation: [x, y, vx, vy] noisy observation of next state
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Physics update
    next_state = apply_physics(state, action, config)
    
    # Boundary collisions
    next_state = apply_boundary_collision(next_state, config)
    
    # Generate noisy observation
    observation = get_observation(next_state, config)
    
    return next_state, observation


def reset(config: MiniverseConfig = None, random_init: bool = True) -> tuple:
    """
    Reset environment to initial state.
    
    Args:
        config: Environment configuration
        random_init: If True, randomize initial position/velocity
    
    Returns:
        state: Initial true state
        observation: Initial noisy observation
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if random_init:
        # Random position in center region, small random velocity
        x = np.random.uniform(config.x_min + 2, config.x_max - 2)
        y = np.random.uniform(config.y_min + 4, config.y_max - 2)  # Start higher up
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
    else:
        # Default: center of arena, stationary
        x = (config.x_min + config.x_max) / 2
        y = (config.y_min + config.y_max) * 0.7  # Start 70% up
        vx, vy = 0.0, 0.0
    
    state = np.array([x, y, vx, vy])
    observation = get_observation(state, config)
    
    return state, observation

# Simulation & Data Collection

def simulate_trajectory(
    T: int = 200,
    policy=None,
    config: MiniverseConfig = None,
    seed: int = None
) -> dict:
    """
    Simulate a trajectory in the Miniverse.
    
    Args:
        T: Number of timesteps
        policy: Function (state) -> action. If None, uses random actions.
        config: Environment configuration
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
            states: (T+1, 4) true states
            observations: (T+1, 4) noisy observations
            actions: (T, 2) actions taken
    """
    if seed is not None:
        np.random.seed(seed)
    if config is None:
        config = DEFAULT_CONFIG
    
    state, obs = reset(config)
    
    states = [state]
    observations = [obs]
    actions = []
    
    for t in range(T):
        if policy is None:
            action = np.random.uniform(-1, 1, size=2)
        else:
            action = policy(state)
        
        state, obs = step(state, action, config)
        
        states.append(state)
        observations.append(obs)
        actions.append(action)
    
    return {
        'states': np.array(states),
        'observations': np.array(observations),
        'actions': np.array(actions),
    }



def plot_trajectory(trajectory: dict, config: MiniverseConfig = None, title: str = None):
    """
    Plot a trajectory in the Miniverse.
    
    Shows:
        - True state trajectory (solid line)
        - Noisy observations (scattered points)
        - Arena boundaries
        - Velocity vectors at select timesteps
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    states = trajectory['states']
    obs = trajectory['observations']
    actions = trajectory['actions']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Plot 1: 2D Trajectory 
    ax = axes[0]
    
    # Arena boundary
    arena = Rectangle(
        (config.x_min, config.y_min),
        config.x_max - config.x_min,
        config.y_max - config.y_min,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(arena)
    
    # True trajectory (color-coded by time)
    points = states[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
    lc = LineCollection(segments, colors=colors, linewidth=2)
    ax.add_collection(lc)
    
    # Observations (noisy)
    ax.scatter(obs[:, 0], obs[:, 1], c='red', s=8, alpha=0.3, label='Observations')
    
    # Start/End markers
    ax.scatter(*states[0, :2], c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(*states[-1, :2], c='blue', s=100, marker='x', zorder=5, label='End')
    
    # Velocity vectors (every 20 steps)
    skip = max(1, len(states) // 10)
    for i in range(0, len(states), skip):
        ax.arrow(
            states[i, 0], states[i, 1],
            states[i, 2] * 0.3, states[i, 3] * 0.3,
            head_width=0.15, head_length=0.1, fc='gray', ec='gray', alpha=0.5
        )
    
    ax.set_xlim(config.x_min - 0.5, config.x_max + 0.5)
    ax.set_ylim(config.y_min - 0.5, config.y_max + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Trajectory (color = time)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Plot 2: State components over time 
    ax = axes[1]
    t = np.arange(len(states))
    ax.plot(t, states[:, 0], label='x', alpha=0.8)
    ax.plot(t, states[:, 1], label='y', alpha=0.8)
    ax.plot(t, states[:, 2], label='vx', linestyle='--', alpha=0.8)
    ax.plot(t, states[:, 3], label='vy', linestyle='--', alpha=0.8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State value')
    ax.set_title('State Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Actions over time 
    ax = axes[2]
    t_a = np.arange(len(actions))
    ax.plot(t_a, actions[:, 0], label='thrust_x', alpha=0.8)
    ax.plot(t_a, actions[:, 1], label='thrust_y', alpha=0.8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(1, color='red', linestyle=':', alpha=0.3)
    ax.axhline(-1, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action value')
    ax.set_title('Control Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.3, 1.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_observation_noise(trajectory: dict, config: MiniverseConfig = None):
    """
    Visualize the observation noise: compare true state vs noisy observations.
    
    This is crucial for World Models: the encoder must learn to denoise.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    states = trajectory['states']
    obs = trajectory['observations']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(len(states))
    labels = ['x', 'y', 'vx', 'vy']
    
    for i, ax in enumerate(axes.flat):
        ax.plot(t, states[:, i], label=f'True {labels[i]}', linewidth=2)
        ax.scatter(t, obs[:, i], s=10, alpha=0.5, label=f'Observed {labels[i]}', color='red')
        
        # Noise band
        if i < 2:
            noise_std = config.pos_noise_std
        else:
            noise_std = config.vel_noise_std
        ax.fill_between(
            t, states[:, i] - 2*noise_std, states[:, i] + 2*noise_std,
            alpha=0.2, label=f'±2σ (σ={noise_std})'
        )
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i]}: True vs Observed')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Observation Noise Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Example Policies (for testing)

def hover_policy(state: np.ndarray, target_y: float = 5.0, kp: float = 0.5, kd: float = 0.3) -> np.ndarray:
    """
    Simple PD controller to hover at target height.
    
    Args:
        state: [x, y, vx, vy]
        target_y: Desired height
        kp: Proportional gain
        kd: Derivative gain
    
    Returns:
        action: [thrust_x, thrust_y]
    """
    x, y, vx, vy = state
    
    # Vertical control: fight gravity + reach target height
    error_y = target_y - y
    thrust_y = kp * error_y - kd * vy
    
    # Horizontal control: stay centered, dampen horizontal motion
    thrust_x = -0.2 * (x - 5.0) - 0.3 * vx
    
    return np.array([thrust_x, thrust_y])


def landing_policy(state: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """
    Policy to land softly at a target location.
    
    Args:
        state: [x, y, vx, vy]
        target: [x_target, y_target] landing pad location
    
    Returns:
        action: [thrust_x, thrust_y]
    """
    if target is None:
        target = np.array([5.0, 0.5])  # Default: center bottom
    
    x, y, vx, vy = state
    tx, ty = target
    
    # Position error
    ex = tx - x
    ey = ty - y
    
    # PD control with different gains based on altitude
    if y > 3.0:
        # High altitude: focus on horizontal positioning
        thrust_x = 0.4 * ex - 0.3 * vx
        thrust_y = 0.2 * ey - 0.2 * vy
    else:
        # Low altitude: focus on soft landing
        thrust_x = 0.3 * ex - 0.4 * vx
        thrust_y = 0.5 * ey - 0.6 * vy  # Strong damping for soft landing
    
    return np.array([thrust_x, thrust_y])


# Demo

if __name__ == "__main__":
    print("=" * 60)
    print("MINIVERSE: 2D Lander Environment for World Models")
    print("=" * 60)
    
    # Create configuration
    config = MiniverseConfig(
        gravity=-2.0,
        drag=0.02,
        thrust_scale=5.0,
        pos_noise_std=0.15,
        vel_noise_std=0.08,
        restitution=0.7,
    )
    
    print("\nConfiguration:")
    print(f"  Arena: [{config.x_min}, {config.x_max}] x [{config.y_min}, {config.y_max}]")
    print(f"  Gravity: {config.gravity}")
    print(f"  Drag: {config.drag}")
    print(f"  Thrust scale: {config.thrust_scale}")
    print(f"  Position noise σ: {config.pos_noise_std}")
    print(f"  Velocity noise σ: {config.vel_noise_std}")
    print(f"  Restitution: {config.restitution}")
    
    # Demo 1: Random policy 
    print("\n[Demo 1] Random Policy Trajectory")
    traj_random = simulate_trajectory(T=300, policy=None, config=config, seed=42)
    plot_trajectory(traj_random, config, title="Random Policy")
    
    # Demo 2: Hover policy 
    print("\n[Demo 2] Hover Policy (PD Controller)")
    traj_hover = simulate_trajectory(
        T=300,
        policy=lambda s: hover_policy(s, target_y=6.0),
        config=config,
        seed=42
    )
    plot_trajectory(traj_hover, config, title="Hover Policy (target y=6.0)")
    
    # Demo 3: Landing policy 
    print("\n[Demo 3] Landing Policy")
    traj_land = simulate_trajectory(
        T=300,
        policy=lambda s: landing_policy(s, target=np.array([7.0, 0.5])),
        config=config,
        seed=42
    )
    plot_trajectory(traj_land, config, title="Landing Policy (target=[7.0, 0.5])")
    
    # Demo 4: Observation noise analysis 
    print("\n[Demo 4] Observation Noise Analysis")
    plot_observation_noise(traj_hover, config)
    
    print("\n" + "=" * 60)
    print("Miniverse ready for World Models training!")
    print("=" * 60)

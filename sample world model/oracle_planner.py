"""
Oracle Planner: Model Predictive Control with Ground Truth World Model

Exercise 1: Implement MPC using the exact physics equations from Miniverse.
This serves as the "oracle" baseline — if we can't control the system with
perfect knowledge of dynamics, learning a model won't help.

Key Concepts:
    - Ground Truth World Model: Uses exact physics for prediction
    - Random Shooting: Sample K action sequences, pick best one
    - Receding Horizon: Re-plan at every timestep

Connection to World Models:
    - Later, we replace GroundTruthWorldModel with a learned model (VAE + RNN)
    - The planning algorithm (random shooting) stays the same
    - Performance gap = "cost of learning" (model error)

Author: World Models Research
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

from miniverse import (
    MiniverseConfig, DEFAULT_CONFIG,
    apply_physics, apply_boundary_collision, clip_action,
    step, reset
)


# =============================================================================
# Ground Truth World Model
# =============================================================================

class GroundTruthWorldModel:
    """
    World Model that uses exact physics equations.
    
    This is the "oracle" — it has perfect knowledge of the environment
    dynamics. In a real World Model, this would be replaced by a learned
    neural network (VAE encoder + MDN-RNN transition model).
    
    Interface:
        next(state, action) -> next_state
    
    This interface will be shared with learned models later.
    """
    
    def __init__(self, config: MiniverseConfig = None):
        """
        Initialize with environment configuration.
        
        Args:
            config: Miniverse configuration (uses DEFAULT_CONFIG if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG
    
    def next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict next state given current state and action.
        
        Uses exact Miniverse physics (no noise).
        
        Args:
            state: Current state [x, y, vx, vy]
            action: Action [thrust_x, thrust_y]
        
        Returns:
            next_state: Predicted next state [x, y, vx, vy]
        """
        # Apply physics (deterministic, no observation noise)
        next_state = apply_physics(state, action, self.config)
        next_state = apply_boundary_collision(next_state, self.config)
        return next_state
    
    def rollout(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Rollout a sequence of actions from initial state.
        
        Args:
            state: Initial state [x, y, vx, vy]
            actions: Action sequence (H, 2)
        
        Returns:
            states: State trajectory (H+1, 4) including initial state
        """
        H = len(actions)
        states = np.zeros((H + 1, 4))
        states[0] = state
        
        for t in range(H):
            states[t + 1] = self.next(states[t], actions[t])
        
        return states


# =============================================================================
# Reward / Cost Functions
# =============================================================================

def reward_reach_target(state: np.ndarray, target: np.ndarray, 
                        velocity_penalty: float = 0.1) -> float:
    """
    Reward for reaching a target position.
    
    Reward = -distance_to_target - velocity_penalty * speed
    
    Args:
        state: [x, y, vx, vy]
        target: [x_target, y_target]
        velocity_penalty: Weight for penalizing high velocity
    
    Returns:
        reward: Scalar reward (higher is better)
    """
    pos = state[:2]
    vel = state[2:]
    
    distance = np.linalg.norm(pos - target)
    speed = np.linalg.norm(vel)
    
    return -distance - velocity_penalty * speed


def reward_soft_landing(state: np.ndarray, target: np.ndarray,
                        velocity_penalty: float = 0.5,
                        height_bonus: float = 0.1) -> float:
    """
    Reward for soft landing at target.
    
    Emphasizes low velocity near target (for landing tasks).
    
    Args:
        state: [x, y, vx, vy]
        target: [x_target, y_target]
        velocity_penalty: Weight for velocity (higher = softer landing)
        height_bonus: Small bonus for being low (encourages descent)
    
    Returns:
        reward: Scalar reward
    """
    pos = state[:2]
    vel = state[2:]
    
    distance = np.linalg.norm(pos - target)
    speed = np.linalg.norm(vel)
    
    # Exponential penalty for distance (smoother gradient)
    distance_cost = distance ** 2
    
    # Velocity penalty increases as we get closer to target
    closeness = 1.0 / (1.0 + distance)
    velocity_cost = velocity_penalty * speed * closeness
    
    # Small bonus for low altitude (encourages landing)
    height_cost = height_bonus * pos[1]
    
    return -(distance_cost + velocity_cost + height_cost)


def cumulative_reward(states: np.ndarray, target: np.ndarray,
                      reward_fn=reward_reach_target,
                      discount: float = 0.99) -> float:
    """
    Compute cumulative discounted reward for a trajectory.
    
    Args:
        states: State trajectory (H+1, 4)
        target: Target position [x, y]
        reward_fn: Reward function to use
        discount: Discount factor γ
    
    Returns:
        total_reward: Cumulative discounted reward
    """
    total = 0.0
    for t, state in enumerate(states):
        total += (discount ** t) * reward_fn(state, target)
    return total


# =============================================================================
# Random Shooting MPC Planner
# =============================================================================

class RandomShootingMPC:
    """
    Model Predictive Control using Random Shooting.
    
    Algorithm:
        1. Sample K random action sequences of length H
        2. Simulate each sequence using the world model
        3. Compute cumulative reward for each trajectory
        4. Return first action of the best sequence
    
    This is a simple but effective planning algorithm that works well
    when we have a good world model. More sophisticated planners
    (CEM, MPPI, gradient-based) can be used for better performance.
    """
    
    def __init__(
        self,
        world_model: GroundTruthWorldModel,
        horizon: int = 20,
        num_samples: int = 1000,
        target: np.ndarray = None,
        reward_fn = reward_reach_target,
        discount: float = 0.99,
        action_noise_std: float = 0.0,
    ):
        """
        Initialize MPC planner.
        
        Args:
            world_model: World model for prediction
            horizon: Planning horizon H
            num_samples: Number of action sequences K
            target: Target position [x, y]
            reward_fn: Reward function
            discount: Discount factor
            action_noise_std: Noise added to best action (exploration)
        """
        self.world_model = world_model
        self.H = horizon
        self.K = num_samples
        self.target = target if target is not None else np.array([5.0, 0.5])
        self.reward_fn = reward_fn
        self.discount = discount
        self.action_noise_std = action_noise_std
        
        # For warm-starting: remember previous best sequence
        self.prev_best_actions = None
    
    def plan(self, state: np.ndarray) -> tuple:
        """
        Plan next action using random shooting.
        
        Args:
            state: Current state [x, y, vx, vy]
        
        Returns:
            best_action: Best action [thrust_x, thrust_y]
            info: Dict with planning diagnostics
        """
        # Generate random action sequences: (K, H, 2)
        action_sequences = np.random.uniform(-1, 1, size=(self.K, self.H, 2))
        
        # Optional: include shifted version of previous best (warm start)
        if self.prev_best_actions is not None:
            # Shift previous best by one timestep, add random action at end
            shifted = np.zeros((self.H, 2))
            shifted[:-1] = self.prev_best_actions[1:]
            shifted[-1] = np.random.uniform(-1, 1, size=2)
            action_sequences[0] = shifted
        
        # Evaluate all sequences
        rewards = np.zeros(self.K)
        
        for k in range(self.K):
            # Rollout trajectory
            trajectory = self.world_model.rollout(state, action_sequences[k])
            
            # Compute cumulative reward
            rewards[k] = cumulative_reward(
                trajectory, self.target, self.reward_fn, self.discount
            )
        
        # Find best sequence
        best_idx = np.argmax(rewards)
        best_actions = action_sequences[best_idx]
        best_reward = rewards[best_idx]
        
        # Save for warm start
        self.prev_best_actions = best_actions.copy()
        
        # Extract first action (MPC principle)
        best_action = best_actions[0]
        
        # Optional: add exploration noise
        if self.action_noise_std > 0:
            best_action += self.action_noise_std * np.random.randn(2)
            best_action = np.clip(best_action, -1, 1)
        
        info = {
            'best_reward': best_reward,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'best_trajectory': self.world_model.rollout(state, best_actions),
        }
        
        return best_action, info


# =============================================================================
# Control Loop
# =============================================================================

def run_mpc_episode(
    config: MiniverseConfig = None,
    target: np.ndarray = None,
    max_steps: int = 300,
    horizon: int = 20,
    num_samples: int = 1000,
    reward_fn = reward_reach_target,
    success_threshold: float = 0.5,
    seed: int = None,
    verbose: bool = True,
) -> dict:
    """
    Run a complete MPC episode.
    
    Args:
        config: Miniverse configuration
        target: Target position
        max_steps: Maximum episode length
        horizon: MPC planning horizon
        num_samples: Number of random samples
        reward_fn: Reward function
        success_threshold: Distance to target for success
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with episode data:
            states, actions, rewards, success, info
    """
    if seed is not None:
        np.random.seed(seed)
    
    if config is None:
        config = DEFAULT_CONFIG
    
    if target is None:
        target = np.array([5.0, 0.5])  # Center bottom
    
    # Initialize world model and planner
    world_model = GroundTruthWorldModel(config)
    planner = RandomShootingMPC(
        world_model=world_model,
        horizon=horizon,
        num_samples=num_samples,
        target=target,
        reward_fn=reward_fn,
    )
    
    # Reset environment
    state, obs = reset(config, random_init=True)
    
    # Episode storage
    states = [state.copy()]
    observations = [obs.copy()]
    actions = []
    rewards = []
    planning_info = []
    
    success = False
    
    for t in range(max_steps):
        # Plan action
        action, info = planner.plan(state)
        
        # Execute in environment
        next_state, next_obs = step(state, action, config)
        
        # Compute reward
        reward = reward_fn(next_state, target)
        
        # Store
        states.append(next_state.copy())
        observations.append(next_obs.copy())
        actions.append(action.copy())
        rewards.append(reward)
        planning_info.append(info)
        
        # Check success
        distance = np.linalg.norm(next_state[:2] - target)
        speed = np.linalg.norm(next_state[2:])
        
        if distance < success_threshold and speed < 1.0:
            success = True
            if verbose:
                print(f"  SUCCESS at step {t+1}! Distance: {distance:.3f}, Speed: {speed:.3f}")
            break
        
        # Progress
        if verbose and (t + 1) % 50 == 0:
            print(f"  Step {t+1}: distance={distance:.3f}, speed={speed:.3f}, reward={reward:.3f}")
        
        state = next_state
    
    return {
        'states': np.array(states),
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'target': target,
        'success': success,
        'final_distance': np.linalg.norm(states[-1][:2] - target),
        'final_speed': np.linalg.norm(states[-1][2:]),
        'config': config,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_mpc_episode(episode: dict, title: str = None):
    """
    Visualize an MPC episode.
    """
    states = episode['states']
    actions = episode['actions']
    rewards = episode['rewards']
    target = episode['target']
    config = episode['config']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ---- Plot 1: 2D Trajectory ----
    ax = axes[0, 0]
    
    # Arena boundary
    arena = Rectangle(
        (config.x_min, config.y_min),
        config.x_max - config.x_min,
        config.y_max - config.y_min,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(arena)
    
    # Target
    target_circle = Circle(target, 0.5, color='green', alpha=0.3, label='Target zone')
    ax.add_patch(target_circle)
    ax.scatter(*target, c='green', s=200, marker='*', zorder=5, label='Target')
    
    # Trajectory (color-coded by time)
    points = states[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.cm.plasma(np.linspace(0, 1, len(segments)))
    lc = LineCollection(segments, colors=colors, linewidth=2)
    ax.add_collection(lc)
    
    # Start/End markers
    ax.scatter(*states[0, :2], c='blue', s=150, marker='o', zorder=5, 
               edgecolor='black', label='Start')
    ax.scatter(*states[-1, :2], c='red', s=150, marker='X', zorder=5,
               edgecolor='black', label='End')
    
    ax.set_xlim(config.x_min - 0.5, config.x_max + 0.5)
    ax.set_ylim(config.y_min - 0.5, config.y_max + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2D Trajectory (color = time)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # ---- Plot 2: Distance to target over time ----
    ax = axes[0, 1]
    distances = np.linalg.norm(states[:, :2] - target, axis=1)
    speeds = np.linalg.norm(states[:, 2:], axis=1)
    t = np.arange(len(states))
    
    ax.plot(t, distances, 'b-', linewidth=2, label='Distance to target')
    ax.plot(t, speeds, 'r--', linewidth=2, label='Speed')
    ax.axhline(0.5, color='green', linestyle=':', label='Success threshold')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.set_title('Distance & Speed vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ---- Plot 3: Actions over time ----
    ax = axes[1, 0]
    t_a = np.arange(len(actions))
    ax.plot(t_a, actions[:, 0], 'b-', linewidth=2, label='thrust_x', alpha=0.8)
    ax.plot(t_a, actions[:, 1], 'r-', linewidth=2, label='thrust_y', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(1, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(-1, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action')
    ax.set_title('Control Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.3, 1.3)
    
    # ---- Plot 4: Cumulative reward ----
    ax = axes[1, 1]
    cumulative = np.cumsum(rewards)
    ax.plot(np.arange(len(rewards)), rewards, 'b-', alpha=0.5, label='Instant reward')
    ax.plot(np.arange(len(rewards)), cumulative / (np.arange(len(rewards)) + 1), 
            'r-', linewidth=2, label='Mean reward')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overall title
    status = "SUCCESS" if episode['success'] else "TIMEOUT"
    if title is None:
        title = f"Oracle MPC Episode ({status})"
    fig.suptitle(
        f"{title}\nFinal distance: {episode['final_distance']:.3f}, "
        f"Final speed: {episode['final_speed']:.3f}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()


def create_mpc_animation(episode: dict, filename: str = None, fps: int = 20):
    """
    Create an animation of the MPC episode.
    """
    states = episode['states']
    target = episode['target']
    config = episode['config']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Arena boundary
    arena = Rectangle(
        (config.x_min, config.y_min),
        config.x_max - config.x_min,
        config.y_max - config.y_min,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(arena)
    
    # Target
    target_circle = Circle(target, 0.5, color='green', alpha=0.3)
    ax.add_patch(target_circle)
    ax.scatter(*target, c='green', s=200, marker='*', zorder=5)
    
    # Agent (will be updated)
    agent, = ax.plot([], [], 'bo', markersize=15, zorder=10)
    trail, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
    
    ax.set_xlim(config.x_min - 0.5, config.x_max + 0.5)
    ax.set_ylim(config.y_min - 0.5, config.y_max + 0.5)
    ax.set_aspect('equal')
    ax.set_title('Oracle MPC: Navigation to Target')
    
    def init():
        agent.set_data([], [])
        trail.set_data([], [])
        return agent, trail
    
    def animate(i):
        agent.set_data([states[i, 0]], [states[i, 1]])
        trail.set_data(states[:i+1, 0], states[:i+1, 1])
        ax.set_title(f'Oracle MPC: Step {i}/{len(states)-1}')
        return agent, trail
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(states), interval=1000/fps, blit=True
    )
    
    if filename:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")
    
    plt.close()
    return anim


# =============================================================================
# Experiments
# =============================================================================

def run_experiments():
    """
    Run verification experiments for the Oracle Planner.
    """
    print("=" * 70)
    print("ORACLE PLANNER: Model Predictive Control with Ground Truth Model")
    print("=" * 70)
    
    config = MiniverseConfig(
        gravity=-2.0,
        drag=0.02,
        thrust_scale=5.0,
        restitution=0.7,
    )
    
    # --- Experiment 1: Reach center-bottom target ---
    print("\n[Experiment 1] Navigate to center-bottom (5.0, 0.5)")
    print("-" * 50)
    
    episode1 = run_mpc_episode(
        config=config,
        target=np.array([5.0, 0.5]),
        max_steps=300,
        horizon=20,
        num_samples=1000,
        reward_fn=reward_reach_target,
        seed=42,
    )
    
    # --- Experiment 2: Different target (corner) ---
    print("\n[Experiment 2] Navigate to corner (8.0, 1.0)")
    print("-" * 50)
    
    episode2 = run_mpc_episode(
        config=config,
        target=np.array([8.0, 1.0]),
        max_steps=300,
        horizon=20,
        num_samples=1000,
        reward_fn=reward_reach_target,
        seed=123,
    )
    
    # --- Experiment 3: Soft landing task ---
    print("\n[Experiment 3] Soft landing at (3.0, 0.5)")
    print("-" * 50)
    
    episode3 = run_mpc_episode(
        config=config,
        target=np.array([3.0, 0.5]),
        max_steps=400,
        horizon=25,
        num_samples=1500,
        reward_fn=reward_soft_landing,
        seed=456,
    )
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    experiments = [
        ("Center-bottom (5.0, 0.5)", episode1),
        ("Corner (8.0, 1.0)", episode2),
        ("Soft landing (3.0, 0.5)", episode3),
    ]
    
    for name, ep in experiments:
        status = "SUCCESS" if ep['success'] else "FAILED"
        print(f"  {name}: {status}")
        print(f"    Final distance: {ep['final_distance']:.4f}")
        print(f"    Final speed:    {ep['final_speed']:.4f}")
        print(f"    Steps:          {len(ep['actions'])}")
    
    # --- Visualizations ---
    print("\n[Generating visualizations...]")
    
    plot_mpc_episode(episode1, "Experiment 1: Center-Bottom Target")
    plot_mpc_episode(episode2, "Experiment 2: Corner Target")
    plot_mpc_episode(episode3, "Experiment 3: Soft Landing")
    
    # --- Create animation for one episode ---
    print("\n[Creating animation...]")
    create_mpc_animation(episode1, "oracle_mpc_navigation.gif", fps=30)
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("If the oracle planner succeeds, the environment is controllable.")
    print("Next: Replace GroundTruthWorldModel with a learned model.")
    print("=" * 70)
    
    return episode1, episode2, episode3


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_experiments()

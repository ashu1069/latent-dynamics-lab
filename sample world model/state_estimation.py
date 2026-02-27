"""
State Estimation for Miniverse using Kalman Filtering

This module extends the 1D Kalman filter from precursors/ to handle the 4D
Miniverse state: [x, y, vx, vy]. It demonstrates how state estimation
connects to World Models — the filter acts as a simple "world model" that
predicts and corrects state estimates.

Key Insight for World Models:
    - Kalman Filter = Linear World Model with analytical solution
    - VAE + RNN = Nonlinear World Model learned from data
    - Both solve the same problem: infer true state from noisy observations

State-Space Model:
    s_{t+1} = A @ s_t + B @ u_t + g + process_noise
    o_t = H @ s_t + observation_noise

Where:
    s = [x, y, vx, vy]^T  (state)
    u = [thrust_x, thrust_y]^T  (control)
    g = gravity contribution
    A = state transition matrix (encodes position-velocity dynamics + drag)
    B = control input matrix
    H = observation matrix (identity for direct observation)

Author: World Models Research
"""

import numpy as np
import matplotlib.pyplot as plt
from miniverse import MiniverseConfig, simulate_trajectory, step, reset, DEFAULT_CONFIG


# =============================================================================
# 4D Kalman Filter for Miniverse
# =============================================================================

class KalmanFilter4D:
    """
    4D Kalman Filter for the Miniverse Lander.
    
    Extends the 1D Kalman filter concept to handle:
    - 4D state: [x, y, vx, vy]
    - Control inputs: [thrust_x, thrust_y]
    - Position-velocity coupling in dynamics
    - Gravity and drag effects
    
    This is a LINEAR Kalman filter. It assumes the Miniverse dynamics
    are approximately linear (which they are, except for boundary collisions).
    """
    
    def __init__(self, config: MiniverseConfig = None):
        """
        Initialize Kalman Filter with Miniverse configuration.
        
        Args:
            config: Miniverse configuration (uses DEFAULT_CONFIG if None)
        """
        if config is None:
            config = DEFAULT_CONFIG
        
        self.config = config
        self.dt = config.dt
        
        # Build system matrices from Miniverse physics
        self._build_system_matrices()
        
        # Initialize belief state
        self.mu = None  # Mean (4D vector)
        self.P = None   # Covariance (4x4 matrix)
        
    def _build_system_matrices(self):
        """
        Build state-space matrices from Miniverse physics.
        
        Miniverse Euler integration:
            vx_new = vx + (thrust_scale * ax - drag * vx) * dt
            vy_new = vy + (thrust_scale * ay - drag * vy + gravity) * dt
            x_new = x + vx_new * dt
            y_new = y + vy_new * dt
            
        This translates to: s_{t+1} = A @ s_t + B @ u_t + g
        """
        dt = self.dt
        drag = self.config.drag
        thrust_scale = self.config.thrust_scale
        gravity = self.config.gravity
        
        # Velocity damping factor
        damp = 1.0 - drag * dt
        
        # State transition matrix A
        # Note: position update uses NEW velocity, so we get dt * damp for position
        self.A = np.array([
            [1.0,  0.0,  damp * dt,  0.0       ],  # x_new = x + vx_new * dt
            [0.0,  1.0,  0.0,        damp * dt ],  # y_new = y + vy_new * dt
            [0.0,  0.0,  damp,       0.0       ],  # vx_new = damp * vx + ...
            [0.0,  0.0,  0.0,        damp      ],  # vy_new = damp * vy + ...
        ])
        
        # Control input matrix B
        # u = [thrust_x, thrust_y]
        # Effect on velocity: thrust_scale * u * dt
        # Effect on position: thrust_scale * u * dt^2 (through new velocity)
        self.B = np.array([
            [thrust_scale * dt**2,  0.0                  ],
            [0.0,                   thrust_scale * dt**2 ],
            [thrust_scale * dt,     0.0                  ],
            [0.0,                   thrust_scale * dt    ],
        ])
        
        # Gravity vector (constant offset)
        # Gravity affects vy, and through vy affects y
        self.g = np.array([
            0.0,              # no gravity effect on x
            gravity * dt**2,  # gravity effect on y (through vy)
            0.0,              # no gravity effect on vx
            gravity * dt,     # gravity effect on vy
        ])
        
        # Observation matrix H (we observe all states directly)
        self.H = np.eye(4)
        
        # Process noise covariance Q
        # Model uncertainty in dynamics (e.g., unmodeled effects, collisions)
        # Tuned based on expected model error
        pos_process_noise = 0.01
        vel_process_noise = 0.1
        self.Q = np.diag([
            pos_process_noise,
            pos_process_noise,
            vel_process_noise,
            vel_process_noise,
        ])
        
        # Observation noise covariance R
        # Based on Miniverse observation noise
        self.R = np.diag([
            self.config.pos_noise_std**2,
            self.config.pos_noise_std**2,
            self.config.vel_noise_std**2,
            self.config.vel_noise_std**2,
        ])
    
    def reset(self, mu0: np.ndarray = None, P0: np.ndarray = None):
        """
        Reset filter to initial belief.
        
        Args:
            mu0: Initial mean (default: center of arena)
            P0: Initial covariance (default: high uncertainty)
        """
        if mu0 is None:
            # Default: center of arena, stationary, with high uncertainty
            mu0 = np.array([
                (self.config.x_min + self.config.x_max) / 2,
                (self.config.y_min + self.config.y_max) / 2,
                0.0,
                0.0,
            ])
        
        if P0 is None:
            # Default: high initial uncertainty
            P0 = np.diag([2.0, 2.0, 1.0, 1.0])
        
        self.mu = mu0.copy()
        self.P = P0.copy()
        
        return self.mu, self.P
    
    def predict(self, action: np.ndarray) -> tuple:
        """
        Prediction step: propagate belief through dynamics.
        
        This is the "world model" component — predicting the next state
        given current belief and action.
        
        Args:
            action: [thrust_x, thrust_y] control input
        
        Returns:
            mu_pred: Predicted mean
            P_pred: Predicted covariance
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Predicted mean: s_{t+1} = A @ s_t + B @ u_t + g
        mu_pred = self.A @ self.mu + self.B @ action + self.g
        
        # Predicted covariance: P_{t+1} = A @ P_t @ A^T + Q
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        return mu_pred, P_pred
    
    def update(self, observation: np.ndarray, mu_pred: np.ndarray, P_pred: np.ndarray) -> tuple:
        """
        Update step: correct prediction using observation.
        
        This is the "perception" component — integrating new sensory
        evidence to refine the state estimate.
        
        Args:
            observation: Noisy observation [x, y, vx, vy]
            mu_pred: Predicted mean (from predict step)
            P_pred: Predicted covariance (from predict step)
        
        Returns:
            mu: Updated (posterior) mean
            P: Updated (posterior) covariance
        """
        # Innovation (measurement residual)
        y = observation - self.H @ mu_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Updated belief
        self.mu = mu_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        # Ensure P stays symmetric and positive semi-definite
        self.P = (self.P + self.P.T) / 2
        
        return self.mu, self.P
    
    def step(self, action: np.ndarray, observation: np.ndarray) -> tuple:
        """
        Full Kalman filter step: predict then update.
        
        Args:
            action: Control input [thrust_x, thrust_y]
            observation: Noisy observation [x, y, vx, vy]
        
        Returns:
            mu: Posterior mean (denoised state estimate)
            P: Posterior covariance (estimation uncertainty)
        """
        mu_pred, P_pred = self.predict(action)
        mu, P = self.update(observation, mu_pred, P_pred)
        return mu, P


def filter_trajectory(trajectory: dict, config: MiniverseConfig = None) -> dict:
    """
    Apply Kalman filter to denoise a Miniverse trajectory.
    
    Args:
        trajectory: Dict with 'states', 'observations', 'actions'
        config: Miniverse configuration
    
    Returns:
        Dict with filtered estimates added:
            'estimates': (T+1, 4) filtered state estimates
            'covariances': (T+1, 4, 4) estimation covariances
            'predictions': (T, 4) one-step-ahead predictions
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    observations = trajectory['observations']
    actions = trajectory['actions']
    T = len(actions)
    
    # Initialize filter with first observation
    kf = KalmanFilter4D(config)
    kf.reset(mu0=observations[0], P0=np.diag([0.5, 0.5, 0.3, 0.3]))
    
    estimates = [kf.mu.copy()]
    covariances = [kf.P.copy()]
    predictions = []
    
    for t in range(T):
        # Predict
        mu_pred, P_pred = kf.predict(actions[t])
        predictions.append(mu_pred.copy())
        
        # Update with observation
        kf.update(observations[t + 1], mu_pred, P_pred)
        
        estimates.append(kf.mu.copy())
        covariances.append(kf.P.copy())
    
    return {
        **trajectory,
        'estimates': np.array(estimates),
        'covariances': np.array(covariances),
        'predictions': np.array(predictions),
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_rmse(true_states: np.ndarray, estimates: np.ndarray) -> dict:
    """
    Compute RMSE between true states and estimates.
    
    Returns:
        Dict with RMSE for each state component and total
    """
    errors = true_states - estimates
    
    return {
        'x': np.sqrt(np.mean(errors[:, 0]**2)),
        'y': np.sqrt(np.mean(errors[:, 1]**2)),
        'vx': np.sqrt(np.mean(errors[:, 2]**2)),
        'vy': np.sqrt(np.mean(errors[:, 3]**2)),
        'position': np.sqrt(np.mean(errors[:, 0]**2 + errors[:, 1]**2)),
        'velocity': np.sqrt(np.mean(errors[:, 2]**2 + errors[:, 3]**2)),
        'total': np.sqrt(np.mean(np.sum(errors**2, axis=1))),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_filtering_results(filtered_traj: dict, config: MiniverseConfig = None):
    """
    Visualize Kalman filtering results: true state vs observation vs estimate.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    states = filtered_traj['states']
    obs = filtered_traj['observations']
    estimates = filtered_traj['estimates']
    covariances = filtered_traj['covariances']
    
    T = len(states)
    t = np.arange(T)
    
    labels = ['x', 'y', 'vx', 'vy']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, ax in enumerate(axes.flat):
        # True state
        ax.plot(t, states[:, i], 'b-', linewidth=2, label='True state', alpha=0.8)
        
        # Noisy observations
        ax.scatter(t, obs[:, i], c='red', s=8, alpha=0.3, label='Observations')
        
        # Kalman estimate
        ax.plot(t, estimates[:, i], 'g-', linewidth=2, label='KF estimate', alpha=0.8)
        
        # Uncertainty band (±2σ)
        sigma = np.sqrt(covariances[:, i, i])
        ax.fill_between(
            t,
            estimates[:, i] - 2 * sigma,
            estimates[:, i] + 2 * sigma,
            color='green', alpha=0.2, label='±2σ'
        )
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i]}: True vs Observed vs Filtered')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Filter Denoising Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_2d_comparison(filtered_traj: dict, config: MiniverseConfig = None):
    """
    Compare trajectories in 2D space: true, observed, and filtered.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    states = filtered_traj['states']
    obs = filtered_traj['observations']
    estimates = filtered_traj['estimates']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Arena boundary
    from matplotlib.patches import Rectangle
    arena = Rectangle(
        (config.x_min, config.y_min),
        config.x_max - config.x_min,
        config.y_max - config.y_min,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(arena)
    
    # Observations (noisy)
    ax.scatter(obs[:, 0], obs[:, 1], c='red', s=15, alpha=0.3, label='Observations', zorder=1)
    
    # True trajectory
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2.5, label='True trajectory', alpha=0.8, zorder=2)
    
    # Filtered trajectory
    ax.plot(estimates[:, 0], estimates[:, 1], 'g--', linewidth=2.5, label='KF estimate', alpha=0.8, zorder=3)
    
    # Start/End markers
    ax.scatter(*states[0, :2], c='green', s=150, marker='o', zorder=5, edgecolor='black', label='Start')
    ax.scatter(*states[-1, :2], c='blue', s=150, marker='X', zorder=5, edgecolor='black', label='End')
    
    ax.set_xlim(config.x_min - 0.5, config.x_max + 0.5)
    ax.set_ylim(config.y_min - 0.5, config.y_max + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Trajectory Comparison: True vs Observed vs Filtered')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_accuracy(filtered_traj: dict, config: MiniverseConfig = None):
    """
    Analyze prediction accuracy: how well does the filter predict the next state?
    
    This is directly relevant to World Models — the prediction step
    is analogous to what the MDN-RNN learns to do.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    states = filtered_traj['states']
    predictions = filtered_traj['predictions']
    
    # Predictions are for states[1:T+1], so align them
    true_next = states[1:]
    T = len(predictions)
    t = np.arange(T)
    
    pred_errors = true_next - predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Prediction errors over time
    ax = axes[0]
    labels = ['x', 'y', 'vx', 'vy']
    for i, label in enumerate(labels):
        ax.plot(t, pred_errors[:, i], label=f'{label} error', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Prediction Error')
    ax.set_title('One-Step Prediction Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution (histogram)
    ax = axes[1]
    ax.hist(pred_errors[:, 0], bins=30, alpha=0.5, label='x', density=True)
    ax.hist(pred_errors[:, 1], bins=30, alpha=0.5, label='y', density=True)
    ax.hist(pred_errors[:, 2], bins=30, alpha=0.5, label='vx', density=True)
    ax.hist(pred_errors[:, 3], bins=30, alpha=0.5, label='vy', density=True)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Accuracy Analysis (World Model Perspective)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STATE ESTIMATION: Kalman Filter for Miniverse")
    print("=" * 70)
    
    # Configuration with moderate observation noise
    config = MiniverseConfig(
        gravity=-2.0,
        drag=0.02,
        thrust_scale=5.0,
        pos_noise_std=0.3,   # Higher noise to make filtering more visible
        vel_noise_std=0.15,
        restitution=0.7,
    )
    
    print("\nConfiguration:")
    print(f"  Position noise σ: {config.pos_noise_std}")
    print(f"  Velocity noise σ: {config.vel_noise_std}")
    
    # --- Generate trajectory with hover policy ---
    print("\n[1] Generating trajectory with hover policy...")
    
    def hover_policy(state, target_y=6.0):
        x, y, vx, vy = state
        thrust_y = 0.5 * (target_y - y) - 0.3 * vy
        thrust_x = -0.2 * (x - 5.0) - 0.3 * vx
        return np.array([thrust_x, thrust_y])
    
    trajectory = simulate_trajectory(
        T=400,
        policy=hover_policy,
        config=config,
        seed=42,
    )
    
    # --- Apply Kalman filter ---
    print("[2] Applying Kalman filter...")
    filtered = filter_trajectory(trajectory, config)
    
    # --- Compute metrics ---
    print("\n[3] Computing RMSE metrics...")
    
    rmse_obs = compute_rmse(filtered['states'], filtered['observations'])
    rmse_kf = compute_rmse(filtered['states'], filtered['estimates'])
    
    print("\n  RMSE (Observations vs True):")
    print(f"    Position: {rmse_obs['position']:.4f}")
    print(f"    Velocity: {rmse_obs['velocity']:.4f}")
    print(f"    Total:    {rmse_obs['total']:.4f}")
    
    print("\n  RMSE (KF Estimate vs True):")
    print(f"    Position: {rmse_kf['position']:.4f}")
    print(f"    Velocity: {rmse_kf['velocity']:.4f}")
    print(f"    Total:    {rmse_kf['total']:.4f}")
    
    improvement = (1 - rmse_kf['total'] / rmse_obs['total']) * 100
    print(f"\n  Noise reduction: {improvement:.1f}%")
    
    # --- Visualizations ---
    print("\n[4] Generating visualizations...")
    
    plot_filtering_results(filtered, config)
    plot_2d_comparison(filtered, config)
    plot_prediction_accuracy(filtered, config)
    
    print("\n" + "=" * 70)
    print("Connection to World Models:")
    print("  - KF Predict step ≈ MDN-RNN transition model")
    print("  - KF Update step ≈ VAE encoder (observation → latent)")
    print("  - The filter maintains a 'belief state' (μ, Σ)")
    print("  - World Models will learn this implicitly in latent space")
    print("=" * 70)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# -----------------------------
# TRUE environment (unknown dynamics)
# -----------------------------
def f_true(x, a, noise_std=0.02):
    return 0.9*x + 0.2*np.sin(x) + a + noise_std*np.random.randn()

# -----------------------------
# Learned model class: linear regression
# x_{t+1} ≈ theta0 + theta_x * x_t + theta_a * a_t
# -----------------------------
def fit_linear_dynamics(xs, a_s, xnexts):
    # Design matrix: [1, x, a]
    X = np.stack([np.ones_like(xs), xs, a_s], axis=1)  # shape (N,3)
    y = xnexts.reshape(-1, 1)                          # shape (N,1)
    # theta = (X^T X)^{-1} X^T y  (least squares)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta.flatten()  # [theta0, theta_x, theta_a]

def f_hat(theta, x, a):
    return theta[0] + theta[1]*x + theta[2]*a

# -----------------------------
# MPC via random shooting
# -----------------------------
def mpc_action(theta, x0, H=15, K=2000, a_max=1.0, q=1.0, r=0.05):
    """
    Sample K candidate action sequences length H, simulate under learned model,
    choose lowest predicted cost. Return first action.
    """
    best_cost = float("inf")
    best_a0 = 0.0

    # Sample actions uniformly in [-a_max, a_max]
    A = np.random.uniform(-a_max, a_max, size=(K, H))

    for k in range(K):
        x = x0
        cost = 0.0
        for t in range(H):
            a = A[k, t]
            cost += q*(x*x) + r*(a*a)
            x = f_hat(theta, x, a)
        if cost < best_cost:
            best_cost = cost
            best_a0 = A[k, 0]

    return best_a0, best_cost

# -----------------------------
# 1) DATA COLLECTION (random actions)
# -----------------------------
def collect_data(N=2000, a_max=1.0):
    xs = []
    a_s = []
    xnexts = []
    x = 1.5  # start away from zero

    for _ in range(N):
        a = np.random.uniform(-a_max, a_max)
        x_next = f_true(x, a)
        xs.append(x)
        a_s.append(a)
        xnexts.append(x_next)
        x = x_next
    return np.array(xs), np.array(a_s), np.array(xnexts)

xs, a_s, xnexts = collect_data(N=2500, a_max=0.6)

theta = fit_linear_dynamics(xs, a_s, xnexts)
print("Learned theta [bias, x_coeff, a_coeff] =", theta)

# -----------------------------
# 2) CONTROL with MPC (learned model)
# -----------------------------
T = 80
H = 18
K = 2500

# Mismatch knob: during control, we can change the true system slightly
def f_true_mismatched(x, a, noise_std=0.02):
    # stronger nonlinearity + slight gain shift (distribution shift)
    return 0.85*x + 0.35*np.sin(1.2*x) + a + noise_std*np.random.randn()

x = 2.2  # initial state for control
traj = [x]
acts = []
pred_costs = []

for t in range(T):
    a, predJ = mpc_action(theta, x, H=H, K=K, a_max=1.0, q=1.0, r=0.05)
    # execute in the REAL environment (mismatched)
    x = f_true_mismatched(x, a)
    traj.append(x)
    acts.append(a)
    pred_costs.append(predJ)

traj = np.array(traj)

# -----------------------------
# 3) Compare against "oracle MPC" that uses the TRUE model (for reference)
# -----------------------------
def mpc_action_oracle(x0, H=15, K=2000, a_max=1.0, q=1.0, r=0.05):
    best_cost = float("inf")
    best_a0 = 0.0
    A = np.random.uniform(-a_max, a_max, size=(K, H))
    for k in range(K):
        x = x0
        cost = 0.0
        for t in range(H):
            a = A[k, t]
            cost += q*(x*x) + r*(a*a)
            x = f_true_mismatched(x, a, noise_std=0.0)  # deterministic planning
        if cost < best_cost:
            best_cost = cost
            best_a0 = A[k, 0]
    return best_a0

x2 = 2.2
traj_oracle = [x2]
acts_oracle = []
for t in range(T):
    a2 = mpc_action_oracle(x2, H=H, K=K, a_max=1.0, q=1.0, r=0.05)
    x2 = f_true_mismatched(x2, a2)
    traj_oracle.append(x2)
    acts_oracle.append(a2)

traj_oracle = np.array(traj_oracle)

# -----------------------------
# 4) PLOTS
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(traj, label="MPC using learned model (biased)")
plt.plot(traj_oracle, label="MPC using true model (oracle)", alpha=0.85)
plt.axhline(0.0, linestyle="--")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Model Predictive Control: failure under model mismatch")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3.5))
plt.plot(acts, label="actions (learned-model MPC)")
plt.plot(acts_oracle, label="actions (oracle MPC)", alpha=0.85)
plt.axhline(0.0, linestyle="--")
plt.xlabel("t")
plt.ylabel("a")
plt.title("Control signals")
plt.legend()
plt.tight_layout()
plt.show()

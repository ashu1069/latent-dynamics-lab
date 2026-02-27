import math, random
import matplotlib.pyplot as plt

# Nonlinear System
# x_{t+1} = sin(x_t) + a_t _ eps, eps ~ N(0, Q)
# o_t = x_t^2 + eta, eta ~ N(0, R)

def randn():
    u1 = max(1e-12, random.random())
    u2 = random.random()
    z = math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2)
    return z

def normal_pdf(x, mean, variance):
    if variance <= 0.0:
        return 0.0
    return (1.0 / math.sqrt(2.0+math.pi*variance)) * math.exp(-0.5*(x-mean)**2/variance)

def simulate_nonlinear_system(T=120, x0 = 0.7, Q=0.12**2, R=0.25**2):
    x = [0.0]*(T+1)
    o = [0.0]*(T+1)
    a = [0.0]*(T+1)

    x[0] = x0
    o[0] = x[0]**2 + math.sqrt(R)*randn()

    for t in range(T):
        a[t] = 0.2 * math.sin(2.0*math.pi*t/T) - 0.02
        x[t+1] = math.sin(x[t]) + a[t] + math.sqrt(Q)*randn()
        o[t+1] = x[t+1]**2 + math.sqrt(R)*randn()

    return x, o, a

def ekf_1d(o, a, mu0=0.0, sigma0=1.0, Q = 0.12**2, R=0.25**2):
    '''
    Extended Kalman filter for 1D non-linear dynamic system.
    Belief at time t: z_t ~ N(mu_t, sigma_t)
    Args:
        o: observations
        a: controls
        mu0: initial mean
        sigma0: initial variance
        Q: process noise variance
        R: observation noise variance
    '''

    T = len(o) - 1
    mu = [0.0]*(T+1)
    P_sigma = [0.0]*(T+1)

    mu[0], P_sigma[0] = mu0, sigma0

    for t in range(T):
        # Prediction step
        mu_pred = math.sin(mu[t]) + a[t]
        F = math.cos(mu[t])       # df/dx at mu[t]
        P_pred = F * P_sigma[t] * F + Q

        # Update with o_{t+1}
        H = 2.0 * mu_pred      # dh/dx at mu_pred
        z_pred = mu_pred * mu_pred
        y = o[t+1] - z_pred

        S = H * P_pred * H + R
        if S < 1e-12:
            mu[t+1], P_sigma[t+1] = mu_pred, max(P_pred, 1e-12)
            continue

        K = (P_pred * H) / S
        mu[t+1] = mu_pred + K * y
        P_sigma[t+1] = (1 - K * H) * P_pred
        P_sigma[t+1] = max(P_sigma[t+1], 1e-12)

    return mu, P_sigma

# Particle Filter for 1D nonlinear dynamic system.

def systematic_resample(particles, weights):
    N = len(particles)
    positions = [(random.random() + i) / N for i in range(N)]

    cumsum = []
    s = 0.0

    for w in weights:
        s += w
        cumsum.append(s)

    new_particles = [0.0]*N
    i=0
    for j, pos in enumerate(positions):
        while pos > cumsum[i]:
            i += 1
            if i >= N:
                i = N-1
                break
        new_particles[j] = particles[i]
    return new_particles

def particle_filter_1d(o, a, N=2000, x0_mean = 0.0, x0_std=1.0, Q=0.12**2, R=0.25**2, resample_threshold=0.5, seed=42):
    '''
    Particle filter for 1D nonlinear dynamic system.
    Belief at time t: x_t ~ sum_i w_i * delta(x_t - x_i)
    Args:
        o: observations
        a: controls
        N: number of particles
        x0_mean: initial mean
        x0_std: initial standard deviation
        Q: process noise variance
        R: observation noise variance
        resample_threshold: resample threshold
        seed: random seed
    Returns:
        mu: posterior mean
        P_sigma: posterior variance
        particles: particles
        weights: weights
    '''
    random.seed(seed)
    T = len(o) - 1
    particles = [x0_mean + x0_std*randn() for _ in range(N)]
    weights = [1.0/N]*(N)

    mean = [0.0]*(T+1)
    std = [0.0]*(T+1)

    def moments(ps, ws):
        m = sum(p*w for p,w in zip(ps, ws))
        v = sum(((p-m)**2)*w for p,w in zip(ps, ws))
        return m, math.sqrt(max(0.0, v))

    # weight using o_0
    for i in range(N):
        weights[i] *= normal_pdf(o[0], particles[i]**2, R)

    s = sum(weights)
    weights = [w/s for w in weights] if s != 0.0 else [1.0/N]*(N)
    mean[0], std[0] = moments(particles, weights)

    for t in range(T):
        # Propagate particles
        for i in range(N):
            particles[i] = math.sin(particles[i]) + a[t] + math.sqrt(Q)*randn()
        # Re-weight using 0_{t+1}
        for i in range(N):
            weights[i] = normal_pdf(o[t+1], particles[i]**2, R)

        s = sum(weights)
        weights = [w/s for w in weights] if s != 0.0 else [1.0/N]*(N)
        mean[t+1], std[t+1] = moments(particles, weights)

        # Resample if necessary or degeneracy
        ess = 1.0 / sum(w**2 for w in weights)
        if ess < resample_threshold * N:
            particles = systematic_resample(particles, weights)
            weights = [1.0/N]*(N)

        mean[t+1], std[t+1] = moments(particles, weights)

    return mean, std, particles, weights

def rmse(x, x_hat):
    return math.sqrt(sum((a-b)**2 for a,b in zip(x, x_hat))/len(x))

# ---- Run experiment ----
random.seed(42)

T = 140
Q = 0.12**2
R = 0.25**2

x_true, o, a = simulate_nonlinear_system(T=T, x0=0.9, Q=Q, R=R)

mu_ekf, P_ekf = ekf_1d(o, a, mu0=0.0, sigma0=1.2**2, Q=Q, R=R)
sig_ekf = [math.sqrt(p) for p in P_ekf]

mu_pf, sig_pf, particles, weights = particle_filter_1d(o, a, N=2500, x0_mean=0.0, x0_std=1.2, Q=Q, R=R)

rmse_ekf = rmse(x_true, mu_ekf)
rmse_pf  = rmse(x_true, mu_pf)

print("Accuracy (lower is better):")
print(f"  EKF RMSE: {rmse_ekf:.4f}")
print(f"  PF  RMSE: {rmse_pf:.4f}")

# ---- Plot state estimates ----
t = list(range(T+1))

ekf_upper = [m + 2*s for m, s in zip(mu_ekf, sig_ekf)]
ekf_lower = [m - 2*s for m, s in zip(mu_ekf, sig_ekf)]
pf_upper  = [m + 2*s for m, s in zip(mu_pf, sig_pf)]
pf_lower  = [m - 2*s for m, s in zip(mu_pf, sig_pf)]

plt.figure(figsize=(11, 4.9))
plt.plot(t, x_true, label="true $x_t$")
plt.plot(t, mu_ekf, label="EKF mean")
plt.fill_between(t, ekf_lower, ekf_upper, alpha=0.15, label="EKF $\\pm 2\\sigma$")
plt.plot(t, mu_pf, label="PF mean")
plt.fill_between(t, pf_lower, pf_upper, alpha=0.15, label="PF $\\pm 2\\sigma$")
plt.xlabel("t")
plt.ylabel("state")
plt.title("Nonlinear filtering: EKF vs Particle Filter")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# ---- Plot observation space (reveals x^2 ambiguity) ----
plt.figure(figsize=(11, 4.2))
plt.scatter(t, o, s=12, alpha=0.35, label="observations $o_t = x_t^2 + \\eta$")
plt.plot(t, [m*m for m in mu_ekf], label="EKF predicted obs $h(\\mu_t)$")
plt.plot(t, [m*m for m in mu_pf], label="PF predicted obs $h(\\mu_t)$")
plt.xlabel("t")
plt.ylabel("observation")
plt.title("Observation space: ambiguity from $x^2$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
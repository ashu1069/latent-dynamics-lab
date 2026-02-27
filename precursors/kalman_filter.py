import math, random
import matplotlib.pyplot as plt

# 1D Kalman filter from scratch
# x_{t+1} = x_t + a_t + eps, eps ~N(0,Q)
# o_t = x_t + eta, eta ~ N(0,R)

def randn():
    u1 = max(1e-12, random.random())
    u2 = random.random()
    z = math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2)
    return z

def simulate(T=100, x0=0.0, Q=0.2**2, R=0.6**2):
    '''
    Simulate true states x_t and observations o_t, with controls a_t.

    Args:
        T: number of time steps
        x0: initial state
        Q: process noise variance
        R: observation noise variance
    Returns:
        x: true states
        o: observations
        a: controls
    '''

    x = [0.0]*(T+1)
    o = [0.0]*(T+1)
    a = [0.0]*(T+1)

    x[0] = x0
    o[0] = x[0] + math.sqrt(R)*randn()

    # controls
    for t in range(T):
        a[t] = 0.2 * math.sin(2.0*math.pi*t/T) + 0.02
        x[t+1] = x[t] + a[t] + math.sqrt(Q)*randn()
        o[t+1] = x[t+1] + math.sqrt(R)*randn()

    return x, o, a

def kalman_filter_1d(o, a, mu0=0.0, sigma0=1.0, Q=0.2**2, R=0.6**2):
    '''
    Kalman filter for 1D linear dynamic system.
    Belief at time t: x_t ~ N(mu_t, sigma_t)
    Args:
        o: observations
        a: controls
        mu0: initial mean
        sigma0: initial variance
        Q: process noise variance
        R: observation noise variance
    Returns:
        mu: posterior mean
        P_sigma: posterior variance
    '''
    T = len(o) - 1
    mu = [0.0]*(T+1)
    P_sigma = [0.0]*(T+1)

    mu[0], P_sigma[0] = mu0, sigma0

    for t in range(T):
        # Prediction step
        mu_pred = mu[t] + a[t]
        P_pred = P_sigma[t] + Q

        # Update step, correction using observation o_{t+1}
        y = o[t+1] - mu_pred # innovation
        S = P_pred + R # innovation covariance
        K = P_pred / S # Kalman gain

        mu[t+1] = mu_pred + K * y
        P_sigma[t+1] = (1-K) * P_pred

    return mu, P_sigma

random.seed(42)
T=120
Q = 0.15**2
R = 0.50**2

x_true, o, a = simulate(T, x0=-1.0, Q=Q, R=R)
mu,P_sigma = kalman_filter_1d(o, a, mu0=0.0, sigma0=1.5**2, Q=Q, R=R)

t = list(range(T+1))
sigma = [math.sqrt(max(0.0, P)) for P in P_sigma]
upper = [m + 2*s for m,s in zip(mu, sigma)]
lower = [m - 2*s for m,s in zip(mu, sigma)]

plt.figure(figsize=(10, 6))
plt.plot(t, x_true, label="true $x_t$")
plt.plot(t, mu, label="belief mean $\\mu_t$")
plt.fill_between(t, lower, upper, alpha=0.2, label="$\\mu_t \\pm 2\\sigma_t$")
plt.scatter(t, o, s=10, alpha=0.25, label="observations $o_t$")
plt.xlabel("t")
plt.ylabel("state / observation")
plt.title("1D Kalman Filter: random-walk + control, noisy observations")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

"""
Post 1 — Partial observability: we need a state, not raw history.

POMDP: hidden s_t, observation o_t. Belief b_t = P(s_t | h_t).
Recursion: b_{t+1} ∝ P(o_{t+1}|s') * Σ_s P(s'|s,a) * b_t[s]
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 2-state POMDP: states 0,1; observations 0,1 (ambiguous)
P = np.array([[0.9, 0.1], [0.1, 0.9]])  # transition P(s'|s)
O = np.array([[0.8, 0.2], [0.3, 0.7]])  # emission P(o|s) — ambiguous

def step(s):
    s = np.random.choice(2, p=P[s])
    o = np.random.choice(2, p=O[s])
    return s, o

def belief_update(b, o):
    b = O[:, o] * (P.T @ b)
    return b / b.sum()

def generate(T=50):
    s = np.random.randint(2)
    b = np.array([0.5, 0.5])
    states, obs, beliefs = [s], [np.random.choice(2, p=O[s])], [b.copy()]
    for _ in range(T - 1):
        s, o = step(s)
        b = belief_update(b, o)
        states.append(s)
        obs.append(o)
        beliefs.append(b.copy())
    return np.array(states), np.array(obs), np.array(beliefs)


def plot(states, obs, beliefs):
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(states))
    ax[0].scatter(t, states, c=states, cmap='coolwarm', s=20)
    ax[0].set_ylabel('State $s_t$')
    ax[0].set_title('Hidden state')
    ax[1].scatter(t, obs, c=obs, cmap='viridis', s=20)
    ax[1].set_ylabel('Observation $o_t$')
    ax[1].set_title('Observations (ambiguous)')
    ax[2].plot(t, beliefs[:, 0], label='P(s=0)')
    ax[2].plot(t, beliefs[:, 1], label='P(s=1)')
    ax[2].set_ylabel('Belief $b_t$')
    ax[2].set_title('Belief: $h_t \\to b_t$')
    ax[2].legend()
    ax[2].set_ylim(0, 1)
    plt.xlabel('$t$')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import sys, os
    if "--save" in sys.argv:
        plt.switch_backend("Agg")

    states, obs, beliefs = generate(80)
    fig = plot(states, obs, beliefs)

    if "--save" in sys.argv:
        fig.savefig(os.path.join(os.path.dirname(__file__), "partial_observability.png"), dpi=150)
    else:
        plt.show()

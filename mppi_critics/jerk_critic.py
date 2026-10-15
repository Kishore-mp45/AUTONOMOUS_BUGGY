"""
mppi_critics/jerk_critic.py
===========================
Penalize high jerk (rate of change of acceleration) for passenger comfort.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class JerkCritic:
    """
    Computes jerk from the trajectory velocity sequence.
    Cost = MPPI_JERK_WEIGHT * sum of squared jerks across the horizon.
    """

    name = "jerk"

    def score(self, trajectories: np.ndarray) -> np.ndarray:
        """
        trajectories : shape (K, T, 4) — state [x, y, theta, v]
        Returns      : shape (K,)
        """
        K, T, _ = trajectories.shape
        if T < 3:
            return np.zeros(K)

        vs     = trajectories[:, :, 3]               # (K, T)
        accels = np.diff(vs, axis=1) / config.DT     # (K, T-1)
        jerks  = np.diff(accels, axis=1) / config.DT # (K, T-2)
        return config.MPPI_JERK_WEIGHT * np.sum(jerks ** 2, axis=1)


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    crit = JerkCritic()
    T = 10
    # Constant speed (zero jerk) vs oscillating speed (high jerk)
    smooth = np.stack([np.arange(T) * 0.5, np.zeros(T), np.zeros(T),
                       np.full(T, 1.0)], axis=-1)[None]
    jerky = np.stack([np.arange(T) * 0.5, np.zeros(T), np.zeros(T),
                      1.0 + 0.3 * np.sin(np.arange(T) * 3.0)], axis=-1)[None]
    costs = crit.score(np.concatenate([smooth, jerky]))
    assert costs[0] < costs[1] and costs[0] == 0.0
    print(f"[PASS] JerkCritic: smooth={costs[0]:.3f}, jerky={costs[1]:.3f}")

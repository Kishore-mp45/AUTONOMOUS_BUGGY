"""
mppi_critics/speed_critic.py
============================
Prevent unnecessary stopping — buggy should maintain a speed appropriate
to local obstacle density. Desired speed is MAX_SPEED on a clear path and
scales down inside the inflation zone.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class SpeedCritic:
    """
    Penalizes speed far from the desired speed.
    Near obstacles: desired speed is lower.
    Clear path: desired speed is MAX_SPEED.
    """

    name = "speed"

    def score(self, trajectories: np.ndarray, costmap) -> np.ndarray:
        """
        trajectories : shape (K, T, 4) — state [x, y, theta, v]
        costmap      : Costmap2D or None
        Returns      : shape (K,)
        """
        vs = trajectories[:, :, 3]                      # (K, T)

        if costmap is None:
            v_desired = np.full_like(vs, config.MAX_SPEED)
        else:
            xs = trajectories[:, :, 0].ravel()
            ys = trajectories[:, :, 1].ravel()
            cell = costmap.query_batch(xs, ys).reshape(vs.shape)
            # Desired speed reduces with local cost, floored at 20%
            v_desired = config.MAX_SPEED * np.maximum(0.2, 1.0 - cell / 100.0)

        err = (v_desired - vs) ** 2
        return config.MPPI_SPEED_WEIGHT * err.sum(axis=1)


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    crit = SpeedCritic()
    T = 10
    t = np.arange(T) * 0.5
    fast = np.stack([t, np.zeros(T), np.zeros(T), np.full(T, 1.5)], axis=-1)[None]
    slow = np.stack([t, np.zeros(T), np.zeros(T), np.full(T, 0.2)], axis=-1)[None]
    costs = crit.score(np.concatenate([fast, slow]), None)
    assert costs[0] < costs[1], "full-speed trajectory must be cheaper on clear road"
    print(f"[PASS] SpeedCritic: fast={costs[0]:.3f}, slow={costs[1]:.3f}")

"""
mppi_critics/goal_critic.py
===========================
Drive buggy toward goal. Cost decreases as distance to goal decreases.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class GoalCritic:
    """
    Cost = weighted distance from final trajectory state to goal,
    plus a smaller per-step progress cost at intermediate states.
    """

    name = "goal"

    def score(self, trajectories: np.ndarray, goal) -> np.ndarray:
        """
        trajectories : shape (K, T, 4)
        goal         : (gx, gy)
        Returns      : shape (K,)
        """
        gx, gy = goal
        K = trajectories.shape[0]

        # Terminal cost (heaviest weight)
        term_dist = np.hypot(trajectories[:, -1, 0] - gx,
                             trajectories[:, -1, 1] - gy)
        costs = config.MPPI_GOAL_WEIGHT * term_dist

        # Intermediate progress cost (lighter) — rewards closing distance early
        step_dist = np.hypot(trajectories[:, :, 0] - gx,
                             trajectories[:, :, 1] - gy)
        costs += 0.1 * step_dist.sum(axis=1)

        return costs


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    crit = GoalCritic()
    T = 10
    # Traj A ends at goal (10, 8.5); traj B ends 5 m away
    def mk(end_x):
        xs = np.linspace(0, end_x, T)
        return np.stack([xs, np.full(T, 8.5), np.zeros(T), np.ones(T)], axis=-1)[None]
    trajs = np.concatenate([mk(10.0), mk(5.0)])
    costs = crit.score(trajs, (10.0, 8.5))
    assert costs[0] < costs[1], "trajectory ending at goal must be cheaper"
    print(f"[PASS] GoalCritic: at-goal={costs[0]:.3f}, far={costs[1]:.3f}")

"""
mppi_critics/path_critic.py
===========================
Keep buggy near the reference path (cross-track error) and near the
maneuver-dependent target lane centre.
"""

import numpy as np
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class PathCritic:
    """
    Cost per step:
      - MPPI_PATH_WEIGHT * cross-track error to nearest reference path point
      - 0.5 * lateral deviation from the target lane centre y
        (target changes with maneuver mode via LaneMap.get_target_lane_y)
    """

    name = "path"

    def __init__(self, lane_map=None):
        # LaneMap injected by MPPI controller at construction
        self.lane_map = lane_map

    def score(
        self,
        trajectories: np.ndarray,
        reference_path: Optional[List[Tuple[float, float]]],
        lane_map=None,
        maneuver_mode: str = "normal",
        include_lane_cost: bool = True,
    ) -> np.ndarray:
        """
        trajectories   : shape (K, T, 4)
        reference_path : list of (x, y) waypoints (may be None/empty)
        lane_map       : LaneMap (falls back to self.lane_map)
        Returns        : shape (K,)
        """
        lane_map = lane_map or self.lane_map
        K, T, _ = trajectories.shape
        costs = np.zeros(K)

        # ── Cross-track error to reference path ──────────────────────
        if reference_path and len(reference_path) > 0:
            path_arr = np.asarray(reference_path, dtype=float)      # (N, 2)
            px = path_arr[:, 0][None, :]                            # (1, N)
            py = path_arr[:, 1][None, :]
            for t in range(T):
                xs = trajectories[:, t, 0][:, None]                 # (K, 1)
                ys = trajectories[:, t, 1][:, None]
                dists = np.hypot(xs - px, ys - py)                  # (K, N)
                cte = dists.min(axis=1)                             # (K,)
                costs += config.MPPI_PATH_WEIGHT * cte

        # ── Lane centering cost ──────────────────────────────────────
        if lane_map is not None and include_lane_cost:
            target_y = lane_map.get_target_lane_y(maneuver_mode)
            ys_all = trajectories[:, :, 1]                          # (K, T)
            costs += (0.5 * np.abs(ys_all - target_y)).sum(axis=1)

        return costs


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from environment.lane_map import LaneMap

    crit = PathCritic(LaneMap())
    path = [(float(x), 8.5) for x in range(0, 21)]
    T = 10
    xs = np.linspace(1, 5, T)
    on_path = np.stack([xs, np.full(T, 8.5), np.zeros(T), np.ones(T)], axis=-1)[None]
    off_path = on_path.copy(); off_path[:, :, 1] = 13.0
    trajs = np.concatenate([on_path, off_path])
    costs = crit.score(trajs, path, LaneMap(), "normal")
    assert costs[0] < costs[1], "on-path trajectory must be cheaper"
    print(f"[PASS] PathCritic: on={costs[0]:.3f}, off={costs[1]:.3f}")

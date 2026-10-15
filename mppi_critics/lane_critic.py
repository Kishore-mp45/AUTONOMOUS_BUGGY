"""
mppi_critics/lane_critic.py
===========================
Enforce Indian left-hand lane discipline.
This is the key new critic not in the previous system version.

  Normal mode:    left lane = zero-ish cost, right lane = LANE_COST_RIGHT per step
  Overtake mode:  right lane temporarily allowed (only boundary cost applies)
  U-turn mode:    rightmost position preferred before arc (boundary cost only)
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from config import ROAD_CENTERLINE_Y


class LaneCritic:
    """
    Applies lane discipline cost at every trajectory point.
    Uses LaneMap.lane_cost_batch() for vectorised evaluation.
    """

    name = "lane"

    def __init__(self, lane_map):
        self.lane_map = lane_map

    def score(
        self,
        trajectories: np.ndarray,
        maneuver_mode: str = "normal",
    ) -> np.ndarray:
        """
        trajectories   : shape (K, T, 4)
        maneuver_mode  : current maneuver mode string
        Returns        : shape (K,)
        """
        K, T, _ = trajectories.shape
        xs = trajectories[:, :, 0]
        ys = trajectories[:, :, 1]

        if maneuver_mode in ("normal", "left_turn", "return_left", "reverse"):
            # Full lane cost: left lane = free, right lane = penalty
            return self.lane_map.lane_cost_batch(xs, ys).sum(axis=1)

        if maneuver_mode in ("right_turn", "u_turn"):
            # Only penalize exiting the road (past road edges)
            return self.lane_map.boundary_cost_batch(xs, ys).sum(axis=1)

        if maneuver_mode == "overtake":
            # In right lane: lethal outside road, plus a small pull to
            # return left as soon as the obstacle is passed.
            costs = self.lane_map.boundary_cost_batch(xs, ys).sum(axis=1)
            overstay = np.maximum(0.0, ys - ROAD_CENTERLINE_Y)
            return costs + (5.0 * overstay).sum(axis=1)

        # Unknown mode → treat as normal
        return self.lane_map.lane_cost_batch(xs, ys).sum(axis=1)


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from environment.lane_map import LaneMap
    from config import LEFT_LANE_CENTER_Y, RIGHT_LANE_CENTER_Y

    crit = LaneCritic(LaneMap())
    T = 10
    t = np.arange(T) * 0.5
    def mk(y):
        return np.stack([t, np.full(T, y), np.zeros(T), np.ones(T)], axis=-1)[None]
    left  = mk(LEFT_LANE_CENTER_Y)
    right = mk(RIGHT_LANE_CENTER_Y)
    outside = mk(config.ROAD_Y_MIN - 0.1)

    # Normal mode: left free, right penalized, boundary lethal
    costs = crit.score(np.concatenate([left, right, outside]), "normal")
    assert costs[0] == 0.0, "left lane must be free in normal mode"
    assert costs[1] >= config.LANE_COST_RIGHT * T
    assert costs[2] >= config.LANE_COST_BOUNDARY * T

    # Overtake mode: right lane allowed, but returning-left pull applies
    c_overtake_right = crit.score(right, "overtake")[0]
    c_overtake_boundary = crit.score(outside, "overtake")[0]
    assert 0.0 < c_overtake_right < config.LANE_COST_RIGHT * T
    assert c_overtake_boundary >= config.LANE_COST_BOUNDARY * T

    # U-turn mode: only boundary cost
    assert crit.score(right, "u_turn")[0] == 0.0
    print("[PASS] LaneCritic verified!")

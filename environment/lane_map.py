"""
environment/lane_map.py
=======================
LaneMap: encodes Indian left-hand road geometry.

Provides:
  - lane_cost(x, y)       : penalty cost for being at (x, y) relative to road
  - lane_cost_batch(...)  : vectorised version for MPPI trajectory scoring
  - in_left_lane(x, y)    : True if y is in left lane band
  - in_right_lane(x, y)   : True if y is in right lane band
  - boundary_cost(x, y)   : lethal if outside road boundaries
  - get_target_lane_y(mode): returns target y for given maneuver mode
  - wrong_lane_violation(x, y, mode) : True if in wrong lane for current mode

Road geometry (from config):
  Road spans y in [ROAD_Y_MIN, ROAD_Y_MAX] = [5, 15]
  Centerline at y = ROAD_CENTERLINE_Y = 10
  Left lane  (default, keep-left rule): y in (5, 10],  center 8.5
  Right lane (temporary overtake only): y in (10, 15), center 12.5
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ROAD_Y_MIN, ROAD_Y_MAX, ROAD_CENTERLINE_Y,
    LANE_WIDTH, LEFT_LANE_CENTER_Y, RIGHT_LANE_CENTER_Y,
    LANE_COST_LEFT, LANE_COST_RIGHT, LANE_COST_BOUNDARY,
)

# Lateral slope of the linear penalty away from lane centre. Kept gentle so
# that the goal critic can pull the buggy to legacy scenario goals sitting on
# the centerline (y=10); a steep slope (2.0) made the buggy orbit the goal at
# the lane centre instead of entering the goal tolerance ring.
_LANE_EDGE_SLOPE = 0.5


class LaneMap:

    def lane_cost(self, x: float, y: float) -> float:
        """
        Returns lane discipline cost at position (x, y).
        Used by LaneCritic inside MPPI.
        """
        # Outside road entirely → lethal
        if y <= ROAD_Y_MIN or y >= ROAD_Y_MAX:
            return LANE_COST_BOUNDARY

        # Left lane (correct lane for Indian traffic)
        if ROAD_Y_MIN < y <= ROAD_CENTERLINE_Y:
            # Penalty increasing toward road edge
            dist_from_center = abs(y - LEFT_LANE_CENTER_Y)
            return LANE_COST_LEFT + _LANE_EDGE_SLOPE * dist_from_center

        # Right lane (wrong lane — soft penalty)
        dist_from_center = abs(y - RIGHT_LANE_CENTER_Y)
        return LANE_COST_RIGHT + _LANE_EDGE_SLOPE * dist_from_center

    def lane_cost_batch(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorised lane cost for arrays of positions (any equal shape).
        Same values as lane_cost() elementwise.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        left_cost = LANE_COST_LEFT + _LANE_EDGE_SLOPE * np.abs(ys - LEFT_LANE_CENTER_Y)
        right_cost = LANE_COST_RIGHT + _LANE_EDGE_SLOPE * np.abs(ys - RIGHT_LANE_CENTER_Y)

        out_of_road = (ys <= ROAD_Y_MIN) | (ys >= ROAD_Y_MAX)
        in_left = (~out_of_road) & (ys <= ROAD_CENTERLINE_Y)

        return np.where(out_of_road, LANE_COST_BOUNDARY,
               np.where(in_left, left_cost, right_cost))

    def boundary_cost(self, x: float, y: float) -> float:
        """Lethal if y exits road limits."""
        if y <= ROAD_Y_MIN or y >= ROAD_Y_MAX:
            return LANE_COST_BOUNDARY
        return 0.0

    def boundary_cost_batch(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised boundary cost (lethal outside road, zero inside)."""
        ys = np.asarray(ys, dtype=float)
        out_of_road = (ys <= ROAD_Y_MIN) | (ys >= ROAD_Y_MAX)
        return np.where(out_of_road, LANE_COST_BOUNDARY, 0.0)

    def get_target_lane_y(self, maneuver_mode: str = "normal") -> float:
        """
        Returns the y-coordinate MPPI path_critic should target.
        Changes during maneuvers:
          "normal"       → LEFT_LANE_CENTER_Y (default)
          "overtake"     → RIGHT_LANE_CENTER_Y (temporary)
          "return_left"  → LEFT_LANE_CENTER_Y
          "right_turn"   → RIGHT_LANE_CENTER_Y (pre-turn positioning)
          "left_turn"    → LEFT_LANE_CENTER_Y
          "u_turn"       → RIGHT_LANE_CENTER_Y (widest arc position)
          "reverse"      → LEFT_LANE_CENTER_Y
        """
        mode_map = {
            "normal":       LEFT_LANE_CENTER_Y,
            "overtake":     RIGHT_LANE_CENTER_Y,
            "return_left":  LEFT_LANE_CENTER_Y,
            "right_turn":   RIGHT_LANE_CENTER_Y,
            "left_turn":    LEFT_LANE_CENTER_Y,
            "u_turn":       RIGHT_LANE_CENTER_Y,
            "reverse":      LEFT_LANE_CENTER_Y,
        }
        return mode_map.get(maneuver_mode, LEFT_LANE_CENTER_Y)

    def in_left_lane(self, x: float, y: float) -> bool:
        return ROAD_Y_MIN < y <= ROAD_CENTERLINE_Y

    def in_right_lane(self, x: float, y: float) -> bool:
        return ROAD_CENTERLINE_Y < y < ROAD_Y_MAX

    def wrong_lane_violation(self, x: float, y: float, maneuver_mode: str = "normal") -> bool:
        """
        Returns True if buggy is in wrong lane for current maneuver.
        Used by metrics recorder to count violations.
        """
        if maneuver_mode in ("normal", "left_turn", "return_left", "reverse"):
            return self.in_right_lane(x, y)
        return False  # overtake/right_turn/u_turn allows right lane temporarily


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    lm = LaneMap()

    # Left lane centre is free
    assert lm.lane_cost(5.0, LEFT_LANE_CENTER_Y) == LANE_COST_LEFT
    # Right lane centre carries the soft penalty
    assert lm.lane_cost(5.0, RIGHT_LANE_CENTER_Y) == LANE_COST_RIGHT
    # Outside road is lethal
    assert lm.lane_cost(5.0, ROAD_Y_MIN - 0.1) == LANE_COST_BOUNDARY
    assert lm.lane_cost(5.0, ROAD_Y_MAX + 0.1) == LANE_COST_BOUNDARY
    # Lane membership
    assert lm.in_left_lane(0.0, 8.5) and not lm.in_right_lane(0.0, 8.5)
    assert lm.in_right_lane(0.0, 12.5) and not lm.in_left_lane(0.0, 12.5)
    # Target y per mode
    assert lm.get_target_lane_y("normal") == LEFT_LANE_CENTER_Y
    assert lm.get_target_lane_y("u_turn") == RIGHT_LANE_CENTER_Y
    # Wrong-lane detection honours maneuver mode
    assert lm.wrong_lane_violation(0.0, 12.5, "normal") is True
    assert lm.wrong_lane_violation(0.0, 12.5, "overtake") is False
    # Batch matches scalar
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([8.5, 12.5, 4.9, 15.1])
    batch = lm.lane_cost_batch(xs, ys)
    scalar = np.array([lm.lane_cost(x, y) for x, y in zip(xs, ys)])
    assert np.allclose(batch, scalar), f"{batch} != {scalar}"

    print("[PASS] LaneMap verified!")

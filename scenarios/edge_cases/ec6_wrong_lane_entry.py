"""
scenarios/edge_cases/ec6_wrong_lane_entry.py
=============================================
Edge Case 6 — Wrong Lane Entry.

Obstacle blocks the LEFT lane. The buggy must temporarily use the RIGHT
lane (overtake), then return to the left lane after passing the obstacle.

Indian left-hand traffic: right lane is for temporary overtaking only.

Metrics (via recorder / LaneMap):
  - wrong-lane timestep count before and after the overtake
  - time to return to the left lane after clearing the obstacle
Pass condition: returns to the left lane within ~5 m of clearing the obstacle.
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
from config import LEFT_LANE_CENTER_Y
import config


class EC6WrongLaneEntry(ScenarioBase):
    """
    EC6: Box squarely on the left lane centre; goal continues down the
    left lane, forcing a right-lane overtake and return.
    """

    # Distance (m of travel) within which the buggy should be back in the
    # left lane after passing the obstacle
    RETURN_WINDOW_M = 5.0

    def __init__(self):
        super().__init__()
        self.name = "ec6_wrong_lane_entry"

        self.start = (1.0, LEFT_LANE_CENTER_Y, 0.0)
        self.goal  = (19.0, LEFT_LANE_CENTER_Y)
        self.global_path = straight_path(1.0, LEFT_LANE_CENTER_Y,
                                         19.0, LEFT_LANE_CENTER_Y, n_pts=40)

        self.obstacles = [
            Obstacle(x=10.0, y=LEFT_LANE_CENTER_Y, width=1.5, height=1.5,
                     label="left_lane_box"),
        ]
        self.static_obstacles = self.obstacles

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC6WrongLaneEntry self-test ===")
    scen = EC6WrongLaneEntry()
    scen.reset(seed=42)
    assert len(scen.obstacles) == 1
    assert scen.obstacles[0].y == LEFT_LANE_CENTER_Y
    assert scen.costmap is not None
    assert scen.get_maneuver_command(0.0) is None
    print(f"Start: {scen.start}, Goal: {scen.goal}")
    print("[PASS] EC6WrongLaneEntry verified.")

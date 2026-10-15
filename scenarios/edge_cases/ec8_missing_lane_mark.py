"""
scenarios/edge_cases/ec8_missing_lane_mark.py
==============================================
Edge Case 8 — Missing Lane Markings.

Lane guidance is disabled for this scenario (disable_lane_cost = True):
the lane critic contributes nothing and the buggy must navigate using
obstacle + path + goal critics only.

Expected: the buggy may drift laterally without the lane pull; the run
tracks mean lateral deviation from the left-lane centre.

Metric: mean lateral deviation from LEFT_LANE_CENTER_Y (recorder).
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
from config import LEFT_LANE_CENTER_Y


class EC8MissingLaneMark(ScenarioBase):
    """
    EC8: One box slightly right of the left-lane centre so the buggy must
    make a lateral decision without any lane-cost guidance.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec8_missing_lane_mark"

        self.start = (1.0, LEFT_LANE_CENTER_Y, 0.0)
        self.goal  = (19.0, LEFT_LANE_CENTER_Y)
        self.global_path = straight_path(1.0, LEFT_LANE_CENTER_Y,
                                         19.0, LEFT_LANE_CENTER_Y, n_pts=40)

        self.obstacles = [
            Obstacle(x=10.0, y=9.6, width=1.5, height=1.5, label="unmarked_box"),
        ]
        self.static_obstacles = self.obstacles

        # EC8 hook: lane critic disabled for this scenario
        self.disable_lane_cost = True

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC8MissingLaneMark self-test ===")
    scen = EC8MissingLaneMark()
    scen.reset(seed=42)
    assert scen.disable_lane_cost is True
    assert len(scen.obstacles) == 1
    assert scen.costmap is not None
    print(f"Start: {scen.start}, Goal: {scen.goal}, lane guidance: DISABLED")
    print("[PASS] EC8MissingLaneMark verified.")

"""
scenarios/edge_cases/ec1_dead_end.py
====================================
Edge Case 1 — Dead End (U-Shaped Trap).
Three walls form a cul-de-sac blocking all forward progress toward goal.
Tests whether controller detects the trap and triggers recovery or gets stuck.
"""

from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class EC1DeadEnd(ScenarioBase):
    """
    EC1: U-shaped dead end blocking forward path.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec1_dead_end"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        # U-shaped barrier walls at x=12..15
        self.obstacles = [
            Obstacle(x=12.0, y=8.0,  width=0.3, height=4.0, label="wall_left"),
            Obstacle(x=15.0, y=8.0,  width=0.3, height=4.0, label="wall_right"),
            Obstacle(x=13.5, y=10.0, width=3.3, height=0.3, label="wall_top"),
        ]
        self.static_obstacles = self.obstacles

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC1DeadEnd self-test ===")
    scen = EC1DeadEnd()
    scen.reset()
    assert len(scen.obstacles) == 3
    print("[PASS] EC1DeadEnd verified.")

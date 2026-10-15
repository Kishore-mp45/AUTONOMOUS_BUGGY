"""
scenarios/edge_cases/ec4_late_braking.py
=========================================
Edge Case 4 — Late Braking.
Obstacle appears close ahead while buggy is traveling at high speed.
Tests whether controller initiates emergency deceleration within one cycle.
"""

from typing import List, Tuple, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class EC4LateBraking(ScenarioBase):
    """
    EC4: Clear path until t=1.0s, when obstacle suddenly appears 4m ahead.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec4_late_braking"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        self.obstacle = Obstacle(x=6.0, y=10.0, width=1.0, height=1.0, label="brake_hazard")
        self.obstacle.active = False

        self.obstacles = [self.obstacle]
        self.static_obstacles = []

    def _on_reset(self, seed: Optional[int] = None):
        self.obstacle = Obstacle(x=6.0, y=10.0, width=1.0, height=1.0, label="brake_hazard")
        self.obstacle.active = False
        self.obstacles = [self.obstacle]
        self.static_obstacles = []
        self.costmap = self.build_costmap()

    def update(self, t: float, robot_state: Optional[np.ndarray] = None) -> None:
        """Trigger pop-up at t >= 1.0s."""
        if t >= 1.0 and not self.obstacle.active:
            self.obstacle.active = True
            if robot_state is not None:
                self.obstacle.x = float(robot_state[0]) + 4.0
            if self.costmap is not None:
                self.costmap.update(self.obstacles)
        elif self.obstacle.active and self.costmap is not None:
            self.costmap.update(self.obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC4LateBraking self-test ===")
    scen = EC4LateBraking()
    scen.reset()
    assert not scen.obstacle.active
    scen.update(t=1.05, robot_state=np.array([2.0, 10.0, 0.0, 1.5]))
    assert scen.obstacle.active
    print(f"Spawned at x={scen.obstacle.x} (expected 2.0 + 4.0 = 6.0)")
    assert abs(scen.obstacle.x - 6.0) < 1e-3
    print("[PASS] EC4LateBraking verified.")

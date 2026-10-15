"""
scenarios/s4_two_obstacles.py
==============================
Scenario 4 — Two Staggered Obstacles.
Upgraded with Obstacle objects, second obstacle slow drift (0.1 m/s),
and Costmap2D integration.
"""

from typing import List, Tuple, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class S4TwoObstacles(ScenarioBase):
    """
    Scenario 4: Two staggered obstacles creating a slalom course.
    Obstacle 2 drifts slowly sideways (vy = 0.1 m/s) to test dynamic slalom avoidance.
    """

    def __init__(self):
        super().__init__()
        self.name = "s4_two_obstacles"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        self.box1 = Obstacle(x=9.0, y=10.5, width=1.2, height=1.2, label="box1")
        self.box2 = Obstacle(x=11.0, y=9.2, width=1.2, height=1.2, dynamic=True, vy=0.1, label="box2_drift")

        self.obstacles = [self.box1, self.box2]
        self.static_obstacles = [self.box1]

    def _on_reset(self, seed: Optional[int] = None):
        """Reset obstacles and re-build costmap."""
        self.box1 = Obstacle(x=9.0, y=10.5, width=1.2, height=1.2, label="box1")
        self.box2 = Obstacle(x=11.0, y=9.2, width=1.2, height=1.2, dynamic=True, vy=0.1, label="box2_drift")
        self.obstacles = [self.box1, self.box2]
        self.static_obstacles = [self.box1]
        self.costmap = self.build_costmap()

    def update(self, t: float, robot_state: Optional[np.ndarray] = None) -> None:
        """Advance drifting box and update costmap."""
        self.box2.step(config.DT)
        if self.costmap is not None:
            self.costmap.update(self.obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== S4TwoObstacles self-test ===")
    scen = S4TwoObstacles()
    scen.reset(seed=42)
    print(f"Obstacles: {len(scen.obstacles)}, Box2 initial y: {scen.box2.y}")
    for _ in range(10):
        scen.update(0.5)
    print(f"Box2 after drifting y: {scen.box2.y:.3f}")
    assert scen.box2.y > 9.2
    print("[PASS] S4TwoObstacles verified.")

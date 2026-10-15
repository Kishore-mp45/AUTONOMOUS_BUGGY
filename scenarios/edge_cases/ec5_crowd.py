"""
scenarios/edge_cases/ec5_crowd.py
==================================
Edge Case 5 — Dynamic Moving Crowd.
Three pedestrians cross the buggy's path at staggered intervals, speeds, and directions:
  - Person 1: crosses at x=8.0,  y: 5 -> 15,  speed=0.8 m/s, starts at t=3.0s
  - Person 2: crosses at x=12.0, y: 15 -> 5,  speed=1.2 m/s, starts at t=6.0s
  - Person 3: crosses at x=15.0, y: 5 -> 15,  speed=0.6 m/s, starts at t=9.0s
Tests multi-agent prediction, hesitation times, and avoidance smoothness.
"""

from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class EC5Crowd(ScenarioBase):
    """
    EC5: Dynamic pedestrian crowd with 3 moving agents crossing at staggered times.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec5_crowd"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        self._init_agents()

    def _init_agents(self):
        self.p1 = Obstacle(x=8.0,  y=5.0,  radius=0.35, dynamic=True, vx=0.0, vy=0.8, label="p1")
        self.p2 = Obstacle(x=12.0, y=15.0, radius=0.35, dynamic=True, vx=0.0, vy=-1.2, label="p2")
        self.p3 = Obstacle(x=15.0, y=5.0,  radius=0.35, dynamic=True, vx=0.0, vy=0.6, label="p3")

        self.p1.active = False
        self.p2.active = False
        self.p3.active = False

        self.obstacles = [self.p1, self.p2, self.p3]
        self.static_obstacles = []

    def _on_reset(self, seed: Optional[int] = None):
        self._init_agents()
        self.costmap = self.build_costmap()

    def update(self, t: float, robot_state: Optional[object] = None) -> None:
        """Activate pedestrians according to schedule and advance positions."""
        dt = config.DT
        if t >= 3.0 and not self.p1.active:
            self.p1.active = True
        if t >= 6.0 and not self.p2.active:
            self.p2.active = True
        if t >= 9.0 and not self.p3.active:
            self.p3.active = True

        for p in self.obstacles:
            if p.active:
                p.step(dt)

        if self.costmap is not None:
            self.costmap.update(self.obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC5Crowd self-test ===")
    scen = EC5Crowd()
    scen.reset()
    assert len(scen.obstacles) == 3
    scen.update(t=4.0)
    assert scen.p1.active and not scen.p2.active and not scen.p3.active
    scen.update(t=7.0)
    assert scen.p1.active and scen.p2.active and not scen.p3.active
    scen.update(t=10.0)
    assert scen.p1.active and scen.p2.active and scen.p3.active
    print("[PASS] EC5Crowd verified.")

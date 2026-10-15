"""
scenarios/s2_person_crossing.py
================================
Scenario 2 — Moving Person Crossing.
Upgraded with Obstacle class, dynamic costmap updating,
frozen pedestrian edge case (every 3rd run), and unpredictable reversal.
"""

from typing import List, Tuple, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config

_PERSON_START_X = 10.0
_PERSON_START_Y = 5.0
_PERSON_RADIUS  = 0.4


class S2PersonCrossing(ScenarioBase):
    """
    Scenario 2: Moving pedestrian with frozen and direction reversal edge cases.
    """

    def __init__(self):
        super().__init__()
        self.name = "s2_person_crossing"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        self.person = Obstacle(
            x=_PERSON_START_X,
            y=_PERSON_START_Y,
            radius=_PERSON_RADIUS,
            dynamic=True,
            vx=0.0,
            vy=0.5,
            label="person"
        )
        self.obstacles = [self.person]
        self.static_obstacles = []
        self.frozen = False

    def _on_reset(self, seed: Optional[int] = None):
        """Reset person and apply edge case variants."""
        # Frozen person mid-path if seed % 3 == 0
        self.frozen = (seed is not None and seed % 3 == 0)
        self.person = Obstacle(
            x=_PERSON_START_X,
            y=_PERSON_START_Y,
            radius=_PERSON_RADIUS,
            dynamic=True,
            vx=0.0,
            vy=0.5,
            label="person"
        )
        self.obstacles = [self.person]
        self.static_obstacles = []
        self.costmap = self.build_costmap()

    def update(self, t: float, robot_state: Optional[np.ndarray] = None) -> None:
        """
        Update pedestrian position and refresh costmap.
        """
        if self.frozen and self.person.y >= 10.0:
            self.person.vy = 0.0  # freeze mid-crossing
        elif t > 12.0 and not self.frozen:
            self.person.vy = -0.3  # reverse direction at t=12s

        self.person.step(config.DT)
        if self.costmap is not None:
            self.costmap.update(self.obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== S2PersonCrossing self-test ===")
    scen = S2PersonCrossing()
    scen.reset(seed=42)
    print(f"Person initial y: {scen.person.y}")
    for step in range(20):
        scen.update(t=step * config.DT)
    print(f"Person after 20 steps y: {scen.person.y:.2f}")
    assert scen.person.y > _PERSON_START_Y
    print("[PASS] S2PersonCrossing verified.")

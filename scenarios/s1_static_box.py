"""
scenarios/s1_static_box.py
===========================
Scenario 1 — Static Box.
Upgraded with Obstacle class, build_costmap(), and seed support.
"""

from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class S1StaticBox(ScenarioBase):
    """
    Scenario 1: Single static rectangular obstacle placed on the straight path.
    """

    def __init__(self):
        super().__init__()
        self.name = "s1_static_box"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        # Standard obstacle
        self.obstacles = [
            Obstacle(x=10.0, y=10.0, width=1.5, height=1.5, label="box")
        ]
        self.static_obstacles = self.obstacles

    def _on_reset(self, seed: Optional[int] = None):
        """Re-create obstacles and costmap on reset."""
        # Optional off-center variation if desired
        y_offset = 0.0
        if seed is not None and seed % 5 == 0:
            y_offset = 0.2  # slight offset for robustness test

        self.obstacles = [
            Obstacle(x=10.0, y=10.0 + y_offset, width=1.5, height=1.5, label="box")
        ]
        self.static_obstacles = self.obstacles
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== S1StaticBox self-test ===")
    scen = S1StaticBox()
    scen.reset(seed=42)
    print(f"Start: {scen.start}, Goal: {scen.goal}, Obstacles: {len(scen.obstacles)}")
    assert len(scen.obstacles) == 1
    assert scen.costmap is not None
    print("[PASS] S1StaticBox verified.")

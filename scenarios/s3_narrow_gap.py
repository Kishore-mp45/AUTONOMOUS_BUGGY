"""
scenarios/s3_narrow_gap.py
===========================
Scenario 3 — Narrow Corridor / Gap.
Upgraded with Obstacle objects, dynamic gap width variants (3.0m, 2.0m, 1.8m),
lateral centerline deviation tracking, and Costmap2D integration.
"""

from typing import List, Tuple, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class S3NarrowGap(ScenarioBase):
    """
    Scenario 3: Narrow corridor gap between two vertical wall segments.
    Gap width varies by run seed (3.0m easy, 2.0m medium, 1.8m hard).
    """

    def __init__(self):
        super().__init__()
        self.name = "s3_narrow_gap"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)
        self.gap_width = 3.0

        self._build_walls(gap=3.0)

    def _build_walls(self, gap: float):
        """Construct upper and lower vertical wall Obstacle objects."""
        self.gap_width = float(gap)
        half_gap = self.gap_width / 2.0

        # Map is 20m high (y from 0 to 20). Center is y=10.
        # Lower wall: from y=0 to y=(10 - half_gap)
        h_bot = max(0.1, 10.0 - half_gap)
        y_bot = h_bot / 2.0

        # Upper wall: from y=(10 + half_gap) to y=20
        h_top = max(0.1, 20.0 - (10.0 + half_gap))
        y_top = (10.0 + half_gap) + h_top / 2.0

        wall_thick = 0.3
        self.obstacles = [
            Obstacle(x=10.0, y=y_top, width=wall_thick, height=h_top, label="wall_top"),
            Obstacle(x=10.0, y=y_bot, width=wall_thick, height=h_bot, label="wall_bot"),
        ]
        self.static_obstacles = self.obstacles

    def _on_reset(self, seed: Optional[int] = None):
        """Configure gap width depending on seed / run_id."""
        if seed is not None and seed % 10 < 4:
            gap = 3.0  # easy
        elif seed is not None and seed % 10 < 7:
            gap = 2.0  # medium
        else:
            gap = 1.8  # hard (realistic tight passage)

        self._build_walls(gap=gap)
        self.costmap = self.build_costmap()

    def get_lateral_deviation(self, state: np.ndarray) -> float:
        """Measure lateral deviation |y - 10.0| from nominal centerline."""
        return abs(float(state[1]) - 10.0)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== S3NarrowGap self-test ===")
    scen = S3NarrowGap()
    scen.reset(seed=0)
    print(f"Seed 0: Gap width = {scen.gap_width}m, obstacles = {len(scen.obstacles)}")
    assert scen.gap_width == 3.0
    scen.reset(seed=5)
    print(f"Seed 5: Gap width = {scen.gap_width}m")
    assert scen.gap_width == 2.0
    scen.reset(seed=9)
    print(f"Seed 9: Gap width = {scen.gap_width}m")
    assert scen.gap_width == 1.8
    print("[PASS] S3NarrowGap verified.")

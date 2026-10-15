"""
scenarios/edge_cases/ec3_tight_turn.py
======================================
Edge Case 3 — Tight 90-Degree Turn.
Sharp left turn at x=10 with tight inner and outer boundary walls.
Tests Ackermann minimum turning radius compliance without clipping inner corner walls.
"""

from typing import List, Tuple, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase
from environment.obstacle import Obstacle
import config


class EC3TightTurn(ScenarioBase):
    """
    EC3: 90-degree left turn at (10, 2) heading upward to (10, 18).
    """

    def __init__(self):
        super().__init__()
        self.name = "ec3_tight_turn"

        self.start = (1.0, 2.0, 0.0)
        self.goal  = (10.0, 18.0)

        # L-shaped reference path
        pts_leg1 = [(float(x), 2.0) for x in np.linspace(1.0, 10.0, 20)]
        pts_leg2 = [(10.0, float(y)) for y in np.linspace(2.0, 18.0, 20)]
        self.global_path = pts_leg1 + pts_leg2[1:]

        # Inner corner block at (9.2, 2.8) and outer wall
        self.obstacles = [
            Obstacle(x=8.5, y=3.5, width=1.5, height=1.5, label="inner_corner"),
            Obstacle(x=11.5, y=1.0, width=0.3, height=4.0, label="outer_wall"),
        ]
        self.static_obstacles = self.obstacles

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC3TightTurn self-test ===")
    scen = EC3TightTurn()
    scen.reset()
    assert len(scen.global_path) == 39
    print("[PASS] EC3TightTurn verified.")

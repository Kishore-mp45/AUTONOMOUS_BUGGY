"""
scenarios/edge_cases/ec10_all_samples_collide.py
=================================================
Edge Case 10 — All Samples Collide.

A box on the path plus two side boxes create a partial cul-de-sac ahead of
the buggy. At full speed every forward MPPI sample is lethal, triggering
the all-collide fallback: sigma expansion → recovery attempts → escape
through one of the lateral corridors (or emergency stop).

Layout (top view, road y in [5, 15]):
                     ┌────┐
        left box     │    │   y=8.5   ← left corridor below (road edge y=5)
                     └────┘
                     ┌────┐
       center box    │    │   y=10.75..9.25 (on the direct path)
                     └────┘
                     ┌────┐
        right box    │    │   y=13.25  ← right corridor above (road edge y=15)
                     └────┘

Expected: MPPI detects all-collide, widens sigma, finds a corridor after
N recovery attempts (metrics: recovery_count, escape time, outcome).
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config


class EC10AllSamplesCollide(ScenarioBase):
    """
    EC10: Center box on path + flank boxes near the left/right lane centres.
    Lateral escape corridors remain open toward the road edges.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec10_all_samples_collide"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        self.obstacles = [
            Obstacle(x=9.0, y=10.0,  width=1.5, height=1.5, label="center_box"),
            Obstacle(x=8.0, y=8.25,  width=1.5, height=1.5, label="left_flank"),
            Obstacle(x=8.0, y=11.75, width=1.5, height=1.5, label="right_flank"),
        ]
        self.static_obstacles = self.obstacles

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC10AllSamplesCollide self-test ===")
    scen = EC10AllSamplesCollide()
    scen.reset(seed=42)
    assert len(scen.obstacles) == 3
    assert scen.costmap is not None
    # Direct path blocked lethally
    assert scen.costmap.query(9.0, 10.0) >= config.LETHAL_COST
    # Corridors open (near road edges)
    assert scen.costmap.query(9.0, 6.0) < config.INSCRIBED_COST
    assert scen.costmap.query(9.0, 14.0) < config.INSCRIBED_COST
    print("Center blocked; lateral corridors open.")
    print("[PASS] EC10AllSamplesCollide verified.")

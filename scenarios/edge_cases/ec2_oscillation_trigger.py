"""
scenarios/edge_cases/ec2_oscillation_trigger.py
================================================
Edge Case 2 — Oscillation Trigger.
Long unobstructed straight path traversed at high speed.
Tests if controllers (especially RPP without smoothing) exhibit steering hunting/oscillation.
"""

from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
import config


class EC2OscillationTrigger(ScenarioBase):
    """
    EC2: Completely open straight path at maximum speed.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec2_oscillation_trigger"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        # Clear environment (no obstacles)
        self.obstacles = []
        self.static_obstacles = []

    def _on_reset(self, seed: Optional[int] = None):
        self.obstacles = []
        self.static_obstacles = []
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC2OscillationTrigger self-test ===")
    scen = EC2OscillationTrigger()
    scen.reset()
    assert len(scen.obstacles) == 0
    print("[PASS] EC2OscillationTrigger verified.")

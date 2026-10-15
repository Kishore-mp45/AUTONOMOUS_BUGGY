"""
scenarios/edge_cases/ec7_u_turn_blocked.py
===========================================
Edge Case 7 — U-Turn Blocked Road.

The road ends in a wall. The buggy must detect the blockage (scenario
commands "u_turn"), execute a 180-degree arc (minimum turning radius
enforced), and return to its source node in the left lane.

Metrics:
  - heading change achieved (target 170–190 degrees)
  - min distance to road boundary during the arc
Pass condition: heading rotates ~180 degrees, no boundary collision.
"""

from typing import Optional
import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
from config import LEFT_LANE_CENTER_Y
import config


class EC7UTurnBlocked(ScenarioBase):
    """
    EC7: Full-width wall at x=13; the buggy returns to its source at (2, 8.5).
    """

    # Robot must be within this distance of the wall before the scenario
    # issues the u_turn command (no premature turns)
    TURN_TRIGGER_X = 10.5

    def __init__(self):
        super().__init__()
        self.name = "ec7_u_turn_blocked"

        self.start = (2.0, LEFT_LANE_CENTER_Y, 0.0)
        # The controller's target is beyond the source; termination is at the source.
        self.goal  = (1.0, LEFT_LANE_CENTER_Y)
        # Reference path leads toward the wall (the trap)
        self.global_path = straight_path(2.0, LEFT_LANE_CENTER_Y,
                                         12.0, LEFT_LANE_CENTER_Y, n_pts=20)

        self.obstacles = [
            Obstacle(x=13.0, y=10.0, width=0.5, height=10.0, label="road_end_wall"),
        ]
        self.static_obstacles = self.obstacles
        self._uturn_commanded = False
        self._return_path_active = False

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self._uturn_commanded = False
        self._return_path_active = False
        self.global_path = straight_path(
            self.start[0],
            self.start[1],
            12.0,
            LEFT_LANE_CENTER_Y,
            n_pts=20,
        )
        self.costmap = self.build_costmap()

    def get_maneuver_command(self, t: float, state: Optional[np.ndarray] = None) -> Optional[str]:
        """Command a U-turn once the robot is close to the blocking wall."""
        if state is not None and float(state[0]) >= self.TURN_TRIGGER_X:
            self._uturn_commanded = True
        if (
            self._uturn_commanded
            and not self._return_path_active
            and state is not None
            and math.cos(float(state[2])) < -0.9
        ):
            self.global_path = straight_path(
                float(state[0]),
                float(state[1]),
                self.start[0],
                self.start[1],
                n_pts=40,
            )
            self._return_path_active = True
        return "u_turn" if self._uturn_commanded else None

    def is_complete(
        self,
        state: np.ndarray,
        t: float,
        goal: Optional[tuple[float, float]] = None,
    ) -> tuple[bool, bool, str]:
        """Finish successfully at the source after the commanded U-turn."""
        if self._return_path_active:
            distance_to_source = math.hypot(
                float(state[0]) - self.start[0],
                float(state[1]) - self.start[1],
            )
            if distance_to_source < config.GOAL_TOLERANCE:
                return True, True, ""
        return super().is_complete(state, t, goal)

    def uturn_commanded(self) -> bool:
        return self._uturn_commanded


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC7UTurnBlocked self-test ===")
    scen = EC7UTurnBlocked()
    scen.reset(seed=42)
    assert len(scen.obstacles) == 1
    # No command before the trigger
    assert scen.get_maneuver_command(0.0, np.array([5.0, 8.5, 0.0, 1.0])) is None
    # Command after passing trigger x
    assert scen.get_maneuver_command(1.0, np.array([11.0, 8.5, 0.0, 1.0])) == "u_turn"
    # Latched — stays commanded even if robot slips back
    assert scen.get_maneuver_command(2.0, np.array([9.0, 8.5, 0.0, 1.0])) == "u_turn"
    print(f"Start: {scen.start}, Goal: {scen.goal}")
    print("[PASS] EC7UTurnBlocked verified.")

"""
scenarios/s1_static_box.py
===========================
Scenario 1 — Static Box

Map layout (20 x 20 m):
    Start : (1, 10)   heading East (theta = 0)
    Goal  : (19, 10)
    Path  : straight horizontal line at y = 10

    Obstacle: solid box centred at (10, 10), size 1.5 x 1.5 m
              Placed directly on the nominal path.

Challenge: Robot must detect the obstacle and steer around it
           (above or below) then return to the path toward the goal.
"""

import copy
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
import config


class S1StaticBox(ScenarioBase):
    """
    Scenario 1: Single static rectangular obstacle placed on the straight path.
    All controllers must detect and navigate around it.
    """

    def __init__(self):
        super().__init__()
        self.name = "s1_static_box"

        # ── Poses ─────────────────────────────────────────────────────
        self.start = (1.0, 10.0, 0.0)       # (x, y, theta)
        self.goal  = (19.0, 10.0)

        # ── Reference path ────────────────────────────────────────────
        # Nominal straight line — obstacle sits right on it.
        # Controllers must deviate from this path to avoid the box.
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        # ── Obstacles ─────────────────────────────────────────────────
        _obstacles = [
            {
                "type": "static",
                "x": 10.0, "y": 10.0,
                "w": 1.5,  "h": 1.5,
                "label": "box",
            }
        ]
        self._initial_obstacles = copy.deepcopy(_obstacles)
        self.obstacles          = copy.deepcopy(_obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = S1StaticBox()
    print(s)
    print(f"  Path length: {len(s.global_path)} waypoints")
    print(f"  Obstacles:   {s.obstacles}")

    import numpy as np
    # Start state
    state = s.get_start_state()
    print(f"  Start state: {state}")
    assert not s.goal_reached(state), "Start should not be at goal"
    assert not s.collision_detected(state), "Start should be obstacle-free"

    # At goal
    goal_state = np.array([19.0, 10.0, 0.0, 0.0])
    assert s.goal_reached(goal_state), "Should detect goal reached"

    # Reset preserves obstacle
    s.reset()
    assert len(s.obstacles) == 1
    assert s.obstacles[0]["x"] == 10.0

    print("[PASS]  S1StaticBox OK")

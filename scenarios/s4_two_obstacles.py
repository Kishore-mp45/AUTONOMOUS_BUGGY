"""
scenarios/s4_two_obstacles.py
==============================
Scenario 4 — Two Staggered Obstacles

Map layout (20 x 20 m):
    Start : (1, 10)   heading East
    Goal  : (19, 10)
    Path  : straight horizontal line at y = 10

    Box 1: centred at (9,  10.5), size 1.2 x 1.2 m
    Box 2: centred at (11,  9.2), size 1.2 x 1.2 m

    Staggered placement — the two boxes overlap in x and are
    offset in y, creating a slalom course.  The robot must weave:
        left of Box 1 (or above) → right of Box 2 (or below).

    Combined corridor between the boxes is:
        Box 1 lower edge: 10.5 - 0.6 = 9.9
        Box 2 upper edge:  9.2 + 0.6 = 9.8
        → Effectively the two boxes nearly close the corridor,
          making the robot commit to one side early.

Challenge: Planning is needed across both boxes simultaneously.
           Greedy local methods may oscillate between the two.
"""

import copy
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
import config


class S4TwoObstacles(ScenarioBase):
    """
    Scenario 4: Two staggered rectangular obstacles require a slalom manoeuvre.
    """

    def __init__(self):
        super().__init__()
        self.name = "s4_two_obstacles"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        _obstacles = [
            {
                "type":  "static",
                "x":     9.0,   "y": 10.5,
                "w":     1.2,   "h":  1.2,
                "label": "box1",
            },
            {
                "type":  "static",
                "x":    11.0,   "y":  9.2,
                "w":     1.2,   "h":  1.2,
                "label": "box2",
            },
        ]
        self._initial_obstacles = copy.deepcopy(_obstacles)
        self.obstacles          = copy.deepcopy(_obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = S4TwoObstacles()
    print(s)
    for obs in s.obstacles:
        print(f"  {obs['label']}  centre=({obs['x']},{obs['y']})  "
              f"w={obs['w']} h={obs['h']}")

    import numpy as np
    # Verify the path centre (y=10) is indeed partially blocked
    # Box 1: x in [8.4, 9.6],  y in [9.9, 11.1]
    # Box 2: x in [10.4, 11.6], y in [8.6,  9.8]
    # At x=9, y=10 → inside box1 footprint?
    state_inside = np.array([9.0, 10.5, 0.0, 1.0])
    assert s.collision_detected(state_inside), "Robot at box1 centre should collide"

    # Robot comfortably passing below box1 (y=9.0)
    state_below = np.array([9.0, 8.8, 0.0, 1.0])
    d = s.min_obstacle_dist(state_below)
    print(f"  At (9, 8.8) below box1: dist={d:.3f} m")
    assert d > config.ROBOT_RADIUS, "Should be clear below box1"

    s.reset()
    assert len(s.obstacles) == 2
    assert s.obstacles[0]["x"] == 9.0
    assert s.obstacles[1]["x"] == 11.0
    print("[PASS]  S4TwoObstacles OK")

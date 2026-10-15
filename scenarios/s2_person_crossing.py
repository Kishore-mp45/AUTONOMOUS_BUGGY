"""
scenarios/s2_person_crossing.py
================================
Scenario 2 — Moving Person Crossing

Map layout (20 x 20 m):
    Start : (1, 10)   heading East
    Goal  : (19, 10)
    Path  : straight horizontal line at y = 10

    Person: circle of radius 0.4 m
            starts at (10, 5)
            moves upward:  vy = +0.5 m/s  (vx = 0)
            crosses y = 10 at t ~= (10 - 5) / 0.5 = 10 s

Challenge: Robot must anticipate or react to the person crossing its path.
           Tests dynamic-obstacle awareness and speed regulation.

Edge cases handled:
  - Person continues past the map top edge (obstacle still tracked,
    distance just becomes very large — no crash risk).
  - Reset restores person to start_y.
"""

import copy
import numpy as np
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
import config


_PERSON_START_X  = 10.0
_PERSON_START_Y  =  5.0
_PERSON_VX       =  0.0
_PERSON_VY       =  0.5   # m/s — crosses path at t ≈ 10 s
_PERSON_RADIUS   =  0.4   # m


class S2PersonCrossing(ScenarioBase):
    """
    Scenario 2: A pedestrian crosses the robot's path perpendicularly.
    The person starts below the path and moves upward at 0.5 m/s.
    """

    def __init__(self):
        super().__init__()
        self.name = "s2_person_crossing"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        _obstacles = [
            {
                "type":    "dynamic",
                "x":       _PERSON_START_X,
                "y":       _PERSON_START_Y,
                "radius":  _PERSON_RADIUS,
                "vx":      _PERSON_VX,
                "vy":      _PERSON_VY,
                "start_x": _PERSON_START_X,
                "start_y": _PERSON_START_Y,
                "label":   "person",
            }
        ]
        self._initial_obstacles = copy.deepcopy(_obstacles)
        self.obstacles          = copy.deepcopy(_obstacles)

    # ------------------------------------------------------------------
    # Update — advance the person each timestep
    # ------------------------------------------------------------------

    def update(self, t: float) -> None:
        """Move the person upward each dt step."""
        dt = config.DT
        for obs in self.obstacles:
            if obs.get("type") == "dynamic":
                obs["x"] += obs.get("vx", 0.0) * dt
                obs["y"] += obs.get("vy", 0.0) * dt


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = S2PersonCrossing()
    print(s)
    print(f"  Initial person y: {s.obstacles[0]['y']:.2f}")

    # Simulate 10 seconds of updates — person should cross y=10
    steps = int(10.0 / config.DT)
    t = 0.0
    for _ in range(steps):
        s.update(t)
        t += config.DT

    y_person = s.obstacles[0]["y"]
    print(f"  Person y after 10s: {y_person:.2f}  (expect ~10.0)")
    assert abs(y_person - 10.0) < 0.1, f"Person should be at y~10 after 10s, got {y_person}"

    # Reset
    s.reset()
    assert abs(s.obstacles[0]["y"] - _PERSON_START_Y) < 1e-6, "Reset failed"
    print("[PASS]  S2PersonCrossing OK")

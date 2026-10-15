"""
scenarios/scenario_base.py
===========================
Abstract base class that every scenario must inherit from.

A scenario defines:
  - start position / heading
  - goal position
  - the global reference path (list of (x,y) waypoints)
  - the obstacle set (static + dynamic)
  - map bounds
  - time limit per run

Concrete scenarios override __init__() to set up their layout,
and optionally override update(t) for dynamic obstacles.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Obstacle dict schema ──────────────────────────────────────────────────
#
# Static rectangle:
#   { "type": "static", "x": cx, "y": cy, "w": width, "h": height }
#
# Dynamic circle (person / moving agent):
#   { "type": "dynamic", "x": cx, "y": cy, "radius": r,
#     "vx": vx, "vy": vy,
#     "start_x": sx, "start_y": sy }   ← used for reset
#
# Static wall segment (thin rect, horizontal or vertical):
#   { "type": "wall", "x": cx, "y": cy, "w": w, "h": h }
#
# ── Path generation helpers ───────────────────────────────────────────────

def straight_path(
    x0: float, y0: float,
    x1: float, y1: float,
    n_pts: int = 40,
) -> List[Tuple[float, float]]:
    """Return a uniformly-spaced straight-line path."""
    xs = np.linspace(x0, x1, n_pts)
    ys = np.linspace(y0, y1, n_pts)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def dist_to_obstacle(rx: float, ry: float, obs: Dict[str, Any]) -> float:
    """
    Distance from point (rx, ry) to the surface of an obstacle.
    Returns 0.0 if inside.  Used by base class and metrics.
    """
    if obs.get("type") == "circle" or "radius" in obs:
        return max(0.0, math.hypot(rx - obs["x"], ry - obs["y"]) - obs["radius"])
    # Axis-aligned rectangle (static / wall / dynamic-rect)
    ox, oy = obs["x"], obs["y"]
    hw = obs.get("w", 1.0) / 2.0
    hh = obs.get("h", 1.0) / 2.0
    cx = float(np.clip(rx, ox - hw, ox + hw))
    cy = float(np.clip(ry, oy - hh, oy + hh))
    return math.hypot(rx - cx, ry - cy)


# ── Base class ────────────────────────────────────────────────────────────

class ScenarioBase(ABC):
    """
    Abstract base for all 5 campus-buggy scenarios.

    Attributes
    ----------
    name        : str    — human-readable scenario identifier
    start       : (x, y, theta)  — robot start pose
    goal        : (x, y)         — target position
    global_path : list of (x,y)  — reference waypoints
    obstacles   : list of dicts  — current obstacle state
    map_bounds  : (xmin, ymin, xmax, ymax)
    time_limit  : float  — seconds before run is marked 'timeout'
    """

    def __init__(self):
        self.name:        str                          = "base"
        self.start:       Tuple[float, float, float]  = (1.0, 10.0, 0.0)
        self.goal:        Tuple[float, float]         = (19.0, 10.0)
        self.global_path: List[Tuple[float, float]]   = []
        self.obstacles:   List[Dict[str, Any]]        = []
        self.map_bounds:  Tuple[float,float,float,float] = (0.0, 0.0,
                                                             config.MAP_W,
                                                             config.MAP_H)
        self.time_limit:  float                       = config.TIME_LIMIT

        # Internal: initial obstacle states for reset
        self._initial_obstacles: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Restore dynamic obstacles to their start positions.
        Call this before every run.
        """
        import copy
        self.obstacles = copy.deepcopy(self._initial_obstacles)
        self._on_reset()

    def _on_reset(self) -> None:
        """Hook for subclasses to do extra reset work (optional)."""
        pass

    def update(self, t: float) -> None:
        """
        Advance dynamic obstacles by one timestep.
        Default: update any obstacle with 'vx'/'vy' fields.

        Parameters
        ----------
        t : float  — current simulation time in seconds (used by Scenario 5)
        """
        dt = config.DT
        for obs in self.obstacles:
            if obs.get("type") == "dynamic":
                obs["x"] += obs.get("vx", 0.0) * dt
                obs["y"] += obs.get("vy", 0.0) * dt

    def get_obstacles(self) -> List[Dict[str, Any]]:
        """Return the current (potentially updated) obstacle list."""
        return self.obstacles

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def goal_reached(self, state: np.ndarray) -> bool:
        """True when robot centre is within GOAL_TOLERANCE of the goal."""
        dx = state[0] - self.goal[0]
        dy = state[1] - self.goal[1]
        return math.hypot(dx, dy) < config.GOAL_TOLERANCE

    def out_of_bounds(self, state: np.ndarray) -> bool:
        """True when robot centre is outside the map bounds."""
        xmin, ymin, xmax, ymax = self.map_bounds
        return (state[0] < xmin or state[0] > xmax or
                state[1] < ymin or state[1] > ymax)

    def min_obstacle_dist(self, state: np.ndarray) -> float:
        """Minimum distance from robot centre to any obstacle surface."""
        if not self.obstacles:
            return float("inf")
        return min(dist_to_obstacle(state[0], state[1], obs)
                   for obs in self.obstacles)

    def collision_detected(self, state: np.ndarray) -> bool:
        """True when robot centre is within ROBOT_RADIUS of any obstacle."""
        return self.min_obstacle_dist(state) < config.ROBOT_RADIUS

    # ------------------------------------------------------------------
    # Path helpers exposed to controllers
    # ------------------------------------------------------------------

    def get_path(self) -> List[Tuple[float, float]]:
        return self.global_path

    def get_start_state(self) -> np.ndarray:
        """Return start as a [x, y, theta, v=0] state vector."""
        x, y, theta = self.start
        return np.array([x, y, theta, 0.0], dtype=float)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"start={self.start}, goal={self.goal}, "
                f"n_obstacles={len(self.obstacles)})")


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ScenarioBase is abstract — no standalone test.")
    print("Import and instantiate a concrete scenario instead.")

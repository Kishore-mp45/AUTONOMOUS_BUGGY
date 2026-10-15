"""
scenarios/scenario_base.py
===========================
Abstract base class for all campus buggy simulation scenarios.
Upgraded with:
  - Unified Obstacle objects
  - Built-in Costmap2D integration (build_costmap())
  - Seed-controlled reproducibility (reset(seed=...))
  - Unified termination check (is_complete(state, t))
"""

import math
import copy
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from environment.obstacle import Obstacle
from environment.costmap import Costmap2D


def straight_path(
    x0: float, y0: float,
    x1: float, y1: float,
    n_pts: int = 40,
) -> List[Tuple[float, float]]:
    """Return a uniformly-spaced straight-line path."""
    xs = np.linspace(x0, x1, n_pts)
    ys = np.linspace(y0, y1, n_pts)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


class ScenarioBase(ABC):
    """
    Abstract base class for simulation scenarios.
    """

    def __init__(self):
        self.name:        str                         = "base"
        self.start:       Tuple[float, float, float] = (1.0, 10.0, 0.0)
        self.goal:        Tuple[float, float]        = (19.0, 10.0)
        self.global_path: List[Tuple[float, float]]  = []
        self.obstacles:   List[Obstacle]             = []
        self.static_obstacles: List[Obstacle]        = []
        self.map_bounds:  Tuple[float, float, float, float] = (0.0, 0.0, config.MAP_W, config.MAP_H)
        self.time_limit:  float                      = config.TIME_LIMIT
        self.costmap:     Optional[Costmap2D]        = None

        # Maneuver / lane guidance hooks (used by edge cases ec6–ec10):
        #   get_maneuver_command() — scenario-driven maneuver override
        #   disable_lane_cost      — ec8: buggy navigates without lane guidance
        #   sensor_noise           — ec9: (pos_std, heading_std) injected into
        #                             the state passed to the controller
        self.disable_lane_cost = False
        self.sensor_noise: Optional[Tuple[float, float]] = None

    def get_maneuver_command(self, t: float, state: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Optional scenario-driven maneuver override ("u_turn" | "reverse" | None).
        Called by main.py each step and fed to the ManeuverStateMachine.
        """
        return None

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Restore obstacles and environment to initial state.
        Optional seed sets numpy random state for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        for obs in self.obstacles:
            obs.reset()

        self.costmap = self.build_costmap()
        self._on_reset(seed)
        return self.get_start_state()

    def _on_reset(self, seed: Optional[int] = None):
        """Hook for subclasses to do scenario-specific reset logic."""
        pass

    def build_costmap(self) -> Costmap2D:
        """
        Build Costmap2D and stamp all static obstacles.
        """
        cm = Costmap2D()
        for obs in self.static_obstacles:
            cm.add_static_obstacle(obs)
        cm.update(self.obstacles)
        return cm

    def update(self, t: float, robot_state: Optional[np.ndarray] = None) -> None:
        """
        Advance dynamic obstacles by one timestep and refresh costmap.
        """
        dt = config.DT
        for obs in self.obstacles:
            obs.step(dt)
        if self.costmap is not None:
            self.costmap.update(self.obstacles)

    def get_obstacles(self) -> List[Any]:
        """Return current obstacles."""
        return self.obstacles

    def is_complete(
        self,
        state: np.ndarray,
        t: float,
        goal: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bool, bool, str]:
        """
        Unified termination check.
        Returns (done: bool, success: bool, reason: str).
        """
        x, y = float(state[0]), float(state[1])
        gx, gy = goal if goal is not None else self.goal
        if math.hypot(x - gx, y - gy) < config.GOAL_TOLERANCE:
            return True, True, ""
        if t >= self.time_limit:
            return True, False, "timeout"
        if self.out_of_bounds(state):
            return True, False, "out_of_bounds"
        return False, False, ""

    def clear_spawn_overlaps(self, state: np.ndarray, previously_active: set) -> List[str]:
        """Move newly activated obstacles clear of the vehicle before planning."""
        x, y, heading = float(state[0]), float(state[1]), float(state[2])
        adjusted = []
        for obstacle in self.obstacles:
            if id(obstacle) in previously_active or not obstacle.active:
                continue
            if obstacle.dist_to_point(x, y) >= config.ROBOT_RADIUS:
                continue

            dx, dy = obstacle.x - x, obstacle.y - y
            distance = math.hypot(dx, dy)
            if distance < 1e-9:
                dx, dy = math.cos(heading), math.sin(heading)
                distance = 1.0
            direction_x, direction_y = dx / distance, dy / distance
            if obstacle.radius is not None:
                extent = obstacle.radius
            elif obstacle.width is not None and obstacle.height is not None:
                extent = math.hypot(obstacle.width / 2.0, obstacle.height / 2.0)
            else:
                extent = 0.0

            offset = extent + config.ROBOT_RADIUS + 0.5
            obstacle.x = x + direction_x * offset
            obstacle.y = y + direction_y * offset
            adjusted.append(obstacle.label or "unnamed obstacle")
            print(
                f"[WARN] Spawn overlap for {adjusted[-1]}; moved it "
                f"{offset:.2f} m clear of the vehicle."
            )
        return adjusted

    def goal_reached(self, state: np.ndarray) -> bool:
        """True when robot is within GOAL_TOLERANCE of goal."""
        dx = state[0] - self.goal[0]
        dy = state[1] - self.goal[1]
        return math.hypot(dx, dy) < config.GOAL_TOLERANCE

    def out_of_bounds(self, state: np.ndarray) -> bool:
        """True when robot is outside map boundaries."""
        xmin, ymin, xmax, ymax = self.map_bounds
        return (state[0] < xmin or state[0] > xmax or
                state[1] < ymin or state[1] > ymax)

    def min_obstacle_dist(self, state: np.ndarray) -> float:
        """Minimum distance from robot center to any active obstacle surface."""
        active = [obs for obs in self.obstacles if getattr(obs, "active", True)]
        if not active:
            return float("inf")
        return min(obs.dist_to_point(state[0], state[1]) for obs in active)

    def collision_detected(self, state: np.ndarray) -> bool:
        """True when robot center is within ROBOT_RADIUS of an obstacle."""
        return self.min_obstacle_dist(state) < config.ROBOT_RADIUS

    def get_path(self) -> List[Tuple[float, float]]:
        return self.global_path

    def get_start_state(self) -> np.ndarray:
        x, y, theta = self.start
        return np.array([x, y, theta, 0.0], dtype=float)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"start={self.start}, goal={self.goal}, "
                f"n_obstacles={len(self.obstacles)})")

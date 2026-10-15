"""
controllers/rpp.py
==================
Regulated Pure Pursuit (RPP) controller.
Upgraded with adaptive lookahead, low-pass steering filter,
curvature-based speed regulation, and Costmap2D integration.
Inherits from ControllerBase.
"""

import math
import numpy as np
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.controller_base import ControllerBase


class RegulatedPurePursuit(ControllerBase):
    """
    Regulated Pure Pursuit controller for path tracking with adaptive lookahead
    and costmap-aware speed regulation.
    """
    name = "RPP"

    def __init__(
        self,
        lookahead_base:     float = config.RPP_LOOKAHEAD,
        lookahead_gain:     float = config.RPP_LOOKAHEAD_GAIN,
        min_lookahead:      float = config.RPP_MIN_LOOKAHEAD,
        max_lookahead:      float = config.RPP_MAX_LOOKAHEAD,
        max_speed:          float = config.RPP_MAX_SPEED,
        min_speed:          float = config.RPP_MIN_SPEED,
        wheelbase:          float = config.WHEELBASE,
        steer_lowpass:      float = config.RPP_STEER_LOWPASS,
        obstacle_slowdown:  bool  = config.RPP_OBSTACLE_SLOWDOWN,
        curvature_slowdown: bool  = config.RPP_CURVATURE_SLOWDOWN,
    ):
        self.lookahead_base     = float(lookahead_base)
        self.lookahead_gain     = float(lookahead_gain)
        self.min_lookahead      = float(min_lookahead)
        self.max_lookahead      = float(max_lookahead)
        self.max_speed          = float(max_speed)
        self.min_speed          = float(min_speed)
        self.L                  = float(wheelbase)
        self.steer_lowpass      = float(steer_lowpass)
        self.obstacle_slowdown  = bool(obstacle_slowdown)
        self.curvature_slowdown = bool(curvature_slowdown)

        self._prev_steer = 0.0

    def reset(self):
        """Reset internal filter state between runs."""
        self._prev_steer = 0.0

    def _get_lookahead(self, v: float) -> float:
        """
        Adaptive lookahead grows with speed:
        L_d = base + gain * |v|, clamped to [min, max].
        """
        L_d = self.lookahead_base + self.lookahead_gain * abs(v)
        return float(np.clip(L_d, self.min_lookahead, self.max_lookahead))

    def _filter_steer(self, new_steer: float) -> float:
        """Exponential moving average low-pass filter on steering command."""
        self._prev_steer = (self.steer_lowpass * self._prev_steer +
                            (1.0 - self.steer_lowpass) * new_steer)
        return float(self._prev_steer)

    def compute_control(
        self,
        state: np.ndarray,
        path: List[Tuple[float, float]],
        goal: Optional[Tuple[float, float]] = None,
        costmap = None,
        obstacles: list = None,
    ) -> Tuple[float, float]:
        """
        Compute (steering_angle, velocity).
        """
        x, y, theta, v = float(state[0]), float(state[1]), float(state[2]), float(state[3])

        # Goal check
        if goal is not None and math.hypot(goal[0] - x, goal[1] - y) < config.GOAL_TOLERANCE:
            return 0.0, 0.0

        L_d = self._get_lookahead(v)

        # ── 1. Find lookahead point ──────────────────────────────────
        lookahead_pt = self._find_lookahead_point(state, path, L_d)
        if lookahead_pt is None:
            return 0.0, 0.0  # end of path

        lx, ly = lookahead_pt

        # ── 2. Compute curvature and steering angle ──────────────────
        dx = lx - x
        dy = ly - y
        angle_to_target = math.atan2(dy, dx)
        alpha = _wrap_angle(angle_to_target - theta)

        dist_to_lookahead = max(1e-4, math.hypot(dx, dy))
        kappa = 2.0 * math.sin(alpha) / dist_to_lookahead

        steer = math.atan2(2.0 * self.L * math.sin(alpha), dist_to_lookahead)
        steer = float(np.clip(steer, -config.MAX_STEER, config.MAX_STEER))

        # Apply low-pass filter
        steer = self._filter_steer(steer)

        # ── 3. Speed regulation ──────────────────────────────────────
        speed = self.max_speed

        if self.curvature_slowdown:
            curvature = abs(math.tan(steer) / self.L)
            speed = speed / (1.0 + 2.0 * curvature)

        if self.obstacle_slowdown:
            if costmap is not None:
                cost_at_pos = costmap.query(x, y)
                if cost_at_pos > 50:
                    speed *= max(0.2, (1.0 - cost_at_pos / 100.0))
            elif obstacles:
                min_d = _min_obstacle_dist(x, y, obstacles)
                if min_d < 3.0:
                    scale = (min_d - config.ROBOT_RADIUS) / (3.0 - config.ROBOT_RADIUS)
                    speed *= float(np.clip(scale, 0.2, 1.0))

        speed = float(np.clip(speed, self.min_speed, self.max_speed))
        return steer, speed

    def _find_lookahead_point(
        self,
        state: np.ndarray,
        path: List[Tuple[float, float]],
        lookahead_dist: float,
    ) -> Optional[Tuple[float, float]]:
        """Find the forward waypoint on path that is >= lookahead_dist away."""
        x, y = state[0], state[1]
        if not path:
            return None

        dists = [math.hypot(px - x, py - y) for px, py in path]
        closest_idx = int(np.argmin(dists))

        for i in range(closest_idx, len(path)):
            px, py = path[i]
            if math.hypot(px - x, py - y) >= lookahead_dist:
                return (px, py)

        return path[-1]

    def __repr__(self) -> str:
        return (
            f"RegulatedPurePursuit("
            f"L_base={self.lookahead_base}, gain={self.lookahead_gain}, "
            f"v=[{self.min_speed},{self.max_speed}])"
        )


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _min_obstacle_dist(x: float, y: float, obstacles: list) -> float:
    min_d = float("inf")
    for obs in obstacles:
        if hasattr(obs, "dist_to_point"):
            d = obs.dist_to_point(x, y)
        elif isinstance(obs, dict):
            if obs.get("type") == "circle" or "radius" in obs:
                d = max(0.0, math.hypot(x - obs["x"], y - obs["y"]) - obs["radius"])
            else:
                hw = obs.get("w", 1.0) / 2.0
                hh = obs.get("h", 1.0) / 2.0
                cx = float(np.clip(x, obs["x"] - hw, obs["x"] + hw))
                cy = float(np.clip(y, obs["y"] - hh, obs["y"] + hh))
                d = math.hypot(x - cx, y - cy)
        else:
            d = float("inf")
        if d < min_d:
            min_d = d
    return min_d


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Upgraded RPP Self-Test ===")
    ctrl = RegulatedPurePursuit()
    path = [(float(i), 10.0) for i in range(21)]
    state = np.array([1.0, 10.0, 0.0, 0.0])
    steer, speed = ctrl.compute_control(state, path, goal=(20.0, 10.0))
    print(f"Start: steer={steer:.4f}, speed={speed:.3f}")
    assert abs(steer) < 0.05
    print("[PASS] Upgraded RPP initialized and computed control cleanly.")

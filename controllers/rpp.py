"""
controllers/rpp.py
==================
Regulated Pure Pursuit (RPP) controller.

Concept:
  Look ahead a fixed distance (lookahead_dist) along the reference path,
  compute the curvature needed to steer toward that point, then regulate
  (reduce) speed based on:
    1. Path curvature — slow down on tight curves
    2. Obstacle proximity — slow down when close to obstacles

Pure Pursuit curvature formula:
    kappa = 2 * sin(alpha) / L_d
    where:
        alpha  = angle from robot heading to lookahead point
        L_d    = lookahead distance
    Steering angle: delta = arctan(kappa * L)   (L = wheelbase)

Speed regulation:
    v_cmd = max(v_min, v_max * min(k_curv, k_obs))
    where k_curv, k_obs are scale factors in [0, 1].
"""

import math
import numpy as np
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class RegulatedPurePursuit:
    """
    Regulated Pure Pursuit controller for path tracking with speed regulation.

    Parameters
    ----------
    lookahead_dist : float
        How far ahead on the path to target (metres).
    max_speed : float
        Maximum commanded speed (m/s).
    min_speed : float
        Minimum commanded speed when slowing down (m/s).
    wheelbase : float
        Robot wheelbase in metres (needed for steering angle conversion).
    obstacle_slowdown : bool
        If True, reduce speed when obstacles are close.
    curvature_slowdown : bool
        If True, reduce speed on high-curvature path segments.
    """

    def __init__(
        self,
        lookahead_dist:    float = config.RPP_LOOKAHEAD,
        max_speed:         float = config.RPP_MAX_SPEED,
        min_speed:         float = config.RPP_MIN_SPEED,
        wheelbase:         float = config.WHEELBASE,
        obstacle_slowdown: bool  = config.RPP_OBSTACLE_SLOWDOWN,
        curvature_slowdown:bool  = config.RPP_CURVATURE_SLOWDOWN,
    ):
        self.L_d               = lookahead_dist
        self.max_speed         = max_speed
        self.min_speed         = min_speed
        self.L                 = wheelbase
        self.obstacle_slowdown = obstacle_slowdown
        self.curvature_slowdown= curvature_slowdown

        # Obstacle slowdown start / stop distances
        self._obs_start_dist = 3.0   # metres — begin slowing
        self._obs_stop_dist  = config.ROBOT_RADIUS + 0.1   # stop threshold

        # Curvature threshold above which we start slowing (rad/m)
        self._curv_threshold = 0.5

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_control(
        self,
        state:     np.ndarray,
        path:      List[Tuple[float, float]],
        obstacles: List[dict],
    ) -> Tuple[float, float]:
        """
        Compute the control command (steering_angle, velocity).

        Parameters
        ----------
        state : np.ndarray [x, y, theta, v]
            Current robot state.
        path : list of (x, y) tuples
            Ordered global path waypoints from start to goal.
        obstacles : list of dicts
            Each obstacle has at least 'x', 'y', and 'radius' or 'w'/'h'.

        Returns
        -------
        (delta, v_cmd) : (float, float)
            Steering angle in radians and commanded speed in m/s.
        """
        x, y, theta, v = state

        # ── 1. Find lookahead point ──────────────────────────────────
        lookahead_pt = self._find_lookahead_point(state, path)
        if lookahead_pt is None:
            # No valid lookahead — probably at/past the goal; stop
            return 0.0, 0.0

        lx, ly = lookahead_pt

        # ── 2. Compute alpha: angle from heading to lookahead ────────
        dx = lx - x
        dy = ly - y
        angle_to_target = math.atan2(dy, dx)
        alpha = _wrap_angle(angle_to_target - theta)

        # ── 3. Pure pursuit curvature & steering angle ───────────────
        # Avoid division by zero when lookahead distance is tiny
        dist_to_lookahead = math.hypot(dx, dy)
        if dist_to_lookahead < 1e-4:
            kappa = 0.0
        else:
            kappa = 2.0 * math.sin(alpha) / dist_to_lookahead

        # Convert curvature to front-wheel steering angle: δ = arctan(κ·L)
        delta = math.atan(kappa * self.L)
        delta = float(np.clip(delta, -config.MAX_STEER, config.MAX_STEER))

        # ── 4. Speed regulation ──────────────────────────────────────
        speed_scale = 1.0

        if self.curvature_slowdown:
            # Slow down proportional to curvature magnitude
            abs_kappa = abs(kappa)
            if abs_kappa > self._curv_threshold:
                # scale linearly: 1.0 at threshold, 0.3 at kappa=3.0
                curv_scale = max(0.3, 1.0 - (abs_kappa - self._curv_threshold) / 3.0)
                speed_scale = min(speed_scale, curv_scale)

        if self.obstacle_slowdown and obstacles:
            min_obs_dist = self._min_obstacle_dist(state, obstacles)
            if min_obs_dist < self._obs_start_dist:
                # Linear slow-down from obs_start_dist to stop_dist
                obs_scale = (min_obs_dist - self._obs_stop_dist) / \
                            (self._obs_start_dist - self._obs_stop_dist)
                obs_scale = float(np.clip(obs_scale, 0.0, 1.0))
                speed_scale = min(speed_scale, obs_scale)

        v_cmd = self.max_speed * speed_scale
        v_cmd = float(np.clip(v_cmd, self.min_speed, self.max_speed))

        return delta, v_cmd

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_lookahead_point(
        self,
        state: np.ndarray,
        path:  List[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """
        Find the first point on the path that is >= lookahead_dist from the robot.
        Searches from the closest path point forward to avoid backward pursuit.
        """
        x, y = state[0], state[1]

        if not path:
            return None

        # Find the index of the closest path point
        dists = [math.hypot(px - x, py - y) for px, py in path]
        closest_idx = int(np.argmin(dists))

        # Search forward from closest index for the lookahead point
        for i in range(closest_idx, len(path)):
            px, py = path[i]
            d = math.hypot(px - x, py - y)
            if d >= self.L_d:
                return (px, py)

        # If no point is far enough, return the last waypoint
        return path[-1]

    def _min_obstacle_dist(
        self,
        state:     np.ndarray,
        obstacles: List[dict],
    ) -> float:
        """Return the minimum clearance distance from robot centre to any obstacle surface."""
        x, y = state[0], state[1]
        min_dist = float("inf")

        for obs in obstacles:
            d = _dist_to_obstacle(x, y, obs)
            if d < min_dist:
                min_dist = d

        return min_dist

    def __repr__(self) -> str:
        return (
            f"RegulatedPurePursuit("
            f"L_d={self.L_d}, "
            f"v=[{self.min_speed},{self.max_speed}], "
            f"obs_slow={self.obstacle_slowdown}, "
            f"curv_slow={self.curvature_slowdown})"
        )


# ── Module-level helpers ──────────────────────────────────────────────────

def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _dist_to_obstacle(rx: float, ry: float, obs: dict) -> float:
    """
    Distance from robot centre (rx, ry) to the surface of an obstacle.
    Supports 'circle' (radius) and 'rect' (w, h) obstacle types.
    Returns 0.0 if inside the obstacle.
    """
    obs_type = obs.get("type", "static")

    if obs_type == "circle" or "radius" in obs:
        dx = rx - obs["x"]
        dy = ry - obs["y"]
        return max(0.0, math.hypot(dx, dy) - obs["radius"])

    # Default: treat as axis-aligned rectangle
    ox, oy = obs["x"], obs["y"]
    hw = obs.get("w", 1.0) / 2.0
    hh = obs.get("h", 1.0) / 2.0

    # Clamp robot position to rect and measure distance
    closest_x = float(np.clip(rx, ox - hw, ox + hw))
    closest_y = float(np.clip(ry, oy - hh, oy + hh))
    return math.hypot(rx - closest_x, ry - closest_y)


# ── Quick self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RPP self-test ===")
    ctrl = RegulatedPurePursuit()
    print(ctrl)

    # Straight path: y=10, x from 0 to 20
    path = [(float(i), 10.0) for i in range(21)]

    # Robot at (1, 10), heading East, speed 0
    state = np.array([1.0, 10.0, 0.0, 0.0])
    obstacles = [{"type": "static", "x": 10, "y": 10, "w": 1.5, "h": 1.5}]

    delta, v = ctrl.compute_control(state, path, obstacles)
    print(f"At (1,10) heading East: delta={delta:.4f} rad, v={v:.3f} m/s")
    assert abs(delta) < 0.01, "Expected near-zero steer on straight path at start"

    # Robot slightly off-path
    state2 = np.array([5.0, 9.0, 0.2, 1.0])
    delta2, v2 = ctrl.compute_control(state2, path, [])
    print(f"At (5,9) off-path: delta={delta2:.4f} rad, v={v2:.3f} m/s")
    assert delta2 > 0, "Expected positive steer to get back to y=10"

    # Close to obstacle — should slow down
    state3 = np.array([7.5, 10.0, 0.0, 1.0])
    delta3, v3 = ctrl.compute_control(state3, path, obstacles)
    print(f"At (7.5,10) near obstacle: delta={delta3:.4f} rad, v={v3:.3f} m/s")
    assert v3 < ctrl.max_speed, "Expected reduced speed near obstacle"

    print("\n[PASS]  RPP OK")

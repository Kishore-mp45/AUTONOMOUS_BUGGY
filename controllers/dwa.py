"""
controllers/dwa.py
==================
Dynamic Window Approach (DWA) controller.
Upgraded to true Ackermann dynamic window sampling (v, delta),
minimum turning radius constraint, steering slew rate limits,
and Costmap2D integration. Inherits from ControllerBase.
"""

import math
import numpy as np
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.controller_base import ControllerBase


class DWA(ControllerBase):
    """
    Dynamic Window Approach controller with Ackermann kinematic window (v, delta).
    """
    name = "DWA"

    def __init__(
        self,
        max_speed:       float = config.MAX_SPEED,
        max_accel:       float = config.MAX_ACCEL,
        max_steer:       float = config.MAX_STEER,
        dt:              float = config.DT,
        v_samples:       int   = config.DWA_V_SAMPLES,
        steer_samples:   int   = config.DWA_STEER_SAMPLES,
        steer_rate:      float = config.DWA_STEER_RATE,
        min_turn_radius: float = config.DWA_MIN_TURN_RADIUS,
        sim_time:        float = config.DWA_SIM_TIME,
        heading_w:       float = config.DWA_HEADING_W,
        clearance_w:     float = config.DWA_CLEARANCE_W,
        speed_w:         float = config.DWA_SPEED_W,
    ):
        self.max_speed       = float(max_speed)
        self.max_accel       = float(max_accel)
        self.max_steer       = float(max_steer)
        self.dt              = float(dt)
        self.v_samples       = int(v_samples)
        self.steer_samples   = int(steer_samples)
        self.steer_rate      = float(steer_rate)
        self.min_turn_radius = float(min_turn_radius)
        self.sim_time        = float(sim_time)
        self.heading_w       = float(heading_w)
        self.clearance_w     = float(clearance_w)
        self.speed_w         = float(speed_w)

        self._L = config.WHEELBASE
        self._prev_steer = 0.0

    def reset(self):
        """Reset internal steering memory between runs."""
        self._prev_steer = 0.0

    def _build_window(self, state: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Reachable (v, delta) in one timestep given:
          - acceleration limits on v
          - steering rate limit on delta
        """
        v = float(state[3])
        v_min = max(0.0, v - self.max_accel * self.dt)
        v_max = min(self.max_speed, v + self.max_accel * self.dt)

        d_min = max(-self.max_steer, self._prev_steer - self.steer_rate * self.dt)
        d_max = min(self.max_steer,  self._prev_steer + self.steer_rate * self.dt)
        return v_min, v_max, d_min, d_max

    def _is_ackermann_feasible(self, v: float, delta: float) -> bool:
        """Reject (v, delta) if turning radius is below physical minimum."""
        if abs(delta) < 1e-5:
            return True
        radius = abs(self._L / math.tan(delta))
        return radius >= self.min_turn_radius

    def _simulate_trajectory(self, state: np.ndarray, v: float, delta: float) -> List[Tuple[float, float]]:
        """
        Forward simulate bicycle model for sim_time seconds.
        Returns list of (x, y) coordinates along trajectory.
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        steps = int(self.sim_time / self.dt)
        traj = []
        for _ in range(steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += (v / self._L) * math.tan(delta) * self.dt
            traj.append((x, y))
        return traj

    def compute_control(
        self,
        state: np.ndarray,
        path: List[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        costmap = None,
        obstacles: list = None,
    ) -> Tuple[float, float]:
        """
        Compute (steering_angle, velocity) using Ackermann dynamic window.
        """
        x, y, theta, v_curr = float(state[0]), float(state[1]), float(state[2]), float(state[3])

        # Goal target fallback
        if goal is None:
            goal = path[-1] if path else (x + 1.0, y)

        if math.hypot(goal[0] - x, goal[1] - y) < config.GOAL_TOLERANCE:
            return 0.0, 0.0

        v_min, v_max, d_min, d_max = self._build_window(state)

        v_samples = np.linspace(v_min, v_max, self.v_samples)
        steer_samples = np.linspace(d_min, d_max, self.steer_samples)

        V_grid, D_grid = np.meshgrid(v_samples, steer_samples)
        V_flat = V_grid.ravel()
        D_flat = D_grid.ravel()
        n_candidates = len(V_flat)

        # 1. Ackermann feasibility check
        feasible = np.ones(n_candidates, dtype=bool)
        if self.min_turn_radius > 0:
            non_zero = np.abs(D_flat) >= 1e-5
            radii = np.full(n_candidates, float('inf'))
            radii[non_zero] = np.abs(self._L / np.tan(D_flat[non_zero]))
            feasible = radii >= self.min_turn_radius

        if not np.any(feasible):
            self._prev_steer = 0.0
            return 0.0, 0.0

        # Filter candidate pool to feasible ones
        cand_v = V_flat[feasible]
        cand_d = D_flat[feasible]
        M = len(cand_v)

        # 2. Vectorized forward simulation of all M trajectories
        steps = int(self.sim_time / self.dt)
        xs = np.zeros((M, steps))
        ys = np.zeros((M, steps))
        cur_x = np.full(M, x)
        cur_y = np.full(M, y)
        cur_th = np.full(M, theta)

        for s in range(steps):
            cur_x += cand_v * np.cos(cur_th) * self.dt
            cur_y += cand_v * np.sin(cur_th) * self.dt
            cur_th += (cand_v / self._L) * np.tan(cand_d) * self.dt
            xs[:, s] = cur_x
            ys[:, s] = cur_y

        # 3. Collision check
        if costmap is not None:
            traj_costs = costmap.query_batch(xs, ys)
            collision_mask = np.any(traj_costs >= config.LETHAL_COST, axis=1)
            clearance_scores = 1.0 - (np.max(traj_costs, axis=1) / 100.0)
        elif obstacles:
            r = config.ROBOT_RADIUS
            # Distance checking across all trajectory points
            min_dists = np.full(M, float('inf'))
            for px_step, py_step in zip(xs.T, ys.T):
                for k in range(M):
                    d = _min_dist_to_obs(px_step[k], py_step[k], obstacles)
                    if d < min_dists[k]:
                        min_dists[k] = d
            collision_mask = min_dists < r
            clearance_scores = np.clip(min_dists / 3.0, 0.0, 1.0)
        else:
            collision_mask = np.zeros(M, dtype=bool)
            clearance_scores = np.ones(M)

        valid_mask = ~collision_mask
        if not np.any(valid_mask):
            # Trapped fallback
            self._prev_steer = 0.0
            return 0.0, 0.0

        # 4. Heading score
        last_xs = xs[valid_mask, -1]
        last_ys = ys[valid_mask, -1]
        last_ths = cur_th[valid_mask]
        goal_angles = np.arctan2(goal[1] - last_ys, goal[0] - last_xs)
        heading_errs = np.abs(np.arctan2(np.sin(goal_angles - last_ths), np.cos(goal_angles - last_ths)))
        heading_scores = 1.0 - heading_errs / math.pi

        # 5. Speed score
        valid_v = cand_v[valid_mask]
        valid_d = cand_d[valid_mask]
        speed_scores = valid_v / self.max_speed if self.max_speed > 0 else np.zeros_like(valid_v)
        valid_clearance = clearance_scores[valid_mask]

        # 6. Total score
        scores = (self.heading_w   * heading_scores +
                  self.clearance_w * valid_clearance +
                  self.speed_w     * speed_scores)

        best_idx = np.argmax(scores)
        best_v = valid_v[best_idx]
        best_d = valid_d[best_idx]

        self._prev_steer = best_d
        return float(best_d), float(best_v)

    def __repr__(self) -> str:
        return (
            f"DWA(v_samples={self.v_samples}, steer_samples={self.steer_samples}, "
            f"sim_time={self.sim_time}, min_turn_r={self.min_turn_radius})"
        )


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _min_dist_to_obs(px: float, py: float, obstacles: list) -> float:
    min_d = float('inf')
    for obs in obstacles:
        if hasattr(obs, "dist_to_point"):
            d = obs.dist_to_point(px, py)
        elif isinstance(obs, dict):
            if obs.get("type") == "circle" or "radius" in obs:
                d = max(0.0, math.hypot(px - obs["x"], py - obs["y"]) - obs["radius"])
            else:
                hw = obs.get("w", 1.0) / 2.0
                hh = obs.get("h", 1.0) / 2.0
                cx = float(np.clip(px, obs["x"] - hw, obs["x"] + hw))
                cy = float(np.clip(py, obs["y"] - hh, obs["y"] + hh))
                d = math.hypot(px - cx, py - cy)
        else:
            d = float('inf')
        if d < min_d:
            min_d = d
    return min_d


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Upgraded Ackermann DWA Self-Test ===")
    dwa = DWA()
    state = np.array([1.0, 10.0, 0.0, 0.5])
    delta, v = dwa.compute_control(state, goal=(19.0, 10.0))
    print(f"DWA computed: delta={delta:.4f} rad, v={v:.3f} m/s")
    assert abs(delta) <= config.MAX_STEER
    assert 0.0 <= v <= config.MAX_SPEED
    print("[PASS] Upgraded Ackermann DWA self-test passed.")

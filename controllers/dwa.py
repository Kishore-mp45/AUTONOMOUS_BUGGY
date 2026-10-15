"""
controllers/dwa.py
==================
Dynamic Window Approach (DWA) controller.

Concept:
  At each timestep, build a 'dynamic window' of velocity commands that are
  reachable within the next dt given the robot's acceleration limits.
  Sample a grid of (v, omega) pairs inside the window, simulate each as a
  short trajectory, score them on three criteria, and return the best.

Scoring (higher = better for all three):
    heading_score   — how directly the trajectory points at the goal
    clearance_score — minimum distance to any obstacle along the trajectory
    velocity_score  — prefer higher linear speeds

Final score = alpha*heading + beta*clearance + gamma*velocity
(alpha, beta, gamma from config)

Note on bicycle model:
  DWA is traditionally formulated for differential-drive robots (v, omega).
  We adapt it here for the bicycle model by converting omega → steering_angle:
      delta = arctan(omega * L / v)    [clipped to max_steer]
  Angular velocity omega = v * tan(delta) / L
"""

import math
import numpy as np
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DWA:
    """
    Dynamic Window Approach controller.

    Parameters
    ----------
    max_speed : float       Maximum linear speed (m/s).
    max_accel : float       Maximum acceleration (m/s²).
    max_steer : float       Maximum steering angle (rad).
    dt        : float       Simulation timestep (s).
    v_samples : int         Number of speed samples.
    w_samples : int         Number of angular-velocity samples.
    sim_time  : float       How far ahead to simulate each trajectory (s).
    heading_w : float       Weight for heading-to-goal score.
    clearance_w : float     Weight for obstacle clearance score.
    speed_w   : float       Weight for speed score.
    """

    def __init__(
        self,
        max_speed:   float = config.MAX_SPEED,
        max_accel:   float = config.MAX_ACCEL,
        max_steer:   float = config.MAX_STEER,
        dt:          float = config.DT,
        v_samples:   int   = config.DWA_V_SAMPLES,
        w_samples:   int   = config.DWA_W_SAMPLES,
        sim_time:    float = config.DWA_SIM_TIME,
        heading_w:   float = config.DWA_HEADING_W,
        clearance_w: float = config.DWA_CLEARANCE_W,
        speed_w:     float = config.DWA_SPEED_W,
    ):
        self.max_speed   = max_speed
        self.max_accel   = max_accel
        self.max_steer   = max_steer
        self.dt          = dt
        self.v_samples   = v_samples
        self.w_samples   = w_samples
        self.sim_time    = sim_time
        self.heading_w   = heading_w
        self.clearance_w = clearance_w
        self.speed_w     = speed_w

        # Maximum yaw rate corresponding to max_steer at max_speed
        # omega_max = v * tan(delta_max) / L
        self._L = config.WHEELBASE

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_control(
        self,
        state:     np.ndarray,
        goal:      Tuple[float, float],
        obstacles: List[dict],
    ) -> Tuple[float, float]:
        """
        Compute the best (steering_angle, velocity) using DWA.
        Fully vectorised over all (v, omega) samples — no Python loops.

        Returns
        -------
        (delta, v_cmd) : steering angle (rad) and linear speed (m/s)
        """
        x, y, theta, v_curr = float(state[0]), float(state[1]), float(state[2]), float(state[3])

        # ── 1. Build dynamic window ──────────────────────────────────
        v_min_dw = max(0.0,            v_curr - self.max_accel * self.dt)
        v_max_dw = min(self.max_speed, v_curr + self.max_accel * self.dt)

        v_ref        = max(v_curr, 0.1)
        omega_max_dw = v_ref * math.tan(self.max_steer) / self._L

        v_1d = np.linspace(v_min_dw, v_max_dw, self.v_samples)   # (Nv,)
        w_1d = np.linspace(-omega_max_dw, omega_max_dw, self.w_samples)  # (Nw,)

        # Meshgrid → flat arrays of all (v, omega) pairs, shape (N,)
        V, W = np.meshgrid(v_1d, w_1d, indexing='ij')  # (Nv, Nw)
        V_f  = V.ravel()   # (N,)
        W_f  = W.ravel()   # (N,)
        N    = len(V_f)

        # ── 2. Vectorised trajectory rollout ─────────────────────────
        # For constant (v, omega): heading evolves as:
        #   theta_t = theta_0 + omega * t
        # Position: x_t = x_0 + v * sum_{k=0}^{t-1} cos(theta_k) * dt
        steps = int(self.sim_time / self.dt)

        # Time steps array
        k_arr = np.arange(steps, dtype=float)          # (T,)

        # Heading at each step: shape (N, T)
        # theta_k = theta + omega * k * dt
        thetas = theta + W_f[:, np.newaxis] * k_arr[np.newaxis, :] * self.dt  # (N,T)

        # Positions via cumulative integration: shape (N, T)
        dx_steps = V_f[:, np.newaxis] * np.cos(thetas) * self.dt  # (N, T)
        dy_steps = V_f[:, np.newaxis] * np.sin(thetas) * self.dt  # (N, T)

        xs = x + np.cumsum(dx_steps, axis=1)   # (N, T)
        ys = y + np.cumsum(dy_steps, axis=1)   # (N, T)

        # Final heading (last step), shape (N,)
        theta_end = thetas[:, -1]

        # ── 3. Collision mask ─────────────────────────────────────────
        # collides[n] = True if any step of trajectory n hits an obstacle
        collides = np.zeros(N, dtype=bool)
        r = config.ROBOT_RADIUS
        for obs in obstacles:
            # d[n, t] = distance from sample n at step t to obstacle surface
            d = _batch_dist_obstacle_2d(xs, ys, obs)   # (N, T)
            collides |= np.any(d < r, axis=1)           # (N,)

        # ── 4. Score valid trajectories ───────────────────────────────
        valid = ~collides   # (N,)

        if not np.any(valid):
            return 0.0, 0.0   # emergency stop

        # --- Heading score ---
        gx, gy = goal
        goal_angle  = np.arctan2(gy - ys[:, -1], gx - xs[:, -1])  # (N,)
        heading_err = np.abs(np.arctan2(
            np.sin(goal_angle - theta_end),
            np.cos(goal_angle - theta_end)
        ))                                                           # (N,)
        h_scores = 1.0 - heading_err / math.pi                      # (N,)

        # --- Clearance score ---
        if obstacles:
            min_clearance = np.full(N, np.inf)
            for obs in obstacles:
                d = _batch_dist_obstacle_2d(xs, ys, obs)    # (N, T)
                d_surf = d - r
                min_c  = np.min(d_surf, axis=1)             # (N,)
                min_clearance = np.minimum(min_clearance, min_c)
            min_clearance = np.clip(min_clearance, 0.0, 3.0)
            c_scores = min_clearance / 3.0
        else:
            c_scores = np.ones(N)

        # --- Speed score ---
        s_scores = V_f / self.max_speed   # (N,)

        # --- Combined score (invalid → -inf) ---
        scores = (self.heading_w   * h_scores +
                  self.clearance_w * c_scores +
                  self.speed_w     * s_scores)
        scores[~valid] = -np.inf

        # ── 5. Pick best ──────────────────────────────────────────────
        best_idx = int(np.argmax(scores))
        best_v   = float(V_f[best_idx])
        best_w   = float(W_f[best_idx])

        # Convert omega → steering angle
        if abs(best_v) < 1e-4:
            best_delta = 0.0
        else:
            best_delta = math.atan(best_w * self._L / best_v)
        best_delta = float(np.clip(best_delta, -self.max_steer, self.max_steer))

        return best_delta, best_v

    def __repr__(self) -> str:
        return (
            f"DWA(v_samples={self.v_samples}, w_samples={self.w_samples}, "
            f"sim_time={self.sim_time}, "
            f"weights=[h={self.heading_w},c={self.clearance_w},s={self.speed_w}])"
        )


# ── Module-level helpers ──────────────────────────────────────────────────

def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _dist_to_obstacle(rx: float, ry: float, obs: dict) -> float:
    """Distance from point (rx, ry) to obstacle surface. 0 if inside."""
    if obs.get("type") == "circle" or "radius" in obs:
        return max(0.0, math.hypot(rx - obs["x"], ry - obs["y"]) - obs["radius"])
    ox, oy = obs["x"], obs["y"]
    hw = obs.get("w", 1.0) / 2.0
    hh = obs.get("h", 1.0) / 2.0
    cx = float(np.clip(rx, ox - hw, ox + hw))
    cy = float(np.clip(ry, oy - hh, oy + hh))
    return math.hypot(rx - cx, ry - cy)


def _batch_dist_obstacle_2d(
    xs:  np.ndarray,   # shape (N, T)
    ys:  np.ndarray,   # shape (N, T)
    obs: dict,
) -> np.ndarray:
    """
    Vectorised distance from N×T points to one obstacle surface.
    Returns array shape (N, T).
    """
    if obs.get("type") == "circle" or "radius" in obs:
        d = np.sqrt((xs - obs["x"])**2 + (ys - obs["y"])**2)
        return np.maximum(0.0, d - obs["radius"])
    # Axis-aligned rectangle
    ox, oy = obs["x"], obs["y"]
    hw = obs.get("w", 1.0) / 2.0
    hh = obs.get("h", 1.0) / 2.0
    cx = np.clip(xs, ox - hw, ox + hw)
    cy = np.clip(ys, oy - hh, oy + hh)
    return np.sqrt((xs - cx)**2 + (ys - cy)**2)



# ── Quick self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DWA self-test ===")
    ctrl = DWA()
    print(ctrl)

    goal = (19.0, 10.0)
    obstacles = [{"type": "static", "x": 10, "y": 10, "w": 1.5, "h": 1.5}]

    # Robot at start, heading East
    state = np.array([1.0, 10.0, 0.0, 0.5])
    delta, v = ctrl.compute_control(state, goal, obstacles)
    print(f"At (1,10), v=0.5: delta={delta:.4f} rad, v_cmd={v:.3f} m/s")
    assert v > 0, "Expected non-zero speed on clear path"

    # Robot very close to obstacle — should get low speed or deflection
    state2 = np.array([8.0, 10.0, 0.0, 1.0])
    delta2, v2 = ctrl.compute_control(state2, goal, obstacles)
    print(f"At (8,10), near obstacle: delta={delta2:.4f} rad, v_cmd={v2:.3f} m/s")

    # Robot blocked entirely (no valid trajectory) — should return (0,0)
    big_obs = [{"type": "static", "x": 1.05, "y": 10.0, "w": 3.0, "h": 3.0}]
    state3 = np.array([1.0, 10.0, 0.0, 0.0])
    delta3, v3 = ctrl.compute_control(state3, goal, big_obs)
    print(f"Fully blocked: delta={delta3:.4f}, v={v3:.3f}  (expect 0,0)")
    assert v3 == 0.0 and delta3 == 0.0, "Expected emergency stop"

    print("\n[PASS]  DWA OK")

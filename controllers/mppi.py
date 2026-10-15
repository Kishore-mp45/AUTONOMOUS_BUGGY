"""
controllers/mppi.py
====================
Model Predictive Path Integral (MPPI) controller.

Concept:
  MPPI is a sampling-based MPC approach rooted in stochastic optimal control.
  At each timestep:
    1.  Sample K perturbation sequences  epsilon_k ~ N(0, sigma^2 * I)
        around the current nominal control sequence U (shape: T x 2).
    2.  Roll out each perturbed trajectory under the robot's kinematic model.
    3.  Evaluate a cost for each trajectory:
            S_k = sum_t [ obstacle_cost(x_t) + path_cost(x_t) + control_cost(u_t) ]
                  + terminal_goal_cost(x_T)
    4.  Compute importance weights:
            w_k = exp( -(S_k - S_min) / lambda )   (numerically stable)
    5.  Update the nominal control sequence:
            U <- U + sum_k( w_k * epsilon_k ) / sum_k(w_k)
    6.  Apply the first action u_0 = [delta, a].
    7.  Shift the sequence one step and warm-start the next iteration.

References:
  Williams et al., "Model Predictive Path Integral Control: From Theory to
  Parallel Computation" (2018).
"""

import math
import numpy as np
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.controller_base import ControllerBase


class MPPI(ControllerBase):
    """
    Model Predictive Path Integral controller.
    Outputs (steering_angle, target_speed) compatible with the main loop.

    Parameters
    ----------
    K      : int    Number of trajectory samples.
    T      : int    Planning horizon (steps).
    sigma  : float  Noise standard deviation for perturbations.
    lam    : float  Temperature — controls selectivity of good trajectories.
    dt     : float  Simulation timestep (s).
    """
    name = "MPPI"

    def __init__(
        self,
        K:     int   = config.MPPI_K,
        T:     int   = config.MPPI_T,
        sigma: float = config.MPPI_SIGMA,
        lam:   float = config.MPPI_LAMBDA,
        dt:    float = config.DT,
    ):
        self.K     = int(K)
        self.T     = int(T)
        self.sigma = float(sigma)
        self.lam   = float(lam)
        self.dt    = float(dt)

        # Cost weights
        self.obs_cost  = config.MPPI_OBS_COST
        self.path_cost = config.MPPI_PATH_COST
        self.ctrl_cost = config.MPPI_CTRL_COST

        # Robot limits
        self.max_speed = config.MAX_SPEED
        self.max_accel = config.MAX_ACCEL
        self.max_steer = config.MAX_STEER
        self.L         = config.WHEELBASE
        self.r         = config.ROBOT_RADIUS

        # Warm-start: nominal control sequence shape (T, 2)  [delta, a]
        self._U: np.ndarray = np.zeros((self.T, 2), dtype=float)

    # ──────────────────────────────────────────────────────────────────────────
    # Main interface (compatible with main.py signature)
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset the warm-start nominal control sequence to zeros."""
        self._U = np.zeros((self.T, 2), dtype=float)

    def compute_control(
        self,
        state:     np.ndarray,
        path:      List[Tuple[float, float]] = None,
        goal:      Optional[Tuple[float, float]] = None,
        costmap    = None,
        obstacles: list = None,
    ) -> Tuple[float, float]:
        """
        Compute the best first control action via MPPI.

        Parameters
        ----------
        state     : np.ndarray [x, y, theta, v]
        path      : reference path as list of (x, y) waypoints
        goal      : (gx, gy) goal position (uses path[-1] if not given)
        costmap   : Costmap2D — used for precise obstacle cost lookup
        obstacles : list of Obstacle objects (fallback if no costmap)

        Returns
        -------
        (steer, speed) : steering angle (rad) and target speed (m/s)
        """
        x, y = float(state[0]), float(state[1])

        # Resolve goal
        if goal is None:
            goal = path[-1] if (path and len(path) > 0) else (x + 1.0, y)

        # Already at goal
        if math.hypot(goal[0] - x, goal[1] - y) < config.GOAL_TOLERANCE:
            return 0.0, 0.0

        # ── 0. Flush warm-start if nominal trajectory enters lethal space ──
        self._maybe_flush_warmstart(state, costmap)

        # ── 1. Sample K noise sequences ────────────────────────────────────
        # epsilon shape: (K, T, 2) — [delta_noise, a_noise]
        epsilon = np.random.normal(0.0, self.sigma, size=(self.K, self.T, 2))

        # Perturbed control sequences: shape (K, T, 2)
        U_k = self._U[np.newaxis, :, :] + epsilon

        # Clip to physical limits
        U_k[:, :, 0] = np.clip(U_k[:, :, 0], -self.max_steer, self.max_steer)
        U_k[:, :, 1] = np.clip(U_k[:, :, 1], -self.max_accel,  self.max_accel)

        # ── 2. Roll out all K trajectories ────────────────────────────────
        costs = self._batch_rollout(state, U_k, path, goal, costmap, obstacles)

        # ── 3. Importance weights (numerically stable) ────────────────────
        beta    = costs.min()
        weights = np.exp(-(costs - beta) / max(self.lam, 1e-6))
        weights /= (weights.sum() + 1e-10)

        # ── 4. Update nominal sequence ────────────────────────────────────
        delta_U = np.einsum("k,ktd->td", weights, epsilon)  # (T, 2)
        self._U = self._U + delta_U

        # Clip after update
        self._U[:, 0] = np.clip(self._U[:, 0], -self.max_steer, self.max_steer)
        self._U[:, 1] = np.clip(self._U[:, 1], -self.max_accel,  self.max_accel)

        # ── 5. Extract first action ───────────────────────────────────────
        steer_cmd = float(self._U[0, 0])
        a_cmd     = float(self._U[0, 1])

        # Shift warm-start sequence
        self._U[:-1] = self._U[1:]
        self._U[-1]  = np.zeros(2)

        # Convert acceleration to target speed for the main loop
        v_curr     = float(state[3])
        speed_cmd  = float(np.clip(v_curr + a_cmd * self.dt, 0.0, self.max_speed))

        return steer_cmd, speed_cmd

    def __repr__(self) -> str:
        return f"MPPI(K={self.K}, T={self.T}, sigma={self.sigma}, lambda={self.lam})"

    # ──────────────────────────────────────────────────────────────────────────
    # Warm-start safety flush
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_flush_warmstart(self, state: np.ndarray, costmap) -> None:
        """
        Probe the FULL nominal trajectory for costmap cost above a low threshold.
        If found, reset ONLY the steering channel to neutral (zero).

        IMPORTANT: We do NOT touch the speed channel.
        Braking hard (old behaviour) caused the robot to slow to near-zero,
        lose forward momentum, and then turn 180° (U-turn) to escape the
        obstacle. Keeping forward speed lets MPPI find left/right avoidance
        maneuvers while the robot continues progressing toward the goal.
        """
        if costmap is None:
            return

        # Fires only when the nominal path actually enters the lethal /
        # inscribed zone. A low threshold (8.0) made the flush trigger on the
        # soft inflation gradient ~2.8 m out, wiping the learned steering
        # manoeuvre every step and causing the robot to stall before the box.
        FLUSH_COST_THRESHOLD = config.INSCRIBED_COST

        x, y, th, v = float(state[0]), float(state[1]), float(state[2]), float(state[3])

        for t in range(self.T):
            delta = float(np.clip(self._U[t, 0], -self.max_steer, self.max_steer))
            a     = float(np.clip(self._U[t, 1], -self.max_accel,  self.max_accel))

            v  = float(np.clip(v + a * self.dt, 0.0, self.max_speed))
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            th += (v / self.L) * math.tan(delta) * self.dt

            fx = x + self.L * math.cos(th)
            fy = y + self.L * math.sin(th)

            if max(costmap.query(x, y), costmap.query(fx, fy)) >= FLUSH_COST_THRESHOLD:
                # Reset ONLY the steer to neutral — preserve forward speed prior
                self._U[:, 0] = 0.0
                # self._U[:, 1] intentionally left unchanged
                return

    # ──────────────────────────────────────────────────────────────────────────
    # Batch trajectory rollout (vectorised over K)
    # ──────────────────────────────────────────────────────────────────────────

    def _batch_rollout(
        self,
        state:     np.ndarray,
        U_k:       np.ndarray,           # shape (K, T, 2)
        path:      List[Tuple[float, float]],
        goal:      Tuple[float, float],
        costmap,
        obstacles: list,
    ) -> np.ndarray:
        """
        Roll out K trajectories simultaneously and return cost array (K,).
        """
        K, T, _ = U_k.shape
        path_arr = np.array(path, dtype=float) if (path and len(path) > 0) else np.zeros((1, 2))
        gx, gy = goal

        # Batch states: (K, 4)  [x, y, theta, v]
        states = np.tile(state, (K, 1)).astype(float)
        costs  = np.zeros(K, dtype=float)

        for t in range(T):
            delta_t = U_k[:, t, 0]   # (K,)  steering
            a_t     = U_k[:, t, 1]   # (K,)  acceleration

            x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

            # Kinematic update (Euler / bicycle model)
            v_new     = np.clip(v + a_t * self.dt, 0.0, self.max_speed)
            x_new     = x + v_new * np.cos(theta) * self.dt
            y_new     = y + v_new * np.sin(theta) * self.dt
            theta_new = theta + (v_new / self.L) * np.tan(delta_t) * self.dt
            theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

            states[:, 0] = x_new
            states[:, 1] = y_new
            states[:, 2] = theta_new
            states[:, 3] = v_new

            # ── Obstacle cost ──────────────────────────────────────────
            if costmap is not None:
                # Front bumper position
                fx = x_new + self.L * np.cos(theta_new)
                fy = y_new + self.L * np.sin(theta_new)

                rear_c  = costmap.query_batch(x_new, y_new)
                front_c = costmap.query_batch(fx, fy)
                max_c   = np.maximum(rear_c, front_c)

                # lethal_mask: inside inscribed/lethal zone — enormous penalty
                lethal_mask = max_c >= config.INSCRIBED_COST

                # Soft repulsion inside the inflation zone. Scaled per-step so
                # that crossing the inflation corridor beside the box costs
                # roughly as much as stopping does over the same horizon —
                # enough to prefer a smooth detour, never enough to make
                # freezing in place the optimal action.
                near_mask = (~lethal_mask) & (max_c > 1.5)

                costs += np.where(lethal_mask, self.obs_cost,
                         np.where(near_mask, 0.1 * max_c, 0.0))
            elif obstacles:
                obs_c = self._batch_obstacle_cost(x_new, y_new, obstacles)
                costs += obs_c

            # ── Path-deviation cost ────────────────────────────────────
            path_c = self._batch_path_cost(x_new, y_new, path_arr)
            costs += self.path_cost * path_c

            # ── Speed tracking cost (incentivise moving at max speed) ──
            # Weight 3.0: over the 4 s horizon, stopping costs ~540, while the
            # cheapest detour around the box costs ~300 (soft repulsion + path
            # deviation). This keeps "drive around" strictly cheaper than
            # "freeze in front of the obstacle", which caused stuck failures.
            costs += 3.0 * (self.max_speed - v_new) ** 2

            # ── Control effort cost (smooth curves) ───────────────────
            costs += self.ctrl_cost * (delta_t ** 2 + (a_t / max(self.max_accel, 1e-6)) ** 2)

            # ── Heading-toward-goal cost ───────────────────────────────
            # Penalises facing away from the goal. A 180° U-turn costs
            # 5 * pi * T ≈ 628 total; a 30° side-step costs ~52.
            # This prevents MPPI from choosing a U-turn to escape obstacles.
            dx_goal = gx - x_new
            dy_goal = gy - y_new
            goal_angle  = np.arctan2(dy_goal, dx_goal)
            heading_err = np.abs(
                np.arctan2(np.sin(theta_new - goal_angle),
                           np.cos(theta_new - goal_angle))
            )
            costs += 1.0 * heading_err

        # ── Terminal: strong goal distance cost ────────────────────────
        # Raised 5.0 → 30.0 so that backing up toward start is always
        # more expensive than curving around the obstacle.
        gx, gy = goal
        term_dist = np.hypot(states[:, 0] - gx, states[:, 1] - gy)
        costs += 50.0 * term_dist

        return costs

    # ──────────────────────────────────────────────────────────────────────────
    # Cost functions (vectorised)
    # ──────────────────────────────────────────────────────────────────────────

    def _batch_obstacle_cost(
        self,
        xs:        np.ndarray,
        ys:        np.ndarray,
        obstacles: list,
    ) -> np.ndarray:
        """
        Return obstacle cost for each sample (K,).
        Hard penalty for collision, soft gradient for near-miss.
        """
        cost = np.zeros(len(xs), dtype=float)
        for obs in obstacles:
            if hasattr(obs, 'active') and not obs.active:
                continue
            d = _batch_dist_to_obstacle(xs, ys, obs)
            collision = d < self.r
            near_miss = (d >= self.r) & (d < self.r * 3.0)
            cost += np.where(
                collision, self.obs_cost,
                np.where(near_miss,
                         self.obs_cost * (1.0 - (d - self.r) / (self.r * 2.0)),
                         0.0)
            )
        return cost

    def _batch_path_cost(
        self,
        xs:       np.ndarray,
        ys:       np.ndarray,
        path_arr: np.ndarray,
    ) -> np.ndarray:
        """
        Return distance to closest path point for each sample (K,).
        """
        if path_arr.shape[0] == 0:
            return np.zeros(len(xs))
        dx = xs[:, np.newaxis] - path_arr[np.newaxis, :, 0]
        dy = ys[:, np.newaxis] - path_arr[np.newaxis, :, 1]
        return np.sqrt(dx ** 2 + dy ** 2).min(axis=1)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _batch_dist_to_obstacle(
    xs: np.ndarray,
    ys: np.ndarray,
    obs,
) -> np.ndarray:
    """
    Vectorised distance from N points to one obstacle surface. Returns (N,).
    Accepts both Obstacle objects and legacy dict obstacles.
    """
    # Obstacle object with dist_to_point
    if hasattr(obs, 'dist_to_point'):
        return np.array([obs.dist_to_point(float(x), float(y))
                         for x, y in zip(xs, ys)], dtype=float)

    # Legacy dict format
    if isinstance(obs, dict):
        if obs.get("type") == "circle" or "radius" in obs:
            d = np.hypot(xs - obs["x"], ys - obs["y"])
            return np.maximum(0.0, d - obs["radius"])
        ox, oy = obs["x"], obs["y"]
        hw = obs.get("w", 1.0) / 2.0
        hh = obs.get("h", 1.0) / 2.0
        cx = np.clip(xs, ox - hw, ox + hw)
        cy = np.clip(ys, oy - hh, oy + hh)
        return np.hypot(xs - cx, ys - cy)

    return np.full(len(xs), float('inf'))


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MPPI self-test ===")
    np.random.seed(42)
    ctrl = MPPI(K=200, T=20)
    print(ctrl)

    path      = [(float(i), 10.0) for i in range(21)]
    state     = np.array([1.0, 10.0, 0.0, 0.0])

    for step in range(5):
        steer, speed = ctrl.compute_control(state, path, goal=(20.0, 10.0))
        v = float(np.clip(state[3] + (speed - state[3]), 0.0, config.MAX_SPEED))
        state = np.array([
            state[0] + v * math.cos(state[2]) * config.DT,
            state[1] + v * math.sin(state[2]) * config.DT,
            state[2] + (v / config.WHEELBASE) * math.tan(steer) * config.DT,
            v,
        ])

    print(f"After 5 steps: steer={steer:.4f} rad, speed={speed:.3f} m/s")
    assert abs(steer) <= config.MAX_STEER
    assert 0.0 <= speed <= config.MAX_SPEED

    ctrl.reset()
    assert np.all(ctrl._U == 0), "Expected zero after reset"

    steer2, speed2 = ctrl.compute_control(state, path, goal=(20.0, 10.0))
    assert abs(steer2) <= config.MAX_STEER
    assert 0.0 <= speed2 <= config.MAX_SPEED

    print("[PASS] MPPI self-test passed.")

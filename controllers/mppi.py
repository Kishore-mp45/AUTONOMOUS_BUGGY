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


class MPPI:
    """
    Model Predictive Path Integral controller.

    Parameters
    ----------
    K      : int    Number of trajectory samples.
    T      : int    Planning horizon (steps).
    sigma  : float  Noise standard deviation for perturbations.
    lam    : float  Temperature — controls selectivity of good trajectories.
    dt     : float  Simulation timestep (s).
    """

    def __init__(
        self,
        K:     int   = config.MPPI_K,
        T:     int   = config.MPPI_T,
        sigma: float = config.MPPI_SIGMA,
        lam:   float = config.MPPI_LAMBDA,
        dt:    float = config.DT,
    ):
        self.K     = K
        self.T     = T
        self.sigma = sigma
        self.lam   = lam
        self.dt    = dt

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
        self._U: np.ndarray = np.zeros((T, 2), dtype=float)

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
        Compute the best first control action via MPPI.

        Parameters
        ----------
        state     : np.ndarray [x, y, theta, v]
        path      : reference path as list of (x, y) waypoints
        obstacles : list of obstacle dicts

        Returns
        -------
        (delta, a) : steering angle (rad) and acceleration (m/s²)
        """
        # ── 1. Sample K noise sequences ──────────────────────────────
        # epsilon shape: (K, T, 2)  — independent perturbations for each sample
        epsilon = np.random.normal(0.0, self.sigma, size=(self.K, self.T, 2))

        # Perturbed control sequences: shape (K, T, 2)
        U_k = self._U[np.newaxis, :, :] + epsilon   # broadcast

        # Clip to physical limits
        U_k[:, :, 0] = np.clip(U_k[:, :, 0], -self.max_steer, self.max_steer)  # delta
        U_k[:, :, 1] = np.clip(U_k[:, :, 1], -self.max_accel,  self.max_accel)  # a

        # ── 2. Roll out all K trajectories ───────────────────────────
        costs = self._batch_rollout(state, U_k, path, obstacles)

        # ── 3. Compute importance weights (numerically stable) ───────
        beta   = costs.min()
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= (weights.sum() + 1e-10)              # normalise

        # ── 4. Update nominal sequence ────────────────────────────────
        # delta_U = sum_k( w_k * epsilon_k )
        delta_U = np.einsum("k,ktd->td", weights, epsilon)   # shape (T, 2)
        self._U = self._U + delta_U

        # Clip again after update
        self._U[:, 0] = np.clip(self._U[:, 0], -self.max_steer, self.max_steer)
        self._U[:, 1] = np.clip(self._U[:, 1], -self.max_accel,  self.max_accel)

        # ── 5. Extract first action and shift warm-start ─────────────
        delta, a = float(self._U[0, 0]), float(self._U[0, 1])

        # Shift sequence: move plan forward by one step
        self._U[:-1] = self._U[1:]
        self._U[-1]  = np.zeros(2)

        return delta, a

    def reset(self):
        """Reset the warm-start nominal control sequence to zeros."""
        self._U = np.zeros((self.T, 2), dtype=float)

    # ------------------------------------------------------------------
    # Batch trajectory rollout (vectorised over K)
    # ------------------------------------------------------------------

    def _batch_rollout(
        self,
        state:     np.ndarray,
        U_k:       np.ndarray,          # shape (K, T, 2)
        path:      List[Tuple[float, float]],
        obstacles: List[dict],
    ) -> np.ndarray:
        """
        Roll out K trajectories simultaneously and return cost array (K,).
        """
        K, T, _ = U_k.shape
        path_arr = np.array(path) if path else np.zeros((1, 2))

        # Initialise batch states: shape (K, 4)  [x, y, theta, v]
        states = np.tile(state, (K, 1)).astype(float)
        costs  = np.zeros(K, dtype=float)

        for t in range(T):
            delta_t = U_k[:, t, 0]   # (K,)
            a_t     = U_k[:, t, 1]   # (K,)

            # Kinematic update (Euler)
            x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

            x_new     = x     + v * np.cos(theta) * self.dt
            y_new     = y     + v * np.sin(theta) * self.dt
            theta_new = theta + v * np.tan(delta_t) / self.L * self.dt
            v_new     = np.clip(v + a_t * self.dt, 0.0, self.max_speed)

            # Wrap theta to [-pi, pi]
            theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

            states[:, 0] = x_new
            states[:, 1] = y_new
            states[:, 2] = theta_new
            states[:, 3] = v_new

            # ── Obstacle cost ──────────────────────────────────────
            obs_c = self._batch_obstacle_cost(x_new, y_new, obstacles)
            costs += obs_c

            # ── Path-deviation cost ────────────────────────────────
            path_c = self._batch_path_cost(x_new, y_new, path_arr)
            costs += self.path_cost * path_c

            # ── Target speed tracking cost ─────────────────────────
            # Incentivize moving at nominal speed along path
            speed_err = (self.max_speed - v_new) ** 2
            costs += 1.0 * speed_err

            # ── Control effort cost ────────────────────────────────
            ctrl_c = self.ctrl_cost * (delta_t ** 2 + (a_t / self.max_accel) ** 2)
            costs += ctrl_c

        # ── Terminal goal distance cost ────────────────────────────
        goal_pt = path_arr[-1]
        term_dist = np.sqrt((states[:, 0] - goal_pt[0])**2 + (states[:, 1] - goal_pt[1])**2)
        costs += 5.0 * term_dist

        return costs

    # ------------------------------------------------------------------
    # Cost functions (vectorised)
    # ------------------------------------------------------------------

    def _batch_obstacle_cost(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        obstacles: List[dict],
    ) -> np.ndarray:
        """
        Return obstacle cost for each sample.  (K,)
        Cost = MPPI_OBS_COST if within robot radius, else 0.
        """
        cost = np.zeros(len(xs), dtype=float)
        for obs in obstacles:
            d = _batch_dist_to_obstacle(xs, ys, obs)
            # Hard penalty for collision, soft for near-miss
            collision = d < self.r
            near_miss = (d >= self.r) & (d < self.r * 3)
            cost += np.where(collision, self.obs_cost,
                    np.where(near_miss, self.obs_cost * (1.0 - (d - self.r) / (self.r * 2)), 0.0))
        return cost

    def _batch_path_cost(
        self,
        xs:       np.ndarray,
        ys:       np.ndarray,
        path_arr: np.ndarray,
    ) -> np.ndarray:
        """
        Return distance to closest path point for each sample.  (K,)
        """
        if path_arr.shape[0] == 0:
            return np.zeros(len(xs))

        # xs/ys: (K,), path_arr: (N, 2)
        # Expand: (K, 1) - (1, N) = (K, N)
        dx = xs[:, np.newaxis] - path_arr[np.newaxis, :, 0]
        dy = ys[:, np.newaxis] - path_arr[np.newaxis, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)           # (K, N)
        return dists.min(axis=1)                      # (K,)

    def __repr__(self) -> str:
        return (
            f"MPPI(K={self.K}, T={self.T}, "
            f"sigma={self.sigma}, lambda={self.lam})"
        )


# ── Module-level helpers ──────────────────────────────────────────────────

def _batch_dist_to_obstacle(
    xs: np.ndarray,
    ys: np.ndarray,
    obs: dict,
) -> np.ndarray:
    """
    Vectorised distance from N points to one obstacle surface.
    Returns array of shape (N,).
    """
    if obs.get("type") == "circle" or "radius" in obs:
        d = np.sqrt((xs - obs["x"]) ** 2 + (ys - obs["y"]) ** 2)
        return np.maximum(0.0, d - obs["radius"])

    # Axis-aligned rect
    ox, oy = obs["x"], obs["y"]
    hw = obs.get("w", 1.0) / 2.0
    hh = obs.get("h", 1.0) / 2.0
    cx = np.clip(xs, ox - hw, ox + hw)
    cy = np.clip(ys, oy - hh, oy + hh)
    return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)


def _dist_to_obstacle_scalar(rx: float, ry: float, obs: dict) -> float:
    """Scalar distance from (rx,ry) to obstacle surface."""
    import math
    if obs.get("type") == "circle" or "radius" in obs:
        return max(0.0, math.hypot(rx - obs["x"], ry - obs["y"]) - obs["radius"])
    ox, oy = obs["x"], obs["y"]
    hw = obs.get("w", 1.0) / 2.0
    hh = obs.get("h", 1.0) / 2.0
    cx = float(np.clip(rx, ox - hw, ox + hw))
    cy = float(np.clip(ry, oy - hh, oy + hh))
    return math.hypot(rx - cx, ry - cy)


# ── Quick self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MPPI self-test ===")
    np.random.seed(42)
    ctrl = MPPI(K=200, T=10)   # Smaller K/T for fast test
    print(ctrl)

    path      = [(float(i), 10.0) for i in range(21)]
    obstacles = [{"type": "static", "x": 10, "y": 10, "w": 1.5, "h": 1.5}]
    state     = np.array([1.0, 10.0, 0.0, 0.0])

    # Run several iterations (warm-up)
    for step in range(5):
        delta, a = ctrl.compute_control(state, path, obstacles)

    print(f"After 5 steps: delta={delta:.4f} rad, a={a:.4f} m/s2")
    assert isinstance(delta, float) and isinstance(a, float), "Expected floats"
    assert -config.MAX_STEER <= delta <= config.MAX_STEER, "Steer out of bounds"
    assert -config.MAX_ACCEL <= a     <= config.MAX_ACCEL, "Accel out of bounds"

    # Reset and check warm-start cleared
    ctrl.reset()
    assert np.all(ctrl._U == 0), "Expected zero after reset"

    # Test with no obstacles — just verify outputs are within physical limits
    delta2, a2 = ctrl.compute_control(state, path, [])
    print(f"No obstacles: delta={delta2:.4f}, a={a2:.4f}")
    assert -config.MAX_STEER <= delta2 <= config.MAX_STEER, "Steer out of bounds"
    assert -config.MAX_ACCEL <= a2     <= config.MAX_ACCEL, "Accel out of bounds"

    print("\n[PASS]  MPPI OK")

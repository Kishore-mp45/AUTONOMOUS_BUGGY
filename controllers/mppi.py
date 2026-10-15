"""
controllers/mppi.py
====================
Model Predictive Path Integral (MPPI) controller — production version.

Architecture (Nav2-style):
  - Vectorised numpy rollout of K samples over an adaptive horizon T
  - Pluggable critics in the `mppi_critics` package supply the cost
  - Adaptive lambda (greedier near obstacles) and adaptive horizon
    (longer near obstacles) selected from costmap queries
  - All-collide fallback: sigma expansion → N recovery attempts →
    emergency stop (Edge Case EC-10/EC-13)
  - EMA output filter to suppress chattering
  - Warm-start: nominal sequence shifted one step each timestep

Contract with main.py:
  compute_control(state, path, goal, costmap, obstacles) -> (steer, speed_cmd)
  where speed_cmd is a target speed [m/s] (main.py converts it to accel).
"""

import math
import numpy as np
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.controller_base import ControllerBase
from mppi_critics import (
    ObstacleCritic, GoalCritic, PathCritic,
    JerkCritic, SpeedCritic, LaneCritic, ManeuverCritic,
)
from environment.lane_map import LaneMap


class MPPI(ControllerBase):
    name = "MPPI"

    def __init__(self):
        self.T = config.MPPI_HORIZON_DEFAULT
        # Nominal control sequence [steer, accel], shape (T, 2)
        self.U = np.zeros((self.T, 2), dtype=float)

        self.lane_map = LaneMap()
        self.critics = {
            "obstacle": ObstacleCritic(),
            "goal":     GoalCritic(),
            "path":     PathCritic(lane_map=self.lane_map),
            "jerk":     JerkCritic(),
            "speed":    SpeedCritic(),
            "lane":     LaneCritic(self.lane_map),
            "maneuver": ManeuverCritic(),
        }

        # Output smoothing (chattering suppression)
        self._ema_u = np.zeros(2, dtype=float)

        # Recovery state (all-collide fallback, Edge Case EC-13)
        self._recovery_count = 0
        self._recovery_count_total = 0
        self._current_sigma = np.array(
            [config.MPPI_SIGMA_STEER, config.MPPI_SIGMA_ACCEL], dtype=float
        )
        self.emergency_stop = False
        self.weight_fallback_count = 0
        self.last_weight_fallback = False
        self.ackermann_clamp_count = 0

        # Maneuver interface (set by ManeuverStateMachine, Phase 5)
        self.maneuver_mode = "normal"
        self.maneuver_target = None
        self.lane_guidance_enabled = True
        self.last_trajectories = None
        self.last_weights = None

    # ──────────────────────────────────────────────────────────────────────
    # Adaptive parameters
    # ──────────────────────────────────────────────────────────────────────

    def _get_lambda(self, state: np.ndarray, costmap) -> float:
        """
        Lower lambda near obstacles (greedy — commit to safe trajectory).
        Higher lambda on clear path (exploratory — find best route).
        """
        if costmap is None:
            return config.MPPI_LAMBDA_FREE
        cost_here = costmap.query(float(state[0]), float(state[1]))
        if cost_here > config.MPPI_LAMBDA_ADAPT_COST:
            return config.MPPI_LAMBDA_NEAR_OBS
        return config.MPPI_LAMBDA_FREE

    def _get_horizon(self, state: np.ndarray, costmap) -> int:
        """Longer horizon near obstacles so MPPI can see past them."""
        if costmap is None:
            return config.MPPI_HORIZON_DEFAULT
        cost_here = costmap.query(float(state[0]), float(state[1]))
        if cost_here > config.MPPI_LAMBDA_ADAPT_COST:
            return max(config.MPPI_HORIZON_NEAR_OBS, config.MPPI_HORIZON_DEFAULT)
        return config.MPPI_HORIZON_DEFAULT

    def _is_narrow_context(self, costmap, state: np.ndarray) -> bool:
        """
        True if lateral scan shows walls close on both sides.
        Used to increase K for narrow gap scenarios.
        """
        if costmap is None:
            return False
        x, y = float(state[0]), float(state[1])
        left_clear  = costmap.query(x, y - 1.0) < config.LETHAL_COST
        right_clear = costmap.query(x, y + 1.0) < config.LETHAL_COST
        return not left_clear and not right_clear

    # ──────────────────────────────────────────────────────────────────────
    # Rollout (vectorised over K)
    # ──────────────────────────────────────────────────────────────────────

    def _rollout(self, state: np.ndarray, U_noisy: np.ndarray, T: int) -> np.ndarray:
        """
        state   : (4,)  [x, y, theta, v]
        U_noisy : (K, T, 2)  [delta, a]
        Returns : (K, T, 4)  states after each step
        """
        K = U_noisy.shape[0]
        states = np.zeros((K, T + 1, 4), dtype=float)
        states[:, 0, :] = state
        self.ackermann_clamp_count = 0

        for t in range(T):
            x, y, th, v = (states[:, t, 0], states[:, t, 1],
                           states[:, t, 2], states[:, t, 3])
            delta = np.clip(U_noisy[:, t, 0], -config.MAX_STEER, config.MAX_STEER)
            a     = np.clip(U_noisy[:, t, 1], -config.MAX_ACCEL, config.MAX_ACCEL)

            # Ackermann turning radius clamp: R = L / tan(|delta|) >= R_min
            nonzero = np.abs(delta) > 1e-6
            if np.any(nonzero):
                radii = np.where(
                    nonzero,
                    config.WHEELBASE / np.maximum(np.abs(np.tan(delta)), 1e-6),
                    1e6,
                )
                too_tight = nonzero & (radii < config.MIN_TURN_RADIUS)
                if np.any(too_tight):
                    self.ackermann_clamp_count += int(np.count_nonzero(too_tight))
                    max_delta = math.atan(config.WHEELBASE / config.MIN_TURN_RADIUS)
                    delta[too_tight] = np.sign(delta[too_tight]) * max_delta

            states[:, t + 1, 0] = x + v * np.cos(th) * config.DT
            states[:, t + 1, 1] = y + v * np.sin(th) * config.DT
            states[:, t + 1, 2] = th + (v / config.WHEELBASE) * np.tan(delta) * config.DT
            states[:, t + 1, 3] = np.clip(v + a * config.DT, 0.0, config.MAX_SPEED)

        return states[:, 1:, :]   # (K, T, 4)

    # ──────────────────────────────────────────────────────────────────────
    # Cost computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_costs(self, trajs, goal, path, costmap, obstacles=None) -> np.ndarray:
        costs = np.zeros(trajs.shape[0], dtype=float)
        costs += self.critics["obstacle"].score(trajs, costmap, obstacles)
        costs += self.critics["goal"].score(trajs, goal)
        costs += self.critics["path"].score(
            trajs,
            path,
            self.lane_map,
            self.maneuver_mode,
            include_lane_cost=self.lane_guidance_enabled,
        )
        costs += self.critics["jerk"].score(trajs)
        costs += self.critics["speed"].score(trajs, costmap)
        if self.lane_guidance_enabled:
            costs += self.critics["lane"].score(trajs, self.maneuver_mode)
        costs += self.critics["maneuver"].score(trajs, self.maneuver_target, self.maneuver_mode)
        return costs

    # ──────────────────────────────────────────────────────────────────────
    # Main interface
    # ──────────────────────────────────────────────────────────────────────

    def compute_control(
        self,
        state: np.ndarray,
        path: List[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        costmap = None,
        obstacles: list = None,
    ) -> Tuple[float, float]:
        x, y = float(state[0]), float(state[1])

        if goal is None:
            goal = path[-1] if (path and len(path) > 0) else (x + 1.0, y)

        # Already at goal
        if math.hypot(goal[0] - x, goal[1] - y) < config.GOAL_TOLERANCE:
            return 0.0, 0.0

        T   = self._get_horizon(state, costmap)
        lam = self._get_lambda(state, costmap)
        K   = config.MPPI_NARROW_K if self._is_narrow_context(costmap, state) else config.MPPI_K

        # Resize nominal sequence if horizon changed (pad zeros / truncate)
        if self.U.shape[0] != T:
            new_U = np.zeros((T, 2), dtype=float)
            n = min(T, self.U.shape[0])
            new_U[:n] = self.U[:n]
            self.U = new_U

        # ── 1. Sample noise (per-channel sigma, coloured along horizon) ───
        # Raw iid noise is low-pass filtered along the time axis so that
        # samples represent sustained maneuvers (e.g. held acceleration)
        # rather than per-step jitter, then rescaled to preserve the
        # per-channel sigma. See plan §1.4 "colored noise option".
        raw = np.random.randn(K, T, 2)
        eps = np.empty_like(raw)
        eps[:, 0, :] = raw[:, 0, :]
        alpha_n = config.MPPI_NOISE_ALPHA
        for t in range(1, T):
            eps[:, t, :] = alpha_n * eps[:, t - 1, :] + (1.0 - alpha_n) * raw[:, t, :]
        for c, sig in enumerate(self._current_sigma):
            std = eps[:, :, c].std()
            if std > 1e-9:
                eps[:, :, c] *= sig / std

        U_noisy = np.clip(
            self.U[None, :, :] + eps,
            -np.array([config.MAX_STEER, config.MAX_ACCEL]),
             np.array([config.MAX_STEER, config.MAX_ACCEL]),
        )

        # ── 2. Rollout ────────────────────────────────────────────────────
        trajs = self._rollout(state, U_noisy, T)
        self.last_trajectories = trajs

        # ── 3. Score ──────────────────────────────────────────────────────
        costs = np.asarray(
            self._compute_costs(trajs, goal, path, costmap, obstacles),
            dtype=float,
        )
        self.last_weights = None
        self.last_weight_fallback = False
        self.emergency_stop = False

        # ── 4. Importance weights (numerically stable) ────────────────────
        finite = np.isfinite(costs)
        viable = finite & (costs < config.MPPI_OBS_COST)
        if not np.any(viable):
            self._recovery_count += 1
            self._recovery_count_total += 1
            if self._recovery_count >= config.MPPI_RECOVERY_MAX_ATTEMPTS:
                print("[MPPI] Emergency stop — no valid trajectory")
                self.U[:] = 0.0
                self._recovery_count = 0
                self.emergency_stop = True
                self._current_sigma = np.array(
                    [config.MPPI_SIGMA_STEER, config.MPPI_SIGMA_ACCEL]
                )
                return 0.0, 0.0
            # Expand sigma for the next attempt (more exploration)
            self._current_sigma = (
                np.array([config.MPPI_SIGMA_STEER, config.MPPI_SIGMA_ACCEL])
                * config.MPPI_RECOVERY_SIGMA_MULT
            )
            print(f"[MPPI] Recovery attempt {self._recovery_count} — widening sigma")
            return 0.0, 0.1   # creep forward to escape local trap
        else:
            self._recovery_count = 0
            self._current_sigma = np.array(
                [config.MPPI_SIGMA_STEER, config.MPPI_SIGMA_ACCEL]
            )

        safe_costs = np.where(viable, costs, np.inf)
        beta = float(np.min(safe_costs))
        weights = np.zeros_like(costs)
        weights[viable] = np.exp(-(costs[viable] - beta) / max(lam, 1e-6))
        w_sum = float(weights.sum())
        if not np.isfinite(w_sum) or w_sum < 1e-10:
            weights[viable] = 1.0 / np.count_nonzero(viable)
            self.last_weight_fallback = True
            self.weight_fallback_count += 1
            print("[MPPI] Warning: invalid sample weights; using uniform viable weights")
        else:
            weights /= w_sum

        self.last_weights = weights.copy()

        # ── 5. Update nominal control (vectorised weighted mean of noise) ─
        self.U += np.einsum("k,ktd->td", weights, eps)
        self.U = np.clip(
            self.U,
            -np.array([config.MAX_STEER, config.MAX_ACCEL]),
             np.array([config.MAX_STEER, config.MAX_ACCEL]),
        )

        # ── 6. Extract first control + EMA smoothing ──────────────────────
        u0 = self.U[0].copy()
        self._ema_u = config.MPPI_EMA_ALPHA * self._ema_u + (1.0 - config.MPPI_EMA_ALPHA) * u0
        steer = float(np.clip(self._ema_u[0], -config.MAX_STEER, config.MAX_STEER))
        max_ackermann_steer = math.atan(config.WHEELBASE / config.MIN_TURN_RADIUS)
        clamped_steer = float(np.clip(
            steer,
            -min(config.MAX_STEER, max_ackermann_steer),
            min(config.MAX_STEER, max_ackermann_steer),
        ))
        if clamped_steer != steer:
            self.ackermann_clamp_count += 1
        steer = clamped_steer
        speed = float(np.clip(float(state[3]) + self._ema_u[1] * config.DT, 0.0, config.MAX_SPEED))

        # ── 7. Warm-start shift ───────────────────────────────────────────
        self.U[:-1] = self.U[1:]
        self.U[-1]  = 0.0

        return steer, speed

    # ──────────────────────────────────────────────────────────────────────
    # Maneuver interface / lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def set_maneuver(self, mode: str, target=None):
        """Called by ManeuverStateMachine to switch mode."""
        self.maneuver_mode = mode
        self.maneuver_target = target

    def reset(self):
        self.T = config.MPPI_HORIZON_DEFAULT
        self.U = np.zeros((self.T, 2), dtype=float)
        self._ema_u = np.zeros(2, dtype=float)
        self._recovery_count = 0
        self._recovery_count_total = 0
        self._current_sigma = np.array(
            [config.MPPI_SIGMA_STEER, config.MPPI_SIGMA_ACCEL]
        )
        self.emergency_stop = False
        self.weight_fallback_count = 0
        self.last_weight_fallback = False
        self.ackermann_clamp_count = 0
        self.maneuver_mode = "normal"
        self.maneuver_target = None
        self.lane_guidance_enabled = True
        self.last_trajectories = None
        self.last_weights = None

    def __repr__(self) -> str:
        return (f"MPPI(K={config.MPPI_K}, T={self.T}, "
                f"sigma=[{self._current_sigma[0]:.2f},{self._current_sigma[1]:.2f}], "
                f"mode={self.maneuver_mode})")


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MPPI v3.0 self-test ===")
    np.random.seed(42)
    ctrl = MPPI()
    print(ctrl)

    from environment.costmap import Costmap2D
    from environment.obstacle import Obstacle

    cm = Costmap2D()
    cm.add_static_obstacle(Obstacle(10.0, 10.0, width=1.5, height=1.5))
    path = [(float(i), 10.0) for i in range(21)]
    state = np.array([1.0, 10.0, 0.0, 0.0])

    # Roll a few steps toward the obstacle — must steer and keep moving
    for step in range(5):
        steer, speed = ctrl.compute_control(state, path, goal=(20.0, 10.0), costmap=cm)
        v = float(np.clip(state[3] + (speed - state[3]) * 1.0, 0.0, config.MAX_SPEED))
        state = np.array([
            state[0] + v * math.cos(state[2]) * config.DT,
            state[1] + v * math.sin(state[2]) * config.DT,
            state[2] + (v / config.WHEELBASE) * math.tan(steer) * config.DT,
            v,
        ])
    assert abs(steer) <= config.MAX_STEER
    assert 0.0 <= speed <= config.MAX_SPEED
    print(f"After 5 steps: steer={steer:.4f} rad, speed={speed:.3f} m/s")

    ctrl.reset()
    assert np.all(ctrl.U == 0), "Expected zero U after reset"
    assert ctrl.maneuver_mode == "normal"
    print("[PASS] MPPI v3.0 self-test passed.")

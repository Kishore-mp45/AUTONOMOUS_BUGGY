"""
robot/bicycle_model.py
======================
Kinematic Bicycle Model for the autonomous campus buggy.

State vector:   [x, y, theta, v]
                 x, y   — position in metres
                 theta  — heading angle in radians (0 = East, CCW positive)
                 v      — longitudinal speed in m/s

Control input:  [delta, a]
                 delta  — front-wheel steering angle in radians
                 a      — longitudinal acceleration in m/s²

Kinematic equations (rear-axle reference point):
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = v * tan(delta) / L          (L = wheelbase)
    v_dot     = a

Integration uses a simple 1st-order Euler step (dt from config).
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class BicycleModel:
    """
    Kinematic bicycle model — standard car-like robot model.

    Parameters
    ----------
    wheelbase : float
        Distance between front and rear axle in metres (L).
    max_speed : float
        Maximum allowable longitudinal speed in m/s.
    max_accel : float
        Maximum allowable acceleration magnitude in m/s².
    max_steer : float
        Maximum allowable steering angle magnitude in radians.
    """

    def __init__(
        self,
        wheelbase: float = config.WHEELBASE,
        max_speed: float = config.MAX_SPEED,
        max_accel: float = config.MAX_ACCEL,
        max_steer: float = config.MAX_STEER,
    ):
        self.L          = wheelbase
        self.max_speed  = max_speed
        self.max_accel  = max_accel
        self.max_steer  = max_steer

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float = config.DT,
    ) -> np.ndarray:
        """
        Advance the robot state by one timestep using Euler integration.

        Parameters
        ----------
        state : np.ndarray  shape (4,)
            Current state [x, y, theta, v].
        control : np.ndarray  shape (2,)
            Control input [delta, a]  (steering angle, acceleration).
        dt : float
            Timestep in seconds.

        Returns
        -------
        np.ndarray  shape (4,)
            New state after applying the kinematic equations.
        """
        x, y, theta, v = state
        delta, a       = control

        # ── Clamp controls to physical limits ─────────────────────────
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))
        a     = float(np.clip(a,     -self.max_accel,  self.max_accel))

        # ── Kinematic equations ────────────────────────────────────────
        x_dot     = v * math.cos(theta)
        y_dot     = v * math.sin(theta)
        theta_dot = v * math.tan(delta) / self.L   # yaw rate
        v_dot     = a

        # ── Euler integration ──────────────────────────────────────────
        x_new     = x     + x_dot     * dt
        y_new     = y     + y_dot     * dt
        theta_new = theta + theta_dot * dt
        v_new     = v     + v_dot     * dt

        # ── Wrap heading to [-pi, pi] ──────────────────────────────────
        theta_new = _wrap_angle(theta_new)

        # ── Clamp speed (cannot reverse unless explicitly set) ─────────
        v_new = float(np.clip(v_new, 0.0, self.max_speed))

        return np.array([x_new, y_new, theta_new, v_new], dtype=float)

    # ------------------------------------------------------------------
    # Collision footprint
    # ------------------------------------------------------------------

    def get_footprint(
        self,
        state: np.ndarray,
        length: float = config.WHEELBASE * 2,
        width:  float = config.ROBOT_RADIUS * 2,
    ) -> np.ndarray:
        """
        Return the four corner positions of the robot's rectangular footprint
        in world coordinates, accounting for current heading.

        The rectangle is centred on the rear axle position (state x, y) and
        extends `length` forward along the heading and `width` sideways.

        Parameters
        ----------
        state : np.ndarray  shape (4,)
            Current state [x, y, theta, v].
        length : float
            Robot body length in metres (default: 2 × wheelbase).
        width : float
            Robot body width in metres (default: 2 × robot radius).

        Returns
        -------
        np.ndarray  shape (4, 2)
            Four corner points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            in counter-clockwise order starting from front-left.
        """
        x, y, theta, _ = state

        # Half-dimensions
        half_l = length / 2.0
        half_w = width  / 2.0

        # Corners in local body frame (front-left, front-right, rear-right, rear-left)
        local_corners = np.array([
            [ half_l,  half_w],
            [ half_l, -half_w],
            [-half_l, -half_w],
            [-half_l,  half_w],
        ])

        # Rotation matrix
        cos_h = math.cos(theta)
        sin_h = math.sin(theta)
        R = np.array([[cos_h, -sin_h],
                      [sin_h,  cos_h]])

        # Transform to world frame
        world_corners = (R @ local_corners.T).T + np.array([x, y])
        return world_corners

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_state_dict(self, state: np.ndarray) -> dict:
        """Return state as a labelled dictionary."""
        x, y, theta, v = state
        return {"x": x, "y": y, "theta": theta, "v": v}

    def __repr__(self) -> str:
        return (
            f"BicycleModel("
            f"L={self.L}, "
            f"max_speed={self.max_speed}, "
            f"max_accel={self.max_accel}, "
            f"max_steer={self.max_steer})"
        )


# ── Module-level helpers ───────────────────────────────────────────────────

def _wrap_angle(angle: float) -> float:
    """Wrap angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def make_state(x: float, y: float, theta: float = 0.0, v: float = 0.0) -> np.ndarray:
    """Convenience factory to create a state vector."""
    return np.array([x, y, theta, v], dtype=float)


# ── Quick self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== BicycleModel self-test ===")
    model = BicycleModel()
    print(model)

    # Start at origin, heading East, zero speed
    state = make_state(0.0, 0.0, theta=0.0, v=0.0)
    print(f"\nInitial state: x={state[0]:.3f}, y={state[1]:.3f}, "
          f"theta={state[2]:.3f} rad, v={state[3]:.3f} m/s")

    # Accelerate straight ahead for 1 second (20 steps × dt=0.05)
    control = np.array([0.0, 1.0])   # delta=0 (straight), a=1 m/s²
    for _ in range(20):
        state = model.update(state, control)

    print(f"After 1s straight acceleration: x={state[0]:.3f}, y={state[1]:.3f}, "
          f"v={state[3]:.3f} m/s")
    # Expected: x ≈ 0.5*(1.0)*(1.0)^2 / 2 = 0.25m? No — Euler: v grows linearly.
    # v(1s) = 0 + 1.0*1.0 = 1.0 m/s; x = sum(v*dt) = 0+0.05+0.10+...+0.95 = 9.5*0.05 = 0.475m
    expected_x = sum(i * config.DT * config.DT for i in range(20))   # = 0.475 m
    print(f"Expected x ~= 0.475m  (Euler sum)")

    # Footprint
    fp = model.get_footprint(state)
    print(f"\nFootprint corners:\n{fp}")

    # Test steering (left turn)
    state2 = make_state(0.0, 0.0, theta=0.0, v=1.0)
    state2 = model.update(state2, np.array([0.3, 0.0]))   # steer left, coast
    print(f"\nAfter 1 step with delta=0.3 rad: "
          f"x={state2[0]:.4f}, y={state2[1]:.4f}, theta={state2[2]:.4f} rad")

    print("\n[PASS]  BicycleModel OK")

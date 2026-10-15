"""
health/monitor.py
=================
HealthMonitor: Checks vehicle and controller health at every simulation timestep.
Detects:
  - Collisions via Costmap2D lethal cost
  - Stuck states (speed < threshold for > timeout)
  - Steering oscillations (rapid sign flips in steering commands)
"""

from typing import Tuple, List, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class HealthMonitor:
    """
    Monitors controller health every timestep.
    Returns status: "ok" | "stuck" | "oscillating" | "collision"
    """

    def __init__(
        self,
        stuck_timeout: float = config.STUCK_TIMEOUT,
        stuck_speed_thresh: float = config.STUCK_SPEED_THRESH,
        oscillation_window: int = config.OSCILLATION_WINDOW,
        oscillation_thresh: int = config.OSCILLATION_THRESH,
        dt: float = config.DT,
        lethal_cost: float = config.LETHAL_COST,
    ):
        self.stuck_timeout      = float(stuck_timeout)
        self.stuck_speed_thresh = float(stuck_speed_thresh)
        self.oscillation_window = int(oscillation_window)
        self.oscillation_thresh = int(oscillation_thresh)
        self.dt                 = float(dt)
        self.lethal_cost        = float(lethal_cost)

        self.speed_history:  List[float] = []
        self.steer_history:  List[float] = []
        self.collision_flag: bool        = False

    def check(self, state: np.ndarray, control: Tuple[float, float], costmap) -> Tuple[str, str]:
        """
        Check health conditions at current timestep.

        Parameters
        ----------
        state : [x, y, theta, v]
        control : (steer, speed)
        costmap : Costmap2D instance

        Returns
        -------
        (status, message) : ("ok" | "stuck" | "oscillating" | "collision", str)
        """
        x, y, theta, v = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        steer, speed   = float(control[0]), float(control[1])

        # ── 1. Collision check via costmap ────────────────────────────
        if costmap is not None and costmap.query(x, y) >= self.lethal_cost:
            self.collision_flag = True
            return "collision", f"Collision at ({x:.2f}, {y:.2f})"

        # ── 2. Stuck check (sustained low speed) ──────────────────────
        self.speed_history.append(abs(v))
        max_speed_window = int(self.stuck_timeout / self.dt)
        if len(self.speed_history) > max_speed_window:
            self.speed_history.pop(0)
            if max(self.speed_history) < self.stuck_speed_thresh:
                return "stuck", f"Robot not moving for >{self.stuck_timeout}s"

        # ── 3. Oscillation check (steering sign flips) ────────────────
        self.steer_history.append(steer)
        if len(self.steer_history) > self.oscillation_window:
            self.steer_history.pop(0)
            sign_flips = sum(
                1 for i in range(1, len(self.steer_history))
                if (self.steer_history[i] * self.steer_history[i - 1] < -1e-6)
            )
            if sign_flips >= self.oscillation_thresh:
                return "oscillating", f"{sign_flips} steer sign flips in {self.oscillation_window} steps"

        return "ok", ""

    def reset(self):
        """Reset history between simulation runs."""
        self.speed_history  = []
        self.steer_history  = []
        self.collision_flag = False


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== HealthMonitor Self-Test ===")
    from environment.costmap import Costmap2D
    from environment.obstacle import Obstacle

    hm = HealthMonitor()
    cm = Costmap2D()
    cm.add_static_obstacle(Obstacle(10.0, 10.0, width=1.5, height=1.5))

    # Test 1: Normal moving state
    state = np.array([2.0, 10.0, 0.0, 1.2])
    status, msg = hm.check(state, (0.01, 1.2), cm)
    assert status == "ok"

    # Test 2: Collision detection
    col_state = np.array([10.0, 10.0, 0.0, 1.0])
    status, msg = hm.check(col_state, (0.0, 1.0), cm)
    print(f"Collision test: status={status}, msg={msg}")
    assert status == "collision"

    # Test 3: Stuck detection
    hm.reset()
    stuck_state = np.array([5.0, 10.0, 0.0, 0.01])
    for _ in range(int(config.STUCK_TIMEOUT / config.DT) + 5):
        status, msg = hm.check(stuck_state, (0.0, 0.0), cm)
    print(f"Stuck test: status={status}, msg={msg}")
    assert status == "stuck"

    # Test 4: Oscillation detection
    hm.reset()
    osc_state = np.array([5.0, 10.0, 0.0, 1.0])
    for i in range(12):
        steer_cmd = 0.4 if i % 2 == 0 else -0.4
        status, msg = hm.check(osc_state, (steer_cmd, 1.0), cm)
    print(f"Oscillation test: status={status}, msg={msg}")
    assert status == "oscillating"

    print("[PASS] HealthMonitor verified!")

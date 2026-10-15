"""
health/monitor.py
=================
HealthMonitor: Checks vehicle and controller health at every simulation timestep.
Detects:
  - Collisions via Costmap2D lethal cost
  - Stuck states (speed < threshold for > timeout)
  - MPPI recovery escalation, lane violations, and road-boundary exits

(The RPP-specific steering-oscillation check was removed with the RPP controller.)
"""

from typing import Tuple, List, Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class HealthMonitor:
    """
    Monitors controller health every timestep.
    Returns status: "ok" | "stuck" | "collision" for the primary health check.
    """

    def __init__(
        self,
        stuck_timeout: float = config.STUCK_TIMEOUT,
        stuck_speed_thresh: float = config.STUCK_SPEED_THRESH,
        dt: float = config.DT,
        lethal_cost: float = config.LETHAL_COST,
    ):
        self.stuck_timeout      = float(stuck_timeout)
        self.stuck_speed_thresh = float(stuck_speed_thresh)
        self.dt                 = float(dt)
        self.lethal_cost        = float(lethal_cost)

        self.speed_history:  List[float] = []
        self.collision_flag: bool        = False
        self.lane_violation_count: int    = 0
        self.boundary_exit: bool          = False

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
        (status, message) : ("ok" | "stuck" | "collision", str)
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

        return "ok", ""

    def check_mppi_recovery(self, recovery_count: int) -> Tuple[str, str]:
        """Report whether MPPI is recovering or has exhausted its attempts."""
        if recovery_count >= config.MPPI_RECOVERY_MAX_ATTEMPTS:
            return "mppi_emergency_stop", "MPPI exhausted recovery attempts"
        if recovery_count > 0:
            return "mppi_recovering", f"Recovery attempt {recovery_count}"
        return "ok", ""

    def check_lane_discipline(
        self,
        state: np.ndarray,
        maneuver_mode: str,
        lane_map,
    ) -> Tuple[str, str]:
        """Count and report a wrong-lane timestep for the active maneuver."""
        x, y = float(state[0]), float(state[1])
        if lane_map.wrong_lane_violation(x, y, maneuver_mode):
            self.lane_violation_count += 1
            return "lane_violation", f"Wrong lane at ({x:.1f},{y:.1f})"
        return "ok", ""

    def check_boundary(self, state: np.ndarray) -> Tuple[str, str]:
        """Report and remember when the vehicle leaves the road bounds."""
        y = float(state[1])
        if y <= config.ROAD_Y_MIN or y >= config.ROAD_Y_MAX:
            self.boundary_exit = True
            return "boundary_exit", f"Exited road at y={y:.2f}"
        return "ok", ""

    def reset(self):
        """Reset history between simulation runs."""
        self.speed_history        = []
        self.collision_flag       = False
        self.lane_violation_count = 0
        self.boundary_exit        = False


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

    print("[PASS] HealthMonitor verified!")

"""
maneuvers/maneuver_state_machine.py
===================================
Decides which maneuver mode MPPI should use at each timestep.
Reads the global path and current vehicle state, and drives the MPPI
controller via mppi.set_maneuver(mode, target).

States:
  NORMAL       — following path in left lane
  PRE_RIGHT    — moving to right lane before a right turn
  IN_LEFT      — executing left turn arc
  IN_RIGHT     — executing right turn arc
  RETURN_LEFT  — returning to left lane after right turn / overtake / U-turn
  IN_UTURN     — executing 180-degree arc
  REVERSING    — moving backward (dead-end escape)
  OVERTAKING   — temporarily in right lane to pass a left-lane obstacle

Transitions are triggered by:
  - Upcoming turn waypoints on the global path
  - Obstacles blocking the left lane (costmap lethal cell ahead)
  - Explicit commands from the scenario ("u_turn" / "reverse" / "normal")
"""

import math
import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from config import ROAD_CENTERLINE_Y

from maneuvers.left_turn  import LeftTurnManeuver
from maneuvers.right_turn import RightTurnManeuver
from maneuvers.u_turn     import UTurnManeuver
from maneuvers.reverse    import ReverseManeuver


class ManeuverStateMachine:

    STATES = [
        "NORMAL", "PRE_RIGHT", "IN_LEFT", "IN_RIGHT", "RETURN_LEFT",
        "IN_UTURN", "REVERSING", "OVERTAKING",
    ]

    def __init__(self, mppi_controller):
        self.mppi   = mppi_controller
        self.state  = "NORMAL"
        self.target = None            # current maneuver intermediate target
        self._reverse_t0 = None       # wall-clock entry time for REVERSING
        self._handled_commands = set()
        self._last_completed_maneuver = None
        self.failure_reason: Optional[str] = None
        self.failed_maneuver: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────
    # Main update — called every timestep from main.py
    # ──────────────────────────────────────────────────────────────────────

    def update(self, vehicle_state: np.ndarray, path, costmap,
               scenario_command: Optional[str] = None, t: float = 0.0):
        x, y, theta, v = (float(vehicle_state[0]), float(vehicle_state[1]),
                          float(vehicle_state[2]), float(vehicle_state[3]))

        # ── External commands override path-based detection ────────────
        if scenario_command not in ("u_turn", "reverse"):
            self._handled_commands.clear()
        if scenario_command == "u_turn" and "u_turn" not in self._handled_commands:
            self._handled_commands.add("u_turn")
            self._enter_uturn(vehicle_state, costmap)
            return
        if scenario_command == "reverse" and "reverse" not in self._handled_commands:
            self._handled_commands.add("reverse")
            self._enter_reverse(t)
            return
        # Releasing a commanded reverse
        if self.state == "REVERSING" and scenario_command == "normal":
            self.state = "NORMAL"
            self.target = None
            self.mppi.set_maneuver("normal")
            return

        # Path-based detection: heading change to upcoming waypoints
        heading_change = self._detect_upcoming_turn(path, vehicle_state)

        if self.state == "NORMAL":
            if self._left_lane_blocked(vehicle_state, costmap):
                self._enter_overtake()
            elif heading_change < -0.5:               # sharp left (>~30 deg)
                self._enter_left_turn(vehicle_state, costmap)
            elif heading_change > 0.5:                # sharp right
                target = self._project_target(
                    RightTurnManeuver.preposition(vehicle_state), costmap
                )
                if target is None:
                    self._fail_maneuver("right_turn", "maneuver_unreachable")
                else:
                    self.state = "PRE_RIGHT"
                    self.target = target
                    self.mppi.set_maneuver("right_turn", self.target)
            else:
                self.mppi.set_maneuver("normal")

        elif self.state == "PRE_RIGHT":
            if RightTurnManeuver.in_right_lane(vehicle_state):
                target = self._project_target(RightTurnManeuver.target(vehicle_state), costmap)
                if target is None:
                    self._fail_maneuver("right_turn", "maneuver_unreachable")
                else:
                    self.state = "IN_RIGHT"
                    self.target = target
                    self.mppi.set_maneuver("right_turn", self.target)

        elif self.state == "OVERTAKING":
            if not self._left_lane_blocked(vehicle_state, costmap):
                self.state = "RETURN_LEFT"
                self.mppi.set_maneuver("return_left")

        elif self.state == "RETURN_LEFT":
            if y <= ROAD_CENTERLINE_Y + 0.3:          # back in left lane
                was_overtaking = self.mppi.maneuver_mode == "return_left" and self.target is None
                self.state = "NORMAL"
                self.target = None
                self.mppi.set_maneuver("normal")
                if was_overtaking:
                    self._last_completed_maneuver = "overtake"

        elif self.state == "IN_LEFT":
            if LeftTurnManeuver.complete(vehicle_state, self.target):
                self._finish_turn()

        elif self.state == "IN_RIGHT":
            if RightTurnManeuver.complete(vehicle_state, self.target):
                self.state = "RETURN_LEFT"
                self.mppi.set_maneuver("return_left")

        elif self.state == "IN_UTURN":
            if UTurnManeuver.complete(vehicle_state, self.target):
                self.state = "RETURN_LEFT"
                self.mppi.set_maneuver("return_left")

        elif self.state == "REVERSING":
            if t - self._reverse_t0 >= ReverseManeuver.MAX_DURATION_S:
                self.state = "NORMAL"
                self.target = None
                self.mppi.set_maneuver("normal")

    # ──────────────────────────────────────────────────────────────────────
    # State entries
    # ──────────────────────────────────────────────────────────────────────

    def _enter_left_turn(self, state: np.ndarray, costmap=None):
        # Left turn: already in left lane, go straight to the arc
        target = self._project_target(LeftTurnManeuver.target(state), costmap)
        if target is None:
            self._fail_maneuver("left_turn", "maneuver_unreachable")
            return
        self.state = "IN_LEFT"
        self.target = target
        self.mppi.set_maneuver("left_turn", self.target)

    def _enter_overtake(self):
        self.state = "OVERTAKING"
        self.target = None
        self.mppi.set_maneuver("overtake")

    def _enter_uturn(self, state: np.ndarray, costmap=None):
        if not UTurnManeuver.feasible():
            self._fail_maneuver("u_turn", "uturn_infeasible")
            return

        target = self._project_target(UTurnManeuver.target(state), costmap)
        if target is None:
            self._fail_maneuver("u_turn", "maneuver_unreachable")
            return

        self.state = "IN_UTURN"
        self.target = target
        self.mppi.set_maneuver("u_turn", self.target)

    @staticmethod
    def _project_target(target, costmap):
        if costmap is None or costmap.query(target[0], target[1]) < config.INSCRIBED_COST:
            return target
        tx, ty = costmap.project_goal_to_free(target[0], target[1])
        if costmap.query(tx, ty) >= config.INSCRIBED_COST:
            return None
        return (tx, ty, target[2])

    def _fail_maneuver(self, maneuver: str, reason: str) -> None:
        self.failure_reason = reason
        self.failed_maneuver = maneuver
        self.state = "NORMAL"
        self.target = None
        self.mppi.set_maneuver("normal")

    def _enter_reverse(self, t: float):
        if self.state != "REVERSING":
            self._reverse_t0 = t
        self.state = "REVERSING"
        self.target = None
        self.mppi.set_maneuver("reverse")

    def _finish_turn(self):
        self.state = "NORMAL"
        self.target = None
        self.mppi.set_maneuver("normal")

    # ──────────────────────────────────────────────────────────────────────
    # Detection helpers
    # ──────────────────────────────────────────────────────────────────────

    def _left_lane_blocked(self, state: np.ndarray, costmap) -> bool:
        """True if costmap shows a lethal cell 2 m ahead on current heading."""
        if costmap is None:
            return False
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        ahead_x = x + 2.0 * math.cos(theta)
        ahead_y = y + 2.0 * math.sin(theta)
        return costmap.query(ahead_x, ahead_y) >= config.LETHAL_COST

    def _detect_upcoming_turn(self, path, state: np.ndarray) -> float:
        """Signed heading change from current heading to the path ~3 waypoints ahead."""
        if not path or len(path) < 2:
            return 0.0
        x, y = float(state[0]), float(state[1])
        dists = [math.hypot(p[0] - x, p[1] - y) for p in path]
        idx = int(np.argmin(dists))
        far = min(idx + 3, len(path) - 1)
        dx = path[far][0] - path[idx][0]
        dy = path[far][1] - path[idx][1]
        if math.hypot(dx, dy) < 1e-6:
            return 0.0
        target_heading = math.atan2(dy, dx)
        return math.atan2(
            math.sin(target_heading - float(state[2])),
            math.cos(target_heading - float(state[2])),
        )

    def reset(self):
        self.state = "NORMAL"
        self.target = None
        self._reverse_t0 = None
        self._handled_commands.clear()
        self._last_completed_maneuver = None
        self.failure_reason = None
        self.failed_maneuver = None


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== ManeuverStateMachine self-test ===")

    class FakeMPPI:
        def __init__(self):
            self.mode, self.target = "normal", None
        def set_maneuver(self, mode, target=None):
            self.mode, self.target = mode, target

    mppi = FakeMPPI()
    msm = ManeuverStateMachine(mppi)

    # 1. Straight path, nothing ahead → stays NORMAL
    state = np.array([2.0, 8.5, 0.0, 1.0])
    path = [(float(x), 8.5) for x in range(0, 21)]
    msm.update(state, path, None)
    assert msm.state == "NORMAL" and mppi.mode == "normal"

    # 2. Sharp left turn in path → IN_LEFT with left target
    lpath = [(2.0, 8.5), (4.0, 8.5), (6.0, 8.5), (6.0, 5.0), (6.0, 2.0)]
    msm.update(state, lpath, None)
    assert msm.state == "IN_LEFT", msm.state
    assert mppi.mode == "left_turn" and mppi.target is not None

    # 3. U-turn command → IN_UTURN, completes on 180-deg heading
    msm.reset()
    state = np.array([10.0, 8.5, 0.0, 1.0])
    msm.update(state, path, None, scenario_command="u_turn")
    assert msm.state == "IN_UTURN" and mppi.mode == "u_turn"
    # Heading still 0 → not complete
    msm.update(state, path, None)
    assert msm.state == "IN_UTURN"
    # Rotate to pi → complete → RETURN_LEFT
    state2 = np.array([10.0, 14.0, math.pi - 0.05, 1.0])
    msm.update(state2, path, None)
    assert msm.state == "RETURN_LEFT", msm.state
    # Back in left lane → NORMAL
    state3 = np.array([10.0, 8.5, -math.pi + 0.05, 1.0])
    msm.update(state3, path, None)
    assert msm.state == "NORMAL" and mppi.mode == "normal"

    # 4. Reverse command → REVERSING, auto-exits after timeout
    msm.reset()
    msm.update(state, path, None, scenario_command="reverse", t=0.0)
    assert msm.state == "REVERSING" and mppi.mode == "reverse"
    msm.update(state, path, None, t=5.0)
    assert msm.state == "NORMAL", msm.state

    # 5. Right turn detection → PRE_RIGHT → IN_RIGHT once in right lane
    msm.reset()
    rpath = [(2.0, 8.5), (4.0, 8.5), (6.0, 8.5), (6.0, 12.5), (6.0, 15.0)]
    msm.update(state, rpath, None)
    assert msm.state == "PRE_RIGHT", msm.state
    state4 = np.array([5.0, 12.5, 0.3, 1.0])
    msm.update(state4, rpath, None)
    assert msm.state == "IN_RIGHT" and mppi.mode == "right_turn"

    print("[PASS] ManeuverStateMachine verified!")

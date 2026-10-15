"""
scenarios/s5_sudden_obstacle.py
================================
Scenario 5 — Sudden Obstacle (Late Spawn)

Map layout (20 x 20 m):
    Start : (1, 10)   heading East
    Goal  : (19, 10)
    Path  : straight horizontal line at y = 10  (initially clear)

    At t = SPAWN_TIME (8.0 s):
        A 1.0 x 1.0 m box spawns at (12, 10) — directly on the path.

    The robot will typically be around x ≈ 9–10 m when the obstacle
    appears (at max_speed 1.5 m/s, distance covered in 8s ≈ 9.5 m from
    near-zero — actual position depends on acceleration profile).

    Edge case handled:
        If the robot has already passed x=12 when the obstacle spawns,
        the spawn position is shifted to x=16 (still ahead on path).
        This prevents an obstacle from materialising inside the robot.

Challenge: Tests real-time reactivity. DWA plans only a short horizon;
           RPP relies on look-ahead; MPPI re-plans with K samples.
           The winner here is whichever brakes / steers fastest after
           the obstacle suddenly appears.
"""

import copy
import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
import config


_SPAWN_TIME = 8.0       # seconds
_SPAWN_X    = 12.0
_SPAWN_Y    = 10.0
_SPAWN_W    = 1.0
_SPAWN_H    = 1.0
_FALLBACK_SPAWN_X = 16.0   # used if robot has already passed _SPAWN_X


class S5SuddenObstacle(ScenarioBase):
    """
    Scenario 5: Path is initially clear.  At t=8s a box spawns at (12,10).
    The robot must react in real time with no forewarning.
    """

    def __init__(self):
        super().__init__()
        self.name = "s5_sudden_obstacle"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        # Start with NO obstacles — path is clear
        self._initial_obstacles = []
        self.obstacles          = []

        # Track whether the obstacle has been spawned yet
        self._spawned = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        """Clear spawn flag so the obstacle re-spawns fresh each run."""
        self._spawned = False

    # ------------------------------------------------------------------
    # Update — check spawn trigger each timestep
    # ------------------------------------------------------------------

    def update(self, t: float, robot_state: np.ndarray = None) -> None:
        """
        Advance scenario.  At t >= SPAWN_TIME, inject the obstacle.

        Parameters
        ----------
        t           : float — current simulation time (seconds)
        robot_state : np.ndarray [x, y, theta, v]  — used to check if the
                      robot has already passed the nominal spawn point.
        """
        if not self._spawned and t >= _SPAWN_TIME:
            spawn_x = _SPAWN_X

            # Edge-case: if robot already passed the spawn point, shift it ahead
            if robot_state is not None:
                robot_x = float(robot_state[0])
                if robot_x > spawn_x - config.ROBOT_RADIUS * 2:
                    spawn_x = _FALLBACK_SPAWN_X
                    # If robot has also passed the fallback, don't spawn
                    # (robot is essentially at goal — let the run finish cleanly)
                    if robot_x > spawn_x - config.ROBOT_RADIUS * 2:
                        self._spawned = True
                        return

            obstacle = {
                "type":  "static",
                "x":     spawn_x,
                "y":     _SPAWN_Y,
                "w":     _SPAWN_W,
                "h":     _SPAWN_H,
                "label": "sudden_box",
            }
            self.obstacles.append(obstacle)
            self._spawned = True

    @property
    def spawned(self) -> bool:
        """True after the obstacle has been injected into the scenario."""
        return self._spawned


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = S5SuddenObstacle()
    print(s)
    print(f"  Obstacles at t=0: {len(s.obstacles)}  (expect 0)")
    assert len(s.obstacles) == 0

    # Simulate up to spawn time
    t = 0.0
    robot = np.array([1.0, 10.0, 0.0, 0.0])
    spawned_at = None
    for step in range(int(10.0 / config.DT)):
        s.update(t, robot_state=robot)
        # Fake robot moving at 1.0 m/s
        robot[0] += 1.0 * config.DT
        if s.spawned and spawned_at is None:
            spawned_at = t
        t += config.DT

    print(f"  Obstacle spawned at t={spawned_at:.2f} s  (expect ~8.0)")
    assert spawned_at is not None, "Obstacle never spawned"
    assert abs(spawned_at - _SPAWN_TIME) < config.DT * 2, \
        f"Wrong spawn time: {spawned_at}"
    assert len(s.obstacles) == 1
    print(f"  Spawn position: ({s.obstacles[0]['x']}, {s.obstacles[0]['y']})")

    # Reset clears obstacle
    s.reset()
    assert len(s.obstacles) == 0, "Obstacles should clear after reset"
    assert not s.spawned, "Spawn flag should clear after reset"

    # Edge case: robot already past spawn point
    s2 = S5SuddenObstacle()
    robot_past = np.array([13.0, 10.0, 0.0, 1.0])   # past x=12
    s2.update(_SPAWN_TIME + 0.1, robot_state=robot_past)
    print(f"  Fallback spawn x: {s2.obstacles[0]['x'] if s2.obstacles else 'no spawn'}")

    print("[PASS]  S5SuddenObstacle OK")

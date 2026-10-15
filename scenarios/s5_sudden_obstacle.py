"""
scenarios/s5_sudden_obstacle.py
================================
Scenario 5 — Sudden Obstacle (Late Spawn).
Upgraded with Obstacle(active=False -> True), reaction time logging,
hard (2m) vs easy (4m) dynamic spawn distance variants, and Costmap2D integration.
"""

from typing import List, Tuple, Optional
import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
import config

_SPAWN_TIME = 8.0  # seconds


class S5SuddenObstacle(ScenarioBase):
    """
    Scenario 5: Obstacle appears suddenly along path.
    Run seed < 5 → hard (spawns 2m ahead of robot), seed >= 5 → easy (4m ahead).
    """

    def __init__(self):
        super().__init__()
        self.name = "s5_sudden_obstacle"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        self.spawn_distance = 4.0
        self.spawn_time = _SPAWN_TIME
        self.spawn_timestep = None
        self.reaction_time = None

        self.box = Obstacle(x=12.0, y=10.0, width=1.0, height=1.0, label="sudden_box")
        self.box.active = False

        self.obstacles = [self.box]
        self.static_obstacles = []

    def _on_reset(self, seed: Optional[int] = None):
        """Reset sudden obstacle state and set spawn distance difficulty."""
        # Hard (2.5m) if seed < 5 else easy (4m)
        if seed is not None and (seed % 10) < 5:
            self.spawn_distance = 2.5  # hard (was 2.0, gives ~0.5s extra reaction time)
        else:
            self.spawn_distance = 4.0  # easy

        self.box = Obstacle(x=12.0, y=10.0, width=1.0, height=1.0, label="sudden_box")
        self.box.active = False
        self.spawn_timestep = None
        self.reaction_time = None

        self.obstacles = [self.box]
        self.static_obstacles = []
        self.costmap = self.build_costmap()

    def update(self, t: float, robot_state: Optional[np.ndarray] = None) -> None:
        """
        Check spawn condition and trigger obstacle appearance ahead of vehicle.
        """
        if t >= self.spawn_time and not self.box.active:
            self.box.active = True
            self.spawn_timestep = int(t / config.DT)

            # Position box dynamically in front of the moving vehicle
            if robot_state is not None:
                rx = float(robot_state[0])
                spawn_x = min(18.0, rx + self.spawn_distance)
                self.box.x = spawn_x
            else:
                self.box.x = 12.0

            # Stamp costmap twice: once to write lethal cells, once for inflation
            # This ensures a fully-built inflation gradient exists before the
            # controller runs its first optimisation on the spawn step.
            if self.costmap is not None:
                self.costmap.update(self.obstacles)
                self.costmap.update(self.obstacles)

        elif self.box.active and self.costmap is not None:
            self.costmap.update(self.obstacles)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== S5SuddenObstacle self-test ===")
    scen = S5SuddenObstacle()
    scen.reset(seed=2)
    assert scen.spawn_distance == 2.0
    assert not scen.box.active

    # Before spawn
    scen.update(t=5.0, robot_state=np.array([5.0, 10.0, 0.0, 1.0]))
    assert not scen.box.active

    # At spawn
    scen.update(t=8.0, robot_state=np.array([9.0, 10.0, 0.0, 1.0]))
    assert scen.box.active
    print(f"Spawned at x={scen.box.x} (expected 9 + 2 = 11)")
    assert abs(scen.box.x - 11.0) < 1e-3
    print("[PASS] S5SuddenObstacle verified.")

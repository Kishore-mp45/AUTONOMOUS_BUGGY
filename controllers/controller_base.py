"""
controllers/controller_base.py
==============================
Common interface for RPP, DWA, and MPPI local planners.
All controllers implement compute_control() and reset().
"""

from typing import List, Tuple
import numpy as np


class ControllerBase:
    """
    Common interface for RPP, DWA, MPPI.
    All controllers must implement compute_control().
    """
    name = "Base"

    def compute_control(
        self,
        state: np.ndarray,
        path: List[Tuple[float, float]],
        goal: Tuple[float, float],
        costmap,
        obstacles: list,
    ) -> Tuple[float, float]:
        """
        Compute control action.

        Args:
            state     : [x, y, theta, v]
            path      : list of (x,y) waypoints (reference path)
            goal      : (gx, gy)
            costmap   : Costmap2D instance
            obstacles : list of Obstacle objects

        Returns:
            (steering_angle [rad], target_speed [m/s])
        """
        raise NotImplementedError

    def reset(self):
        """Called at the start of each run. Reset internal state."""
        pass

"""
maneuvers/u_turn.py
===================
U-turn maneuver helper.

Indian left-hand rule: move to the rightmost position available, execute
the 180-degree arc (minimum turning radius enforced by MPPI's rollout
clamp), then return to the left lane.
"""

import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from config import ROAD_Y_MAX, MANEUVER_UTURN_RADIUS


class UTurnManeuver:
    """Supplies the U-turn target and completion (heading) check."""

    # Offset from the right road edge for the pre-arc position
    EDGE_OFFSET = 0.5

    @classmethod
    def target(cls, state: np.ndarray):
        """
        Target (x, y, theta): far-right lane position, heading reversed
        by 180 degrees (wrapped to [-pi, pi]).
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        return (x, ROAD_Y_MAX - cls.EDGE_OFFSET, math.atan2(
            math.sin(theta + math.pi), math.cos(theta + math.pi)))

    @staticmethod
    def complete(state: np.ndarray, target) -> bool:
        """True when within ~12 degrees of the target heading."""
        if target is None:
            return False
        heading_diff = abs(math.atan2(
            math.sin(float(state[2]) - target[2]),
            math.cos(float(state[2]) - target[2]),
        ))
        return heading_diff < 0.2

    @staticmethod
    def feasible() -> bool:
        """
        EC-15: a U-turn needs road width >= 2 * turn radius (down and back
        across the road). Returns False when the road is too narrow.
        """
        road_width = config.ROAD_Y_MAX - config.ROAD_Y_MIN
        return road_width >= 2.0 * MANEUVER_UTURN_RADIUS

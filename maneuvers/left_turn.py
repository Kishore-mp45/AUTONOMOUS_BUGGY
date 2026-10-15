"""
maneuvers/left_turn.py
======================
Left-turn maneuver helper.

Indian left-hand rule: for a left turn, stay in the LEFTMOST position
before the turn point — no wide swing into the right lane.
"""

import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from config import LEFT_LANE_CENTER_Y


class LeftTurnManeuver:
    """Supplies the intermediate target and completion check for a left turn."""

    # metres past the vehicle where the turn target is placed
    LOOKAHEAD = 3.0

    @classmethod
    def target(cls, state: np.ndarray):
        """
        Target waypoint (x, y, theta) for the left-turn arc:
        ahead on the left lane centre, heading rotated -90 degrees.
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        return (x + cls.LOOKAHEAD, LEFT_LANE_CENTER_Y, theta - math.pi / 2.0)

    @staticmethod
    def complete(state: np.ndarray, target) -> bool:
        """True when the vehicle is within tolerance of the turn target."""
        if target is None:
            return True
        return math.hypot(state[0] - target[0], state[1] - target[1]) < 0.5

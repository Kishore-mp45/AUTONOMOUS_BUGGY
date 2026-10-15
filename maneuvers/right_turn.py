"""
maneuvers/right_turn.py
=======================
Right-turn maneuver helper.

Indian left-hand rule: before a right turn, cross to the RIGHT lane,
complete the turn, then return to the left lane (RETURN_LEFT state).
"""

import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from config import RIGHT_LANE_CENTER_Y, ROAD_CENTERLINE_Y


class RightTurnManeuver:
    """Supplies pre-positioning and turn targets plus completion check."""

    # metres past the vehicle where targets are placed
    LOOKAHEAD = 3.0

    @classmethod
    def preposition(cls, state: np.ndarray):
        """
        Pre-turn target: move to the right lane centre before the turn.
        The ManeuverStateMachine holds this until the vehicle is in the
        right lane, then switches to the arc target.
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        return (x + cls.LOOKAHEAD, RIGHT_LANE_CENTER_Y, theta)

    @classmethod
    def target(cls, state: np.ndarray):
        """
        Target waypoint (x, y, theta) for the right-turn arc:
        ahead on the right lane centre, heading rotated +90 degrees.
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        return (x + cls.LOOKAHEAD, RIGHT_LANE_CENTER_Y, theta + math.pi / 2.0)

    @staticmethod
    def in_right_lane(state: np.ndarray) -> bool:
        return float(state[1]) > ROAD_CENTERLINE_Y

    @staticmethod
    def complete(state: np.ndarray, target) -> bool:
        if target is None:
            return True
        return math.hypot(state[0] - target[0], state[1] - target[1]) < 0.5

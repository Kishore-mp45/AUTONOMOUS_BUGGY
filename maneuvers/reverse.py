"""
maneuvers/reverse.py
====================
Reverse maneuver helper.

Used for dead-end escapes: back up slowly, then resume normal planning.
Reverse is only entered on explicit scenario command and exits when the
scenario releases it (command "normal") or a timeout elapses.
"""

import math
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ReverseManeuver:

    # Reverse for at most this many seconds before reverting to NORMAL
    MAX_DURATION_S = 4.0

    @staticmethod
    def target(state: np.ndarray):
        """
        Target for reverse: stay on the left lane centre y, keep heading,
        move backwards (the MPPI speed critic is inactive in reverse mode —
        the state machine caps speed via MANEUVER_REVERSE_MAX_SPEED).
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        return (x, y, theta)

    @staticmethod
    def max_speed() -> float:
        return config.MANEUVER_REVERSE_MAX_SPEED

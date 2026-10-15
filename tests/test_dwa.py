"""
tests/test_dwa.py
=================
Unit tests for Ackermann Dynamic Window Approach (DWA) controller.
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.dwa import DWA
from environment.costmap import Costmap2D
from environment.obstacle import Obstacle


def test_ackermann_feasibility():
    """Steering angles yielding radius < DWA_MIN_TURN_RADIUS are flagged unfeasible."""
    dwa = DWA()
    # At delta = MAX_STEER (0.6 rad), radius = L / tan(0.6) = 0.5 / 0.684 = ~0.73m >= 0.5m
    assert dwa._is_ackermann_feasible(1.0, 0.2)
    # If delta approached pi/2, radius -> 0, which violates DWA_MIN_TURN_RADIUS
    unfeasible_delta = math.atan(config.WHEELBASE / (config.DWA_MIN_TURN_RADIUS * 0.5))
    assert not dwa._is_ackermann_feasible(1.0, unfeasible_delta)


def test_no_valid_trajectory_returns_stop():
    """If all candidate trajectories result in collision, safe stop (0, 0) is returned."""
    dwa = DWA()
    cm = Costmap2D()
    # Obstacle wall completely engulfing vehicle area
    wall = Obstacle(x=2.0, y=10.0, width=8.0, height=8.0)
    cm.add_static_obstacle(wall)
    state = np.array([1.0, 10.0, 0.0, 0.5])
    steer, speed = dwa.compute_control(state, goal=(19.0, 10.0), costmap=cm)
    assert steer == 0.0
    assert speed == 0.0


def test_window_respects_speed_limits():
    """Dynamic window speeds must remain strictly within [0.0, MAX_SPEED]."""
    dwa = DWA()
    state = np.array([1.0, 10.0, 0.0, config.MAX_SPEED])
    v_min, v_max, d_min, d_max = dwa._build_window(state)
    assert v_max <= config.MAX_SPEED
    assert v_min >= 0.0
    assert -config.MAX_STEER <= d_min <= d_max <= config.MAX_STEER

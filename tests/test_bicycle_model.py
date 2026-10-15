"""
tests/test_bicycle_model.py
===========================
Unit tests for kinematics, limits, Ackermann constraints, and jerk tracking.
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from robot.bicycle_model import BicycleModel, make_state


def test_straight_line():
    """v > 0, delta = 0 -> robot should travel along heading without turning."""
    model = BicycleModel()
    state = make_state(x=0.0, y=0.0, theta=0.0, v=1.0)
    # 1 second straight at 1.0 m/s
    for _ in range(int(1.0 / config.DT)):
        state = model.update(state, np.array([0.0, 0.0]))
    
    assert math.isclose(state[0], 1.0, rel_tol=1e-2, abs_tol=1e-2)
    assert math.isclose(state[1], 0.0, abs_tol=1e-3)
    assert math.isclose(state[2], 0.0, abs_tol=1e-3)


def test_turning():
    """delta > 0 -> heading should increase (left turn in standard counter-clockwise convention)."""
    model = BicycleModel()
    state = make_state(x=0.0, y=0.0, theta=0.0, v=1.0)
    # Steer left by 0.2 rad for 0.5s
    for _ in range(10):
        state = model.update(state, np.array([0.2, 0.0]))
    
    assert state[2] > 0.0
    assert state[1] > 0.0


def test_speed_clamp():
    """Large positive acceleration must not exceed MAX_SPEED."""
    model = BicycleModel()
    state = make_state(x=0.0, y=0.0, theta=0.0, v=1.0)
    state = model.update(state, np.array([0.0, 100.0]))
    assert state[3] <= config.MAX_SPEED


def test_steer_clamp():
    """Large steering command must be clamped to [-MAX_STEER, MAX_STEER]."""
    model = BicycleModel()
    state = make_state(x=0.0, y=0.0, theta=0.0, v=1.0)
    # Command steer = 10.0 rad
    state = model.update(state, np.array([10.0, 0.0]))
    assert abs(model.steering_angle) <= config.MAX_STEER


def test_ackermann_constraint():
    """Steering angle violating minimum turning radius must be clamped."""
    model = BicycleModel()
    # At minimum turn radius R_min = 0.5m, max allowable delta = arctan(L / R_min)
    max_delta = math.atan(config.WHEELBASE / config.DWA_MIN_TURN_RADIUS)
    state = make_state(x=0.0, y=0.0, theta=0.0, v=1.0)
    state = model.update(state, np.array([max_delta + 0.3, 0.0]))
    assert abs(model.steering_angle) <= max_delta + 1e-5


def test_jerk_tracking():
    """Two consecutive acceleration changes compute real-time jerk."""
    model = BicycleModel()
    state = make_state(x=0.0, y=0.0, theta=0.0, v=0.5)
    model.update(state, np.array([0.0, 0.5]))
    model.update(state, np.array([0.0, 1.0]))
    assert model.current_jerk >= 0.0

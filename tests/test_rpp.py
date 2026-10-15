"""
tests/test_rpp.py
=================
Unit tests for Regulated Pure Pursuit (RPP) controller.
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.rpp import RegulatedPurePursuit
from environment.costmap import Costmap2D


def test_lookahead_grows_with_speed():
    """Adaptive lookahead distance must scale with vehicle velocity."""
    rpp = RegulatedPurePursuit()
    l_slow = rpp._get_lookahead(0.2)
    l_fast = rpp._get_lookahead(1.5)
    assert l_fast > l_slow
    assert config.RPP_MIN_LOOKAHEAD <= l_slow <= config.RPP_MAX_LOOKAHEAD
    assert config.RPP_MIN_LOOKAHEAD <= l_fast <= config.RPP_MAX_LOOKAHEAD


def test_low_pass_filter_smooths_steer():
    """Steering command filter prevents sharp instantaneous jumps."""
    rpp = RegulatedPurePursuit()
    rpp._filtered_steer = 0.0
    # Step input of 0.5 rad
    filtered = rpp._filter_steer(0.5)
    assert filtered < 0.5
    assert filtered > 0.0


def test_goal_reached_returns_zero():
    """When vehicle arrives at goal within tolerance, zero control is issued."""
    rpp = RegulatedPurePursuit()
    goal = (19.0, 10.0)
    state = np.array([19.0, 10.0, 0.0, 0.1])
    path = [(18.0, 10.0), (19.0, 10.0)]
    steer, speed = rpp.compute_control(state, path=path, goal=goal)
    assert steer == 0.0
    assert speed == 0.0

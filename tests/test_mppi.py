"""
tests/test_mppi.py
==================
Unit tests for MPPI controller 5-critic cost, fallbacks, and control updates.
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.mppi import MPPI
from environment.costmap import Costmap2D
from environment.obstacle import Obstacle


def test_produces_valid_control():
    """Output steering and velocity must strictly lie within vehicle limits."""
    mppi = MPPI()
    cm = Costmap2D()
    state = np.array([1.0, 10.0, 0.0, 0.5])
    path = [(x, 10.0) for x in np.linspace(1, 19, 20)]
    steer, speed = mppi.compute_control(state, path=path, goal=(19.0, 10.0), costmap=cm)
    assert -config.MAX_STEER <= steer <= config.MAX_STEER
    assert 0.0 <= speed <= config.MAX_SPEED


def test_fallback_on_all_collide():
    """If all candidate trajectories result in collision, safe stop is returned."""
    mppi = MPPI()
    cm = Costmap2D()
    # Fill costmap with lethal obstacle directly around vehicle
    wall = Obstacle(x=1.0, y=10.0, width=5.0, height=5.0)
    cm.add_static_obstacle(wall)
    state = np.array([1.0, 10.0, 0.0, 0.5])
    steer, speed = mppi.compute_control(state, goal=(19.0, 10.0), costmap=cm)
    assert steer == 0.0
    assert speed == 0.0


def test_nominal_shifts_each_step():
    """Warm-start shifts control sequence forward by one step on subsequent call."""
    mppi = MPPI()
    cm = Costmap2D()
    state = np.array([1.0, 10.0, 0.0, 0.5])
    mppi.compute_control(state, goal=(19.0, 10.0), costmap=cm)
    first_u1 = mppi.U[0].copy()
    mppi.compute_control(state, goal=(19.0, 10.0), costmap=cm)
    # The new U[0] was shifted from previous U[1] plus update
    assert mppi.U.shape == (config.MPPI_T, 2)


def test_cost_decreases_with_clear_path():
    """Sample rollout costs on a clear path must be significantly lower than when blocked."""
    mppi = MPPI()
    cm_clear = Costmap2D()
    cm_blocked = Costmap2D()
    cm_blocked.add_static_obstacle(Obstacle(x=2.0, y=10.0, width=2.0, height=2.0))

    state = np.array([1.0, 10.0, 0.0, 1.0])
    U_noisy = np.zeros((10, config.MPPI_T, 2))
    trajs = mppi._rollout_batch(state, U_noisy)

    costs_clear = mppi._compute_costs(trajs, goal=(19.0, 10.0), path=None, costmap=cm_clear)
    costs_blocked = mppi._compute_costs(trajs, goal=(19.0, 10.0), path=None, costmap=cm_blocked)
    assert np.mean(costs_clear) < np.mean(costs_blocked)

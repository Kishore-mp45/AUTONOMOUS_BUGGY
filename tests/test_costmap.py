"""
tests/test_costmap.py
=====================
Unit tests for Costmap2D lethal cells, decay inflation, and projection.
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from environment.costmap import Costmap2D
from environment.obstacle import Obstacle


def test_lethal_at_obstacle():
    """Cells directly inside an obstacle must evaluate to LETHAL_COST."""
    cm = Costmap2D()
    obs = Obstacle(x=10.0, y=10.0, width=1.0, height=1.0)
    cm.add_static_obstacle(obs)
    cost = cm.query(10.0, 10.0)
    assert cost == config.LETHAL_COST


def test_free_far_from_obstacle():
    """Cells far from any obstacle (e.g., 5m) must have near-zero cost."""
    cm = Costmap2D()
    obs = Obstacle(x=10.0, y=10.0, width=1.0, height=1.0)
    cm.add_static_obstacle(obs)
    cost = cm.query(2.0, 2.0)
    assert cost < 5.0


def test_inflation_monotone():
    """Cost strictly increases as you approach the obstacle surface."""
    cm = Costmap2D()
    obs = Obstacle(x=10.0, y=10.0, radius=0.5)
    cm.add_static_obstacle(obs)
    c_far  = cm.query(10.0, 12.0)  # 1.5m from surface
    c_mid  = cm.query(10.0, 11.0)  # 0.5m from surface
    c_near = cm.query(10.0, 10.6)  # 0.1m from surface
    assert c_far < c_mid < c_near


def test_goal_projection():
    """Goal positioned inside an obstacle is projected out to the nearest free cell."""
    cm = Costmap2D()
    obs = Obstacle(x=10.0, y=10.0, width=2.0, height=2.0)
    cm.add_static_obstacle(obs)
    proj_x, proj_y = cm.project_goal_to_free(10.0, 10.0)
    assert cm.query(proj_x, proj_y) < config.INSCRIBED_COST


def test_update_clears_old_obstacle():
    """Moving a dynamic obstacle frees its previous location."""
    cm = Costmap2D()
    dyn_obs = Obstacle(x=5.0, y=5.0, radius=0.4, dynamic=True)
    cm.update([dyn_obs])
    assert cm.query(5.0, 5.0) == config.LETHAL_COST

    # Move obstacle to (10.0, 10.0)
    dyn_obs.x = 10.0
    dyn_obs.y = 10.0
    cm.update([dyn_obs])

    # Old position should no longer be lethal
    assert cm.query(5.0, 5.0) < config.LETHAL_COST
    # New position should be lethal
    assert cm.query(10.0, 10.0) == config.LETHAL_COST

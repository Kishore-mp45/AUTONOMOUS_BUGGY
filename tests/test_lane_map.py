"""Unit tests for road geometry and left-hand lane rules."""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from environment.lane_map import LaneMap


def test_left_lane_center_has_zero_cost():
    lane_map = LaneMap()

    assert lane_map.lane_cost(5.0, config.LEFT_LANE_CENTER_Y) == 0.0


def test_right_lane_center_has_soft_penalty():
    lane_map = LaneMap()

    assert lane_map.lane_cost(5.0, config.RIGHT_LANE_CENTER_Y) == config.LANE_COST_RIGHT


def test_outside_road_has_boundary_cost():
    lane_map = LaneMap()

    assert lane_map.lane_cost(5.0, config.ROAD_Y_MIN - 0.1) == config.LANE_COST_BOUNDARY


def test_normal_mode_targets_left_lane():
    assert LaneMap().get_target_lane_y("normal") == config.LEFT_LANE_CENTER_Y


def test_uturn_mode_targets_right_lane():
    assert LaneMap().get_target_lane_y("u_turn") == config.RIGHT_LANE_CENTER_Y


def test_right_lane_is_violation_in_normal_mode():
    lane_map = LaneMap()

    assert lane_map.wrong_lane_violation(
        5.0, config.RIGHT_LANE_CENTER_Y, "normal"
    ) is True


def test_right_lane_is_allowed_during_overtake():
    lane_map = LaneMap()

    assert lane_map.wrong_lane_violation(
        5.0, config.RIGHT_LANE_CENTER_Y, "overtake"
    ) is False


def test_batch_lane_cost_matches_scalar_queries():
    lane_map = LaneMap()
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([
        config.LEFT_LANE_CENTER_Y,
        config.RIGHT_LANE_CENTER_Y,
        config.ROAD_Y_MIN - 0.1,
        config.ROAD_Y_MAX + 0.1,
    ])

    batch = lane_map.lane_cost_batch(xs, ys)
    scalar = np.array([lane_map.lane_cost(x, y) for x, y in zip(xs, ys)])

    np.testing.assert_allclose(batch, scalar)

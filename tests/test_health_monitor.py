"""Unit tests for MPPI-related health checks."""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from environment.lane_map import LaneMap
from health.monitor import HealthMonitor


def test_mppi_recovery_statuses():
    monitor = HealthMonitor()

    assert monitor.check_mppi_recovery(0) == ("ok", "")
    assert monitor.check_mppi_recovery(1) == (
        "mppi_recovering",
        "Recovery attempt 1",
    )
    assert monitor.check_mppi_recovery(config.MPPI_RECOVERY_MAX_ATTEMPTS) == (
        "mppi_emergency_stop",
        "MPPI exhausted recovery attempts",
    )


def test_lane_discipline_counts_only_violations():
    monitor = HealthMonitor()
    lane_map = LaneMap()
    wrong_lane_state = np.array([2.0, config.RIGHT_LANE_CENTER_Y, 0.0, 0.5])
    left_lane_state = np.array([2.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5])

    assert monitor.check_lane_discipline(
        wrong_lane_state, "normal", lane_map
    ) == ("lane_violation", "Wrong lane at (2.0,12.5)")
    assert monitor.check_lane_discipline(
        wrong_lane_state, "overtake", lane_map
    ) == ("ok", "")
    assert monitor.check_lane_discipline(
        left_lane_state, "normal", lane_map
    ) == ("ok", "")
    assert monitor.lane_violation_count == 1


def test_boundary_check_tracks_road_exit_and_reset_clears_state():
    monitor = HealthMonitor()
    inside_road = np.array([2.0, config.ROAD_CENTERLINE_Y, 0.0, 0.5])
    outside_road = np.array([2.0, config.ROAD_Y_MIN, 0.0, 0.5])

    assert monitor.check_boundary(inside_road) == ("ok", "")
    assert monitor.check_boundary(outside_road) == (
        "boundary_exit",
        f"Exited road at y={config.ROAD_Y_MIN:.2f}",
    )
    assert monitor.boundary_exit is True

    monitor.check_lane_discipline(
        np.array([2.0, config.RIGHT_LANE_CENTER_Y, 0.0, 0.5]),
        "normal",
        LaneMap(),
    )
    monitor.reset()

    assert monitor.lane_violation_count == 0
    assert monitor.boundary_exit is False

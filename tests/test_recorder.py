"""Unit tests for Phase 8 lane, maneuver, and recovery metrics."""

import os
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from environment.lane_map import LaneMap
from environment.obstacle import Obstacle
from metrics.recorder import MetricsRecorder


def _record_step(recorder, t, y, maneuver_mode="normal", recovery_count=0):
    state = np.array([float(t), y, 0.0, 0.5])
    recorder.step(
        state=state,
        obstacles=[],
        t=t,
        lane_map=LaneMap(),
        maneuver_mode=maneuver_mode,
        mppi_recovery_count=recovery_count,
    )


def test_lane_violation_count_and_intervals():
    recorder = MetricsRecorder("MPPI", "lane_test", 0)

    _record_step(recorder, 0.0, config.RIGHT_LANE_CENTER_Y)
    _record_step(recorder, 0.1, config.RIGHT_LANE_CENTER_Y)
    _record_step(recorder, 0.2, config.LEFT_LANE_CENTER_Y)
    _record_step(recorder, 0.3, config.RIGHT_LANE_CENTER_Y, "overtake")
    recorder.finish(success=False, t=0.4, reason="test_complete")

    assert recorder.lane_violation_count == 2
    assert recorder.wrong_lane_durations == [(0.0, 0.2)]
    result = recorder.to_dict()
    assert result["lane_violations"] == 2
    assert result["wrong_lane_durations"] == "[(0.0, 0.2)]"
    assert "oscillation_count" not in result
    assert "recovery_count" not in result


def test_recovery_attempts_count_increments_not_repeated_counter_values():
    recorder = MetricsRecorder("MPPI", "recovery_test", 0)

    for i, count in enumerate((0, 1, 1, 2, 3, 4, 0)):
        _record_step(recorder, i * config.DT, config.LEFT_LANE_CENTER_Y, recovery_count=count)

    assert recorder.recovery_count_total == config.MPPI_RECOVERY_MAX_ATTEMPTS
    assert recorder.to_dict()["recovery_count_total"] == config.MPPI_RECOVERY_MAX_ATTEMPTS


def test_maneuver_results_are_recorded_in_summary():
    recorder = MetricsRecorder("MPPI", "maneuver_test", 0)
    recorder.record_maneuver_result("u_turn", True)
    recorder.record_maneuver_result("right_turn", False)

    assert recorder.maneuver_success == {"u_turn": True, "right_turn": False}
    assert recorder.to_dict()["maneuver_success"] == (
        "{'u_turn': True, 'right_turn': False}"
    )


def test_step_without_phase_eight_inputs_remains_compatible():
    recorder = MetricsRecorder("MPPI", "legacy_call", 0)
    recorder.step(
        np.array([0.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5]),
        [],
        0.0,
    )

    assert recorder.lane_violation_count == 0
    assert recorder.recovery_count_total == 0


def test_timestep_csv_has_distinct_filename(tmp_path):
    recorder = MetricsRecorder("MPPI", "csv_test", 0)
    recorder.step(
        np.array([0.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5]),
        [],
        0.0,
    )

    output_path = recorder.save_timestep_csv(str(tmp_path))

    assert output_path.endswith("MPPI__csv_test__run00_timesteps.csv")
    assert os.path.exists(output_path)


def test_edge_condition_metrics_are_recorded():
    recorder = MetricsRecorder("MPPI", "edge_metrics", 0)
    recorder.goal_adjusted = True
    recorder.gap_width = 1.5
    state = np.array([0.0, config.LEFT_LANE_CENTER_Y + 0.5, 0.0, 0.0])
    obstacles = [Obstacle(x=0.6, y=state[1], radius=0.2)]

    recorder.step(
        state=state,
        obstacles=obstacles,
        t=0.0,
        lane_map=LaneMap(),
        reference_path=[(0.0, config.LEFT_LANE_CENTER_Y), (10.0, config.LEFT_LANE_CENTER_Y)],
        lane_guidance_enabled=False,
        narrow_gap_violation=True,
        weight_fallback_count=2,
        ackermann_clamp_count=3,
    )
    state[3] = 1.0
    recorder.step(
        state=state,
        obstacles=obstacles,
        t=config.DT,
        reference_path=[(0.0, config.LEFT_LANE_CENTER_Y), (10.0, config.LEFT_LANE_CENTER_Y)],
        lane_guidance_enabled=False,
        narrow_gap_violation=True,
        weight_fallback_count=2,
        ackermann_clamp_count=3,
    )

    recorder.finish(success=True, t=2 * config.DT)
    result = recorder.to_dict()
    assert result["goal_adjusted"] == 1
    assert result["hesitation_time_s"] == config.DT
    assert result["near_miss_count"] == 1
    assert result["cross_track_error_mean"] == 0.5
    assert result["lateral_drift_mean"] == 0.5
    assert result["narrow_gap_violations"] == 2
    assert result["weight_fallback_count"] == 2
    assert result["ackermann_clamp_count"] == 6
    assert recorder.near_miss_count == 1
    assert recorder.failure_reason == ""
    assert recorder.completed is True

"""Smoke tests for the Phase 10 MPPI-only report outputs."""

import os
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.report_generator import generate_all_reports


def test_report_generator_creates_all_eight_mppi_plots(tmp_path):
    results_dir = tmp_path / "results"
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True)
    results = pd.DataFrame([
        {
            "controller": "MPPI",
            "scenario": "s1_static_box",
            "run_id": 0,
            "completed": 1,
            "completion_time": 10.0,
            "min_obstacle_dist": 0.6,
            "peak_acceleration": 1.0,
            "peak_jerk": 2.0,
            "lane_violations": 1,
            "wrong_lane_durations": "[(1.0, 1.5)]",
            "maneuver_success": "{'u_turn': True}",
            "recovery_count_total": 2,
        },
        {
            "controller": "MPPI",
            "scenario": "s1_static_box",
            "run_id": 1,
            "completed": 0,
            "completion_time": 8.0,
            "min_obstacle_dist": 0.2,
            "peak_acceleration": 1.2,
            "peak_jerk": 3.0,
            "lane_violations": 0,
            "wrong_lane_durations": "[]",
            "maneuver_success": "{'u_turn': False}",
            "recovery_count_total": 0,
        },
    ])
    results.to_csv(results_dir / "all_results.csv", index=False)
    pd.DataFrame({"t": [0.0, 0.1], "x": [1.0, 1.1], "y": [8.5, 8.6]}).to_csv(
        raw_dir / "MPPI__s1_static_box__run00.csv", index=False
    )

    generate_all_reports(str(results_dir))

    expected = {
        "plot1_path_comparison.png",
        "plot2_min_distance.png",
        "plot3_completion_time.png",
        "plot4_accel_jerk.png",
        "plot5_failure_heatmap.png",
        "plot6_lane_discipline.png",
        "plot7_maneuver_success.png",
        "plot8_mppi_recovery.png",
    }
    output_files = {path.name for path in (results_dir / "plots").glob("*.png")}
    assert expected <= output_files
    assert all((results_dir / "plots" / name).stat().st_size > 0 for name in expected)

"""
metrics/report_generator.py
===========================
Generates the MPPI-only analysis plots (Plots 1 through 8) from run results.

Generated Outputs in results/plots/:
  - plot1_path_comparison.png
  - plot2_min_distance.png
  - plot3_completion_time.png
  - plot4_accel_jerk.png
  - plot5_failure_heatmap.png
  - plot6_lane_discipline.png
  - plot7_maneuver_success.png
  - plot8_mppi_recovery.png
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visualization.plot_results import (
    plot_path_comparison,
    plot_min_distance,
    plot_completion_time,
    plot_accel_jerk,
    plot_failure_heatmap,
    plot_lane_discipline,
    plot_maneuver_success,
    plot_mppi_recovery,
)


def generate_all_reports(
    results_dir: str = "results",
    results_csv: str = None,
    raw_dir: str = None,
    plots_dir: str = None,
):
    """
    Generate all 8 plots and verify summary table.
    """
    all_csv = results_csv or os.path.join(results_dir, "all_results.csv")
    plots_dir = plots_dir or os.path.join(results_dir, "plots")
    raw_dir = raw_dir or os.path.join(results_dir, "raw")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(all_csv):
        print(f"[WARN] {all_csv} not found. Run simulation first.")
        return

    df = pd.read_csv(all_csv)
    if "controller" in df.columns and set(df["controller"].dropna().unique()) - {"MPPI"}:
        raise ValueError("MPPI-only reports require controller='MPPI' for every result row")
    print(f"[REPORT] Generating 8 MPPI plots from {len(df)} run records...")

    # Plot 1: Path Comparison
    print("  -> Plot 1: Path Comparison...")
    plot_path_comparison(raw_dir, os.path.join(plots_dir, "plot1_path_comparison.png"))

    # Plot 2: Min Obstacle Distance
    print("  -> Plot 2: Min Obstacle Distance...")
    plot_min_distance(df, os.path.join(plots_dir, "plot2_min_distance.png"))

    # Plot 3: Completion Time
    print("  -> Plot 3: Completion Time...")
    plot_completion_time(df, os.path.join(plots_dir, "plot3_completion_time.png"))

    # Plot 4: Peak Acceleration & Jerk
    print("  -> Plot 4: Peak Accel & Jerk...")
    plot_accel_jerk(df, os.path.join(plots_dir, "plot4_accel_jerk.png"))

    # Plot 5: Failure Heatmap
    print("  -> Plot 5: Failure Heatmap...")
    plot_failure_heatmap(df, os.path.join(plots_dir, "plot5_failure_heatmap.png"))

    print("  -> Plot 6: Lane Discipline...")
    plot_lane_discipline(df, os.path.join(plots_dir, "plot6_lane_discipline.png"))

    print("  -> Plot 7: Maneuver Success...")
    plot_maneuver_success(df, os.path.join(plots_dir, "plot7_maneuver_success.png"))

    print("  -> Plot 8: MPPI Recovery...")
    plot_mppi_recovery(df, os.path.join(plots_dir, "plot8_mppi_recovery.png"))

    print(f"[REPORT] All 8 plots generated successfully in {plots_dir}/")


def generate_report(results_csv: str = None, raw_dir: str = None, plots_dir: str = None, summary_csv: str = None):
    """Generate reports from the requested result and output paths."""
    res_dir = os.path.dirname(results_csv) if results_csv else "results"
    generate_all_reports(res_dir, results_csv, raw_dir, plots_dir)


if __name__ == "__main__":
    generate_all_reports()

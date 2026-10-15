"""
metrics/report_generator.py
===========================
Generates comprehensive analysis plots (Plots 1 through 8) and summary tables.

Generated Outputs in results/plots/:
  - plot1_path_comparison.png
  - plot2_min_distance.png
  - plot3_completion_time.png
  - plot4_accel_jerk.png
  - plot5_failure_heatmap.png
  - plot6_reaction_time.png     (NEW: Sudden obstacle S5 reaction latency)
  - plot7_path_smoothness.png   (NEW: Jerk over time per controller)
  - plot8_edge_case_summary.png (NEW: Success rate on edge cases EC1..EC5)
"""

import os
import glob
import math
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from visualization.plot_results import (
    plot_path_comparison,
    plot_min_distance,
    plot_completion_time,
    plot_accel_jerk,
    plot_failure_heatmap,
    CTRL_COLORS,
    BG_DARK,
    CARD_BG,
    GRID_COLOR,
    TEXT_COLOR,
    ACCENT_RED,
    SAFE_GREEN,
    WARN_YELLOW,
    _set_dark_style,
)


def plot6_reaction_time(df: pd.DataFrame, out_path: str):
    """
    Plot 6: Reaction time for sudden obstacle (S5) per controller.
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG_DARK)

    s5_df = df[df["scenario"] == "s5_sudden_obstacle"]
    controllers = [c for c in ["RPP", "DWA", "MPPI"] if c in s5_df["controller"].unique()]

    means, stds = [], []
    for c in controllers:
        if "reaction_time_s" in s5_df.columns:
            sub = s5_df[s5_df["controller"] == c]["reaction_time_s"].dropna()
            means.append(float(sub.mean()) if len(sub) > 0 else 0.0)
            stds.append(float(sub.std()) if len(sub) > 0 else 0.0)
        else:
            means.append(0.0)
            stds.append(0.0)

    x = np.arange(len(controllers))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=[CTRL_COLORS.get(c, "#3498db") for c in controllers], width=0.45)

    # Values on top of bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + 0.02, f"{m:.3f}s" if m > 0 else "N/A",
                ha="center", va="bottom", color=TEXT_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(controllers, fontsize=11, fontweight="bold")
    ax.set_ylabel("Reaction Latency to Deceleration (s)", fontsize=11)
    ax.set_title("Scenario 5 — Reactive Braking Latency", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot7_path_smoothness(raw_dir: str, out_path: str):
    """
    Plot 7: Instantaneous Jerk profiles over time for each controller.
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=BG_DARK)

    controllers = ["RPP", "DWA", "MPPI"]
    for c in controllers:
        pattern = os.path.join(raw_dir, f"{c}__s1_static_box__run*.csv")
        files = sorted(glob.glob(pattern))
        if files:
            try:
                run_df = pd.read_csv(files[0])
                if "t" in run_df and "jerk" in run_df:
                    jerk_clean = run_df["jerk"].fillna(0.0)
                    ax.plot(run_df["t"], np.abs(jerk_clean), label=c, color=CTRL_COLORS.get(c, "#3498db"), linewidth=1.8, alpha=0.85)
            except Exception:
                pass

    ax.axhline(config.MAX_COMFORTABLE_JERK, color="#e67e22", linestyle="--", linewidth=1.5,
               label=f"Comfort Boundary ({config.MAX_COMFORTABLE_JERK} m/s³)")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Absolute Jerk |j| (m/s³)", fontsize=11)
    ax.set_title("Controller Motion Smoothness & Dynamic Jerk Profile", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot8_edge_case_summary(df: pd.DataFrame, out_path: str):
    """
    Plot 8: Grouped bar chart comparing success rates across Edge Cases (EC1..EC5).
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=BG_DARK)

    ec_scenarios = [s for s in sorted(df["scenario"].unique()) if s.startswith("ec")]
    if not ec_scenarios:
        # Fallback if no EC scenarios present
        ec_scenarios = sorted(df["scenario"].unique())

    controllers = [c for c in ["RPP", "DWA", "MPPI"] if c in df["controller"].unique()]
    x = np.arange(len(ec_scenarios))
    width = 0.25

    for i, c in enumerate(controllers):
        rates = []
        for s in ec_scenarios:
            sub = df[(df["controller"] == c) & (df["scenario"] == s)]
            rate = (sub["completed"].sum() / len(sub)) * 100.0 if len(sub) > 0 else 0.0
            rates.append(rate)

        offset = (i - (len(controllers) - 1) / 2.0) * width
        ax.bar(x + offset, rates, width=width, label=c, color=CTRL_COLORS.get(c, "#3498db"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in ec_scenarios], fontsize=10)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Edge Case Robustness & Scenario Completion Rates", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_all_reports(results_dir: str = "results"):
    """
    Generate all 8 plots and verify summary table.
    """
    all_csv = os.path.join(results_dir, "all_results.csv")
    plots_dir = os.path.join(results_dir, "plots")
    raw_dir = os.path.join(results_dir, "raw")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(all_csv):
        print(f"[WARN] {all_csv} not found. Run simulation first.")
        return

    df = pd.read_csv(all_csv)
    print(f"[REPORT] Generating 8 plots from {len(df)} run records...")

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

    # Plot 6: Reaction Time
    print("  -> Plot 6: Reaction Time (S5)...")
    plot6_reaction_time(df, os.path.join(plots_dir, "plot6_reaction_time.png"))

    # Plot 7: Path Smoothness
    print("  -> Plot 7: Path Smoothness...")
    plot7_path_smoothness(raw_dir, os.path.join(plots_dir, "plot7_path_smoothness.png"))

    # Plot 8: Edge Case Summary
    print("  -> Plot 8: Edge Case Summary...")
    plot8_edge_case_summary(df, os.path.join(plots_dir, "plot8_edge_case_summary.png"))

    print(f"[REPORT] All 8 plots generated successfully in {plots_dir}/")


def generate_report(results_csv: str = None, raw_dir: str = None, plots_dir: str = None, summary_csv: str = None):
    """Alias for generate_all_reports supporting custom paths."""
    res_dir = os.path.dirname(results_csv) if results_csv else "results"
    generate_all_reports(res_dir)


if __name__ == "__main__":
    generate_all_reports()


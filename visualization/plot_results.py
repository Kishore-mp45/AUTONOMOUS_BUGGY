"""
visualization/plot_results.py
=============================
Generates presentation-grade Matplotlib plots from experiment results.

Plots generated as per Implementation Plan Phase 6B:
1. Plot 1: Path Comparison (results/plots/plot1_path_comparison.png)
   - 5 subplots (one per scenario)
   - Overlays paths for RPP (blue), DWA (green), MPPI (orange)
   - Renders obstacles in red, start (star) and goal (bullseye)
2. Plot 2: Min Obstacle Distance (results/plots/plot2_min_distance.png)
   - Grouped bar / box plot across scenarios (s1..s5) for RPP, DWA, MPPI
   - Red dashed line at y = 0.3 m (safety collision boundary / ROBOT_RADIUS)
3. Plot 3: Completion Time (results/plots/plot3_completion_time.png)
   - Mean completion time ± std deviation for successful runs
   - Grouped by controller & scenario
4. Plot 4: Peak Acceleration & Jerk (results/plots/plot4_accel_jerk.png)
   - Dual subplots: Peak Acceleration (m/s²) and Peak Jerk (m/s³)
   - Lines per controller across scenarios
   - Passenger comfort threshold reference lines (1.5 m/s² and 3.0 m/s³)
5. Plot 5: Failure Heatmap (results/plots/plot5_failure_heatmap.png)
   - Controllers on Y-axis (RPP, DWA, MPPI), Scenarios on X-axis (s1..s5)
   - Matrix showing count of failures out of runs
   - Color scale: green = 0, yellow = 1..3, red = 4+
"""

import os
import glob
import math
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & batch generation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Color Standards ──────────────────────────────────────────────────────────
CTRL_COLORS = {
    "RPP":  "#2980b9",  # Strong Blue
    "DWA":  "#27ae60",  # Strong Green
    "MPPI": "#e67e22",  # Vivid Orange
}

BG_DARK      = "#131722"
CARD_BG      = "#1e2230"
GRID_COLOR   = "#2a3142"
TEXT_COLOR   = "#e0e6ed"
ACCENT_RED   = "#e74c3c"
SAFE_GREEN   = "#2ecc71"
WARN_YELLOW  = "#f1c40f"


def _set_dark_style():
    """Apply consistent, high-contrast dark theme to Matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "font.family": "sans-serif",
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": GRID_COLOR,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1: Path Comparison (5 subplots)
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_comparison(raw_csv_dir: str, out_path: str):
    """
    Renders 5 subplots showing vehicle trajectories for each controller.
    Scans raw CSV directory for time series files: {ctrl}__{scenario}__run*.csv
    """
    _set_dark_style()

    from scenarios.s1_static_box      import S1StaticBox
    from scenarios.s2_person_crossing import S2PersonCrossing
    from scenarios.s3_narrow_gap      import S3NarrowGap
    from scenarios.s4_two_obstacles   import S4TwoObstacles
    from scenarios.s5_sudden_obstacle import S5SuddenObstacle

    scenario_map = {
        "s1_static_box":      S1StaticBox(),
        "s2_person_crossing": S2PersonCrossing(),
        "s3_narrow_gap":      S3NarrowGap(),
        "s4_two_obstacles":   S4TwoObstacles(),
        "s5_sudden_obstacle": S5SuddenObstacle(),
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), facecolor=BG_DARK)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    scen_keys = list(scenario_map.keys())

    for idx, scen_key in enumerate(scen_keys):
        ax = axes[idx]
        scen = scenario_map[scen_key]
        ax.set_title(scen_key.replace("_", " ").title(), fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(0, config.MAP_W)
        ax.set_ylim(0, config.MAP_H)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        if idx == 0:
            ax.set_ylabel("Y (m)")

        # Draw obstacles
        obstacles = scen.get_obstacles()
        for obs in obstacles:
            if obs.get("type") == "circle" or "radius" in obs:
                r = obs.get("radius", 0.4)
                circle = patches.Circle((obs["x"], obs["y"]), r, facecolor=ACCENT_RED, edgecolor="#ff7979", alpha=0.8, zorder=3)
                ax.add_patch(circle)
            else:
                w = obs.get("w", 1.0)
                h = obs.get("h", 1.0)
                rect = patches.Rectangle((obs["x"] - w/2.0, obs["y"] - h/2.0), w, h, facecolor=ACCENT_RED, edgecolor="#ff7979", alpha=0.8, zorder=3)
                ax.add_patch(rect)

        # Draw start and goal
        ax.scatter([scen.start[0]], [scen.start[1]], c="#f1c40f", marker="*", s=180, label="Start", zorder=5)
        ax.scatter([scen.goal[0]],  [scen.goal[1]],  c="#2ecc71", marker="o", s=130, edgecolors="white", linewidths=1.5, label="Goal", zorder=5)

        # Plot representative trajectory for each controller (e.g. run00)
        for ctrl, col in CTRL_COLORS.items():
            pattern = os.path.join(raw_csv_dir, f"{ctrl}__{scen_key}__run*.csv")
            matches = sorted(glob.glob(pattern))
            if matches:
                # Plot the first run trajectory
                try:
                    df_run = pd.read_csv(matches[0])
                    if "x" in df_run and "y" in df_run:
                        ax.plot(df_run["x"], df_run["y"], color=col, linewidth=2.0, label=ctrl, alpha=0.9, zorder=4)
                except Exception:
                    pass

        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.85, fontsize=9)

    plt.suptitle("Campus Buggy Sim — Trajectory Path Comparison Across Scenarios", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: Minimum Obstacle Distance
# ═══════════════════════════════════════════════════════════════════════════

def plot_min_distance(df: pd.DataFrame, out_path: str):
    """
    Bar / Box comparison of minimum distance to obstacles per scenario and controller.
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG_DARK)

    scenarios = sorted(df["scenario"].unique())
    controllers = [c for c in ["RPP", "DWA", "MPPI"] if c in df["controller"].unique()]

    x = np.arange(len(scenarios))
    bar_width = 0.25

    for i, ctrl in enumerate(controllers):
        sub_df = df[df["controller"] == ctrl]
        means = []
        stds  = []
        for scen in scenarios:
            vals = sub_df[sub_df["scenario"] == scen]["min_obstacle_dist"].dropna()
            # replace 999 with NaN or clip
            vals = vals[vals < 50.0]
            means.append(vals.mean() if len(vals) else 0.0)
            stds.append(vals.std() if len(vals) else 0.0)

        offset = (i - (len(controllers) - 1) / 2.0) * bar_width
        ax.bar(
            x + offset, means, width=bar_width, yerr=stds,
            capsize=4, label=ctrl, color=CTRL_COLORS.get(ctrl, "#3498db"),
            alpha=0.85, edgecolor=GRID_COLOR
        )

    # Danger line
    ax.axhline(config.MIN_SAFE_DISTANCE, color=ACCENT_RED, linestyle="--", linewidth=1.8,
               label=f"Collision Threshold ({config.MIN_SAFE_DISTANCE}m)")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=10)
    ax.set_ylabel("Minimum Distance to Obstacle (m)", fontsize=11)
    ax.set_title("Minimum Clearance to Nearest Obstacle Across Scenarios", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3: Completion Time
# ═══════════════════════════════════════════════════════════════════════════

def plot_completion_time(df: pd.DataFrame, out_path: str):
    """
    Mean completion time ± standard deviation for completed runs.
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG_DARK)

    scenarios = sorted(df["scenario"].unique())
    controllers = [c for c in ["RPP", "DWA", "MPPI"] if c in df["controller"].unique()]

    x = np.arange(len(scenarios))
    bar_width = 0.25

    for i, ctrl in enumerate(controllers):
        sub_df = df[(df["controller"] == ctrl) & (df["completed"] == 1)]
        means = []
        stds  = []
        for scen in scenarios:
            vals = sub_df[sub_df["scenario"] == scen]["completion_time"].dropna()
            means.append(vals.mean() if len(vals) else 0.0)
            stds.append(vals.std() if len(vals) else 0.0)

        offset = (i - (len(controllers) - 1) / 2.0) * bar_width
        ax.bar(
            x + offset, means, width=bar_width, yerr=stds,
            capsize=4, label=ctrl, color=CTRL_COLORS.get(ctrl, "#3498db"),
            alpha=0.85, edgecolor=GRID_COLOR
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=10)
    ax.set_ylabel("Mean Completion Time (s)", fontsize=11)
    ax.set_title("Completion Time Comparison (Successful Runs)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4: Peak Acceleration & Jerk
# ═══════════════════════════════════════════════════════════════════════════

def plot_accel_jerk(df: pd.DataFrame, out_path: str):
    """
    Dual subplot line chart displaying peak acceleration and jerk with passenger comfort limits.
    """
    _set_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_DARK)

    scenarios = sorted(df["scenario"].unique())
    controllers = [c for c in ["RPP", "DWA", "MPPI"] if c in df["controller"].unique()]

    # Subplot 1: Peak Acceleration
    for ctrl in controllers:
        means = []
        for scen in scenarios:
            vals = df[(df["controller"] == ctrl) & (df["scenario"] == scen)]["peak_acceleration"].dropna()
            means.append(vals.mean() if len(vals) else 0.0)
        ax1.plot(scenarios, means, marker="o", linewidth=2.2, label=ctrl, color=CTRL_COLORS.get(ctrl, "#3498db"))

    ax1.axhline(config.MAX_COMFORTABLE_ACCEL, color="#e67e22", linestyle="--", linewidth=1.5,
                label=f"Comfort Limit ({config.MAX_COMFORTABLE_ACCEL} m/s²)")
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax1.set_ylabel("Peak Acceleration (m/s²)", fontsize=11)
    ax1.set_title("Peak Acceleration", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.85)

    # Subplot 2: Peak Jerk
    for ctrl in controllers:
        means = []
        for scen in scenarios:
            vals = df[(df["controller"] == ctrl) & (df["scenario"] == scen)]["peak_jerk"].dropna()
            means.append(vals.mean() if len(vals) else 0.0)
        ax2.plot(scenarios, means, marker="s", linewidth=2.2, label=ctrl, color=CTRL_COLORS.get(ctrl, "#3498db"))

    ax2.axhline(config.MAX_COMFORTABLE_JERK, color="#e67e22", linestyle="--", linewidth=1.5,
                label=f"Comfort Limit ({config.MAX_COMFORTABLE_JERK} m/s³)")
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax2.set_ylabel("Peak Jerk (m/s³)", fontsize=11)
    ax2.set_title("Peak Jerk (Rate of Acceleration Change)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.85)

    plt.suptitle("Passenger Ride Quality & Dynamic Smoothness", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Plot 5: Failure Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_failure_heatmap(df: pd.DataFrame, out_path: str):
    """
    Heatmap displaying count of failures out of runs per (Controller, Scenario).
    Green = 0, Yellow = 1..3, Red = 4+
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG_DARK)

    scenarios = sorted(df["scenario"].unique())
    controllers = [c for c in ["RPP", "DWA", "MPPI"] if c in df["controller"].unique()]

    # Build matrix
    matrix = np.zeros((len(controllers), len(scenarios)), dtype=int)
    for r, ctrl in enumerate(controllers):
        for c, scen in enumerate(scenarios):
            sub = df[(df["controller"] == ctrl) & (df["scenario"] == scen)]
            failures = int((sub["completed"] == 0).sum())
            matrix[r, c] = failures

    # Custom colormap: Green -> Yellow -> Red
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("fail_cmap", colors, N=100)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(4, int(matrix.max())), aspect="auto")

    # Annotations
    for r in range(len(controllers)):
        for c in range(len(scenarios)):
            val = matrix[r, c]
            ax.text(c, r, f"{val} fails", ha="center", va="center",
                    color="#131722" if val < 3 else "white", fontweight="bold", fontsize=11)

    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=10)
    ax.set_yticks(np.arange(len(controllers)))
    ax.set_yticklabels(controllers, fontsize=11, fontweight="bold")
    ax.set_title("Controller Failure Count Heatmap (Lower is Better)", fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Failures Count", color=TEXT_COLOR)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate all 5 plots and write to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(os.path.dirname(output_dir), "raw")

    print("  -> Generating Plot 1: Path Comparison...")
    plot_path_comparison(raw_dir, os.path.join(output_dir, "plot1_path_comparison.png"))

    print("  -> Generating Plot 2: Min Obstacle Distance...")
    plot_min_distance(df, os.path.join(output_dir, "plot2_min_distance.png"))

    print("  -> Generating Plot 3: Completion Time...")
    plot_completion_time(df, os.path.join(output_dir, "plot3_completion_time.png"))

    print("  -> Generating Plot 4: Peak Acceleration & Jerk...")
    plot_accel_jerk(df, os.path.join(output_dir, "plot4_accel_jerk.png"))

    print("  -> Generating Plot 5: Failure Heatmap...")
    plot_failure_heatmap(df, os.path.join(output_dir, "plot5_failure_heatmap.png"))


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== plot_results.py Self-Test ===")
    # Create synthetic test dataframe
    test_rows = []
    scens = ["s1_static_box", "s2_person_crossing", "s3_narrow_gap", "s4_two_obstacles", "s5_sudden_obstacle"]
    ctrls = ["RPP", "DWA", "MPPI"]
    for s in scens:
        for c in ctrls:
            for r in range(3):
                test_rows.append({
                    "controller": c,
                    "scenario": s,
                    "run_id": r,
                    "completed": 1 if (c != "RPP" or "gap" not in s) else 0,
                    "failure_reason": "" if (c != "RPP" or "gap" not in s) else "collision",
                    "completion_time": 18.2 + np.random.randn() * 0.5,
                    "path_length": 18.5,
                    "min_obstacle_dist": 0.45 + np.random.rand() * 0.3,
                    "peak_acceleration": 0.95,
                    "peak_jerk": 1.4,
                    "mean_speed": 1.1,
                    "final_x": 19.0,
                    "final_y": 10.0,
                })
    test_df = pd.DataFrame(test_rows)
    test_out = os.path.join("results", "test_plots")
    generate_all_plots(test_df, test_out)
    print(f"[PASS] All 5 plots generated into {test_out}")

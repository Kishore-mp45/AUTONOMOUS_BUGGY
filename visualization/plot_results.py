"""
visualization/plot_results.py
=============================
Generates presentation-grade MPPI-only Matplotlib plots from experiment results.

Plots generated as per Implementation Plan Phase 10:
1. Plot 1: Path Comparison (results/plots/plot1_path_comparison.png)
   - 5 subplots (one per scenario)
   - Overlays MPPI run paths and the lane geometry
2. Plot 2: Min Obstacle Distance (results/plots/plot2_min_distance.png)
   - Obstacle-clearance box plot across scenarios
   - Red dashed line at y = 0.3 m (safety collision boundary / ROBOT_RADIUS)
3. Plot 3: Completion Time (results/plots/plot3_completion_time.png)
   - Mean completion time ± std deviation for successful runs
   - Grouped by scenario
4. Plot 4: Peak Acceleration & Jerk (results/plots/plot4_accel_jerk.png)
   - Dual subplots: Peak Acceleration (m/s²) and Peak Jerk (m/s³)
   - Metrics per scenario
   - Passenger comfort threshold reference lines (1.5 m/s² and 3.0 m/s³)
5. Plot 5: Failure Heatmap (results/plots/plot5_failure_heatmap.png)
   - Scenarios on X-axis with per-scenario failure counts
6. Plot 6: Lane discipline — wrong-lane time per scenario
7. Plot 7: Maneuver success rate
8. Plot 8: MPPI recovery count per scenario
"""

import os
import glob
import math
import ast
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & batch generation
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Color Standards ──────────────────────────────────────────────────────────
CTRL_COLORS = {
    "MPPI": "#00d2ff",
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
    """Overlay up to ten MPPI runs per scenario, excluding detailed-log files."""
    _set_dark_style()
    files = sorted(
        path for path in glob.glob(os.path.join(raw_csv_dir, "MPPI__*__run*.csv"))
        if not path.endswith("_timesteps.csv")
    )
    scenario_files: Dict[str, List[str]] = {}
    for path in files:
        name = os.path.basename(path)
        scenario_name = name[len("MPPI__"):].split("__run", 1)[0]
        scenario_files.setdefault(scenario_name, []).append(path)

    scenarios = sorted(scenario_files) or ["No MPPI run data"]
    ncols = min(3, len(scenarios))
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 5.5 * nrows),
        facecolor=BG_DARK, squeeze=False,
    )

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_title(scenario.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlim(0, config.MAP_W)
        ax.set_ylim(0, config.MAP_H)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        if idx % ncols == 0:
            ax.set_ylabel("Y (m)")
        ax.axhspan(config.ROAD_Y_MIN, config.ROAD_Y_MAX, color="#253044", alpha=0.45)
        ax.axhline(config.ROAD_Y_MIN, color=ACCENT_RED, linewidth=1.2)
        ax.axhline(config.ROAD_Y_MAX, color=ACCENT_RED, linewidth=1.2)
        ax.axhline(config.ROAD_CENTERLINE_Y, color=WARN_YELLOW, linestyle="--", linewidth=1.0)
        ax.axhline(config.LEFT_LANE_CENTER_Y, color=SAFE_GREEN, linestyle=":", linewidth=1.0)

        if scenario == "No MPPI run data":
            ax.text(0.5, 0.5, "No MPPI run CSVs found", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_COLOR)
            continue
        for run_idx, path in enumerate(scenario_files[scenario][:10]):
            try:
                run_df = pd.read_csv(path)
            except (OSError, pd.errors.ParserError):
                continue
            if "x" not in run_df or "y" not in run_df:
                continue
            alpha = max(0.15, 0.8 - run_idx * 0.05)
            ax.plot(run_df["x"], run_df["y"], color=CTRL_COLORS["MPPI"],
                    linewidth=1.5, alpha=alpha,
                    label="MPPI run" if run_idx == 0 else None, zorder=4)
            if run_idx == 0 and len(run_df):
                ax.scatter(run_df["x"].iloc[0], run_df["y"].iloc[0],
                           c=WARN_YELLOW, marker="*", s=90, zorder=5, label="Start")
                ax.scatter(run_df["x"].iloc[-1], run_df["y"].iloc[-1],
                           c=SAFE_GREEN, marker="o", s=45, zorder=5, label="Final")
        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.85, fontsize=9)

    for idx in range(len(scenarios), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.suptitle("Campus Buggy — MPPI Trajectories by Scenario", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: Minimum Obstacle Distance
# ═══════════════════════════════════════════════════════════════════════════

def plot_min_distance(df: pd.DataFrame, out_path: str):
    """
    Box plot of minimum distance to obstacles across MPPI runs per scenario.
    """
    _set_dark_style()
    fig, ax = plt.subplots(figsize=(max(8, len(df["scenario"].unique()) * 0.9), 5.5), facecolor=BG_DARK)

    scenarios = sorted(df["scenario"].unique())
    data = []
    box_positions = []
    for scenario in scenarios:
        vals = pd.to_numeric(
            df.loc[df["scenario"] == scenario, "min_obstacle_dist"], errors="coerce"
        ).dropna()
        vals = vals[vals < 50.0].to_numpy()
        if len(vals):
            data.append(vals)
            box_positions.append(scenarios.index(scenario) + 1)
    positions = np.arange(1, len(scenarios) + 1)
    if data:
        box = ax.boxplot(data, positions=box_positions, patch_artist=True, showfliers=True)
        for patch in box["boxes"]:
            patch.set_facecolor(CTRL_COLORS["MPPI"])
            patch.set_alpha(0.75)
        for element in ("whiskers", "caps", "medians"):
            for artist in box[element]:
                artist.set_color(TEXT_COLOR)

    # Danger line
    ax.axhline(config.MIN_SAFE_DISTANCE, color=ACCENT_RED, linestyle="--", linewidth=1.8,
               label=f"Collision Threshold ({config.MIN_SAFE_DISTANCE}m)")

    ax.set_xticks(positions)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=10)
    ax.set_ylabel("Minimum Distance to Obstacle (m)", fontsize=11)
    ax.set_title("MPPI Minimum Clearance Across Runs", fontsize=14, fontweight="bold", pad=12)

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
    controllers = ["MPPI"] if "MPPI" in df["controller"].unique() else []

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
    ax.set_title("MPPI Completion Time (Successful Runs)", fontsize=14, fontweight="bold", pad=12)
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
    controllers = ["MPPI"] if "MPPI" in df["controller"].unique() else []

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
    controllers = ["MPPI"] if "MPPI" in df["controller"].unique() else ["MPPI"]

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
    ax.set_title("MPPI Failure Count by Scenario", fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Failures Count", color=TEXT_COLOR)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_lane_discipline(df: pd.DataFrame, out_path: str):
    """Plot cumulative wrong-lane duration per scenario, averaged per run."""
    _set_dark_style()
    scenarios = sorted(df["scenario"].dropna().unique())
    durations = []
    for scenario in scenarios:
        values = []
        for _, row in df[df["scenario"] == scenario].iterrows():
            raw = row.get("wrong_lane_durations", "[]")
            try:
                intervals = ast.literal_eval(raw) if isinstance(raw, str) else []
                total = sum(max(0.0, float(end) - float(start)) for start, end in intervals)
            except (ValueError, SyntaxError, TypeError):
                total = float(row.get("lane_violations", 0) or 0) * config.DT
            values.append(total)
        durations.append(float(np.mean(values)) if values else 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 0.9), 5.2), facecolor=BG_DARK)
    x = np.arange(len(scenarios))
    bars = ax.bar(x, durations, color=CTRL_COLORS["MPPI"], alpha=0.85)
    for bar, value in zip(bars, durations):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}s",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_ylabel("Mean time in wrong lane (s)")
    ax.set_title("MPPI Lane Discipline by Scenario")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_maneuver_success(df: pd.DataFrame, out_path: str):
    """Plot success percentages for maneuver outcomes explicitly recorded per run."""
    _set_dark_style()
    outcomes: Dict[str, List[bool]] = {}
    if "maneuver_success" in df.columns:
        for raw in df["maneuver_success"].dropna():
            try:
                values = ast.literal_eval(raw) if isinstance(raw, str) else {}
            except (ValueError, SyntaxError):
                continue
            if isinstance(values, dict):
                for maneuver, success in values.items():
                    outcomes.setdefault(str(maneuver), []).append(bool(success))

    fig, ax = plt.subplots(figsize=(8, 5.0), facecolor=BG_DARK)
    if outcomes:
        names = sorted(outcomes)
        rates = [100.0 * np.mean(outcomes[name]) for name in names]
        bars = ax.bar(names, rates, color=SAFE_GREEN, alpha=0.85)
        for bar, name in zip(bars, names):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.0f}% ({sum(outcomes[name])}/{len(outcomes[name])})",
                    ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Success rate (%)")
    else:
        ax.text(0.5, 0.5, "No maneuver outcomes recorded", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("MPPI Maneuver Success Rate")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_mppi_recovery(df: pd.DataFrame, out_path: str):
    """Plot mean and maximum MPPI recovery attempts per scenario."""
    _set_dark_style()
    scenarios = sorted(df["scenario"].dropna().unique())
    means, maxima = [], []
    for scenario in scenarios:
        values = pd.to_numeric(
            df.loc[df["scenario"] == scenario, "recovery_count_total"],
            errors="coerce",
        ).dropna() if "recovery_count_total" in df.columns else pd.Series(dtype=float)
        means.append(float(values.mean()) if len(values) else 0.0)
        maxima.append(float(values.max()) if len(values) else 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 0.9), 5.2), facecolor=BG_DARK)
    x = np.arange(len(scenarios))
    width = 0.38
    ax.bar(x - width / 2, means, width, label="Mean", color=CTRL_COLORS["MPPI"], alpha=0.9)
    ax.bar(x + width / 2, maxima, width, label="Maximum", color=WARN_YELLOW, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_ylabel("Recovery attempts per run")
    ax.set_title("MPPI Recovery Activity by Scenario")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate all 8 plots and write to the output directory.
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

    print("  -> Generating Plot 6: Lane Discipline...")
    plot_lane_discipline(df, os.path.join(output_dir, "plot6_lane_discipline.png"))

    print("  -> Generating Plot 7: Maneuver Success...")
    plot_maneuver_success(df, os.path.join(output_dir, "plot7_maneuver_success.png"))

    print("  -> Generating Plot 8: MPPI Recovery...")
    plot_mppi_recovery(df, os.path.join(output_dir, "plot8_mppi_recovery.png"))


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== plot_results.py Self-Test ===")
    # Create synthetic test dataframe
    test_rows = []
    scens = ["s1_static_box", "s2_person_crossing", "s3_narrow_gap", "s4_two_obstacles", "s5_sudden_obstacle"]
    ctrls = ["MPPI"]
    for s in scens:
        for c in ctrls:
            for r in range(3):
                test_rows.append({
                    "controller": c,
                    "scenario": s,
                    "run_id": r,
                    "completed": 1 if ("gap" not in s) else 0,
                    "failure_reason": "" if ("gap" not in s) else "collision",
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
    print(f"[PASS] All plots generated into {test_out}")

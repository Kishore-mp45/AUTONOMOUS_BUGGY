"""
main.py
========
Entry point for the Campus Buggy Local Planner Comparison experiment.

Usage:
    python main.py                     # run all combos, headless (fast)
    python main.py --viz               # run with live pygame visualisation
    python main.py --runs 3            # override RUNS_PER_COMBO (quick test)
    python main.py --ctrl RPP          # only run one controller
    python main.py --scenario s1       # only run matching scenarios
    python main.py --headless          # explicit headless (same as default)
    python main.py --no-plots          # skip post-run plot generation

Loop structure (as per implementation plan):
    for scenario in SCENARIOS:
        for controller in CONTROLLERS:
            for run_id in range(RUNS_PER_COMBO):
                scenario.reset()
                recorder = MetricsRecorder(...)
                run_simulation(...)
                all_results.append(recorder.to_dict())

    df = pd.DataFrame(all_results)
    df.to_csv('results/all_results.csv')

After all runs, summary_table.csv is generated and plots are produced.
"""

import os
import sys
import argparse
import time
import math

import numpy as np
import pandas as pd

# ── Path setup (allows running from any working directory) ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config

# ── Robot ───────────────────────────────────────────────────────────────
from robot.bicycle_model import BicycleModel, make_state

# ── Controllers ─────────────────────────────────────────────────────────
from controllers.rpp  import RegulatedPurePursuit
from controllers.dwa  import DWA
from controllers.mppi import MPPI

# ── Scenarios ───────────────────────────────────────────────────────────
from scenarios.s1_static_box      import S1StaticBox
from scenarios.s2_person_crossing import S2PersonCrossing
from scenarios.s3_narrow_gap      import S3NarrowGap
from scenarios.s4_two_obstacles   import S4TwoObstacles
from scenarios.s5_sudden_obstacle import S5SuddenObstacle

# ── Metrics ─────────────────────────────────────────────────────────────
from metrics.recorder import MetricsRecorder

# ── Results directories ─────────────────────────────────────────────────
_RESULTS_DIR = os.path.join(_HERE, "results")
_RAW_DIR     = os.path.join(_RESULTS_DIR, "raw")
_PLOTS_DIR   = os.path.join(_RESULTS_DIR, "plots")
_ALL_CSV     = os.path.join(_RESULTS_DIR, "all_results.csv")
_SUMMARY_CSV = os.path.join(_RESULTS_DIR, "summary_table.csv")

os.makedirs(_RAW_DIR,   exist_ok=True)
os.makedirs(_PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Controller & Scenario registries
# ═══════════════════════════════════════════════════════════════════════════

def build_controllers() -> dict:
    """Instantiate all three controllers from config defaults."""
    return {
        "RPP":  RegulatedPurePursuit(),
        "DWA":  DWA(),
        "MPPI": MPPI(),
    }


def build_scenarios() -> dict:
    """Instantiate all five scenarios."""
    return {
        "s1_static_box":      S1StaticBox(),
        "s2_person_crossing": S2PersonCrossing(),
        "s3_narrow_gap":      S3NarrowGap(),
        "s4_two_obstacles":   S4TwoObstacles(),
        "s5_sudden_obstacle": S5SuddenObstacle(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Core simulation loop — one run
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(
    scenario,
    controller,
    recorder:  MetricsRecorder,
    model:     BicycleModel,
    viz=None,
    ctrl_name: str = "",
    scenario_name: str = "",
    run_id:    int = 0,
) -> MetricsRecorder:
    """
    Execute one episode: one controller on one scenario, full metrics recording.

    Returns the completed MetricsRecorder.
    """
    state     = scenario.get_start_state().copy()
    t         = 0.0
    max_steps = int(config.TIME_LIMIT / config.DT)

    # Reset MPPI warm-start for each fresh run
    if hasattr(controller, "reset"):
        controller.reset()

    for _step in range(max_steps):

        # ── 1. Update scenario dynamics (moving obstacles / spawn) ──
        if hasattr(scenario, "_spawned"):          # S5SuddenObstacle
            scenario.update(t, robot_state=state)
        else:
            scenario.update(t)

        obstacles = scenario.get_obstacles()

        # ── 2. Record this timestep ─────────────────────────────────
        recorder.step(state, obstacles, t)

        # ── 3. Check terminal conditions BEFORE computing control ───
        if recorder.failed:
            recorder.finish(success=False, t=t)
            break

        if scenario.goal_reached(state):
            recorder.finish(success=True, t=t)
            break

        if scenario.out_of_bounds(state):
            recorder.mark_timeout(t)
            recorder.finish(success=False, t=t)
            break

        # ── 4. Compute control ─────────────────────────────────────
        try:
            delta, v_cmd = _compute_control(
                controller, state,
                goal=scenario.goal,
                path=scenario.get_path(),
                obstacles=obstacles,
            )
        except Exception as exc:
            print(f"    [WARN] Controller error at t={t:.2f}s: {exc}")
            delta, v_cmd = 0.0, 0.0   # safe stop

        # ── 5. Convert v_cmd -> acceleration ───────────────────────
        a = _v_cmd_to_accel(v_cmd, float(state[3]))

        # ── 6. Advance robot state ─────────────────────────────────
        state = model.update(state, np.array([delta, a]))
        t    += config.DT

        # ── 7. Live visualisation (optional) ───────────────────────
        if viz is not None:
            keep_running = viz.render(
                state=state,
                obstacles=obstacles,
                path=scenario.get_path(),
                goal=scenario.goal,
                ctrl_name=ctrl_name,
                scenario_name=scenario_name,
                run_id=run_id,
                t=t,
            )
            if not keep_running:
                raise KeyboardInterrupt("Pygame window closed.")

    else:
        # Loop exhausted without goal/failure => timeout
        recorder.mark_timeout(t)
        recorder.finish(success=False, t=t)

    # Safety net — ensure finish() was called
    if not recorder.completed and not recorder.failed:
        recorder.finish(success=False, t=t)

    return recorder


# ═══════════════════════════════════════════════════════════════════════════
# Controller dispatch helper
# ═══════════════════════════════════════════════════════════════════════════

def _compute_control(controller, state, goal, path, obstacles):
    """
    Unified dispatch — each controller has a slightly different call signature.
    All normalised here to return (steering_angle_rad, speed_setpoint_m/s).
    """
    if isinstance(controller, RegulatedPurePursuit):
        return controller.compute_control(state, path, obstacles)

    elif isinstance(controller, DWA):
        return controller.compute_control(state, goal, obstacles)

    elif isinstance(controller, MPPI):
        # MPPI returns (delta, acceleration).  Convert a -> v_next for uniformity.
        delta, a = controller.compute_control(state, path, obstacles)
        v_next   = float(np.clip(state[3] + a * config.DT, 0.0, config.MAX_SPEED))
        return delta, v_next

    else:
        raise TypeError(f"Unknown controller type: {type(controller)}")


def _v_cmd_to_accel(v_cmd: float, v_curr: float) -> float:
    """Convert a speed setpoint to an acceleration command, clipped to limits."""
    a = (v_cmd - v_curr) / config.DT
    return float(np.clip(a, -config.MAX_ACCEL, config.MAX_ACCEL))


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw per-run results into one row per controller x scenario.

    Columns:
        controller, scenario,
        completion_rate, avg_time_s, std_time_s,
        min_obs_dist_m, peak_accel, peak_jerk, failures
    """
    rows = []
    for (ctrl, scen), grp in df.groupby(["controller", "scenario"]):
        n_total    = len(grp)
        n_complete = int(grp["completed"].sum())
        failures   = n_total - n_complete
        done       = grp[grp["completed"] == 1]

        rows.append({
            "controller":      ctrl,
            "scenario":        scen,
            "completion_rate": f"{n_complete}/{n_total}",
            "avg_time_s":      round(done["completion_time"].mean(), 2) if len(done) else float("nan"),
            "std_time_s":      round(done["completion_time"].std(),  2) if len(done) else float("nan"),
            "min_obs_dist_m":  round(grp["min_obstacle_dist"].min(), 3),
            "peak_accel":      round(grp["peak_acceleration"].max(),  3),
            "peak_jerk":       round(grp["peak_jerk"].max(),          3),
            "failures":        failures,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Progress display
# ═══════════════════════════════════════════════════════════════════════════

def _progress_bar(done: int, total: int, width: int = 28) -> str:
    frac   = done / total if total > 0 else 0
    filled = int(width * frac)
    bar    = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done:3d}/{total}"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Campus Buggy — Local Planner Comparison Experiment"
    )
    parser.add_argument("--viz",      action="store_true",
                        help="Enable live pygame visualisation")
    parser.add_argument("--headless", action="store_true",
                        help="Force headless mode (no pygame)")
    parser.add_argument("--runs",     type=int, default=config.RUNS_PER_COMBO,
                        help=f"Runs per controller x scenario combo "
                             f"(default: {config.RUNS_PER_COMBO})")
    parser.add_argument("--ctrl",     type=str, default=None,
                        help="Only run this controller: RPP | DWA | MPPI")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Only run scenarios whose name contains this substring")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip post-run matplotlib plot generation")
    args = parser.parse_args()

    use_viz = args.viz and not args.headless and not config.HEADLESS
    runs    = args.runs

    # ── Instantiate everything ─────────────────────────────────────
    model       = BicycleModel()
    controllers = build_controllers()
    scenarios   = build_scenarios()

    # Filters
    if args.ctrl:
        key = args.ctrl.upper()
        if key not in controllers:
            print(f"[ERROR] Unknown controller '{key}'. "
                  f"Choose from: {list(controllers)}")
            sys.exit(1)
        controllers = {key: controllers[key]}

    if args.scenario:
        scenarios = {k: v for k, v in scenarios.items() if args.scenario in k}
        if not scenarios:
            print(f"[ERROR] No scenario matching '{args.scenario}'.")
            sys.exit(1)

    total_runs = len(scenarios) * len(controllers) * runs

    print(f"\n{'='*62}")
    print(f"  Campus Buggy  --  Local Planner Comparison")
    print(f"{'='*62}")
    print(f"  Controllers : {list(controllers.keys())}")
    print(f"  Scenarios   : {list(scenarios.keys())}")
    print(f"  Runs each   : {runs}")
    print(f"  Total runs  : {total_runs}")
    print(f"  Visualise   : {'YES (pygame)' if use_viz else 'NO (headless)'}")
    print(f"{'='*62}\n")

    # ── Optional pygame init ───────────────────────────────────────
    viz = None
    if use_viz:
        try:
            from visualization.pygame_viz import PygameViz
            viz = PygameViz()
        except Exception as e:
            print(f"[WARN] Pygame unavailable: {e}\n       Running headless.")
            viz = None

    # ── Main experiment loop ───────────────────────────────────────
    all_results = []
    run_count   = 0
    wall_start  = time.time()

    try:
        for scenario_name, scenario in scenarios.items():
            for ctrl_name, controller in controllers.items():

                outcomes = []

                for run_id in range(runs):
                    run_count += 1
                    scenario.reset()

                    recorder = MetricsRecorder(ctrl_name, scenario_name, run_id)

                    t_run_wall = time.time()

                    run_simulation(
                        scenario=scenario,
                        controller=controller,
                        recorder=recorder,
                        model=model,
                        viz=viz,
                        ctrl_name=ctrl_name,
                        scenario_name=scenario_name,
                        run_id=run_id,
                    )

                    elapsed_wall = time.time() - t_run_wall

                    # Store
                    result = recorder.to_dict()
                    all_results.append(result)

                    # Save per-run time-series CSV
                    raw_csv = os.path.join(
                        _RAW_DIR,
                        f"{ctrl_name}__{scenario_name}__run{run_id:02d}.csv"
                    )
                    recorder.save_csv(raw_csv)

                    # Outcome label
                    if recorder.completed:
                        tag = "OK"
                    else:
                        r = recorder.failure_reason or "?"
                        tag = r[:4].upper()
                    outcomes.append(tag)

                    # Print progress
                    bar = _progress_bar(run_count, total_runs)
                    print(
                        f"  {bar}  "
                        f"{ctrl_name:<4}|{scenario_name:<24}|"
                        f"run {run_id+1:2d}/{runs}  "
                        f"[{tag:<4}]  "
                        f"wall {elapsed_wall:.1f}s"
                    )

                # Per-combo summary line
                ok = outcomes.count("OK")
                print(f"    --> {ctrl_name}/{scenario_name}: "
                      f"{ok}/{runs} OK  {outcomes}\n")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving partial results...")

    finally:
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass

    # ── Save results ───────────────────────────────────────────────
    if not all_results:
        print("[WARN] No results collected.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(_ALL_CSV, index=False)
    print(f"\n[SAVED] {len(df)} rows  ->  {_ALL_CSV}")

    summary = build_summary_table(df)
    summary.to_csv(_SUMMARY_CSV, index=False)
    print(f"[SAVED] Summary      ->  {_SUMMARY_CSV}")

    # ── Console summary ────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  RESULTS SUMMARY")
    print(f"{'='*62}")
    print(summary.to_string(index=False))
    print(f"{'='*62}\n")

    # ── Generate plots ─────────────────────────────────────────────
    if not args.no_plots:
        try:
            from visualization.plot_results import generate_all_plots
            print("[PLOTS] Generating result plots...")
            generate_all_plots(df, _PLOTS_DIR)
            print(f"[PLOTS] Saved to {_PLOTS_DIR}/\n")
        except Exception as exc:
            print(f"[WARN] Plot generation failed: {exc}")
            import traceback
            traceback.print_exc()

    # ── Final wall-clock ───────────────────────────────────────────
    total_wall = time.time() - wall_start
    per_run    = total_wall / max(run_count, 1)
    print(f"[DONE] {run_count} runs in {total_wall:.1f}s  "
          f"({per_run:.2f}s/run avg)")
    print(f"       Results: {_RESULTS_DIR}/")


if __name__ == "__main__":
    main()

"""
main.py
========
Entry point for the Campus Buggy Local Planner Comparison experiment.

Usage:
    python main.py                     # run all core combos, headless
    python main.py --viz               # run with live pygame visualisation
    python main.py --runs 3            # override RUNS_PER_COMBO (quick test)
    python main.py --ctrl RPP          # only run one controller
    python main.py --scenario s1       # only run matching scenarios
    python main.py --headless          # explicit headless
    python main.py --edge-cases        # run edge case scenarios alongside core scenarios
    python main.py --edge-only         # run ONLY edge case scenarios
    python main.py --report            # automatically run report generator after simulation
    python main.py --no-plots          # skip post-run plot generation

Loop structure (as per implementation plan Phase G):
    validate()
    for scenario in SCENARIOS:
        for controller in CONTROLLERS:
            for run_id in range(RUNS_PER_COMBO):
                seed = BASE_SEED + run_id
                scenario.reset(seed=seed)
                controller.reset()
                monitor.reset()
                costmap = scenario.build_costmap()
                ...
                control = controller.compute_control(state, path, goal, costmap, obstacles)
                status, msg = monitor.check(state, control, costmap)
                ...
                recorder.save_timestep_csv()
"""

import os
import sys
import argparse
import time
import math
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ── Path setup (allows running from any working directory) ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config
config.validate()

# ── Robot ───────────────────────────────────────────────────────────────
from robot.bicycle_model import BicycleModel

# ── Controllers ─────────────────────────────────────────────────────────
from controllers.rpp  import RegulatedPurePursuit
from controllers.dwa  import DWA
from controllers.mppi import MPPI

# ── Health Monitor ──────────────────────────────────────────────────────
from health.monitor import HealthMonitor

# ── Scenarios ───────────────────────────────────────────────────────────
from scenarios.s1_static_box      import S1StaticBox
from scenarios.s2_person_crossing import S2PersonCrossing
from scenarios.s3_narrow_gap      import S3NarrowGap
from scenarios.s4_two_obstacles   import S4TwoObstacles
from scenarios.s5_sudden_obstacle import S5SuddenObstacle

# ── Edge Case Scenarios ─────────────────────────────────────────────────
from scenarios.edge_cases import (
    EC1DeadEnd,
    EC2OscillationTrigger,
    EC3TightTurn,
    EC4LateBraking,
    EC5Crowd,
)

# ── Metrics ─────────────────────────────────────────────────────────────
from metrics.recorder import MetricsRecorder

# ── Results directories ─────────────────────────────────────────────────
_RESULTS_DIR   = os.path.join(_HERE, "results")
_RAW_DIR       = os.path.join(_RESULTS_DIR, "raw")
_PLOTS_DIR     = os.path.join(_RESULTS_DIR, "plots")
_EDGE_DIR      = os.path.join(_RESULTS_DIR, "edge_case_results")
_ALL_CSV       = os.path.join(_RESULTS_DIR, "all_results.csv")
_SUMMARY_CSV   = os.path.join(_RESULTS_DIR, "summary_table.csv")
_EDGE_ALL_CSV  = os.path.join(_EDGE_DIR, "all_edge_results.csv")

os.makedirs(_RAW_DIR,   exist_ok=True)
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_EDGE_DIR,  exist_ok=True)


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


def build_core_scenarios() -> dict:
    """Instantiate the 5 core benchmark scenarios."""
    return {
        "s1_static_box":      S1StaticBox(),
        "s2_person_crossing": S2PersonCrossing(),
        "s3_narrow_gap":      S3NarrowGap(),
        "s4_two_obstacles":   S4TwoObstacles(),
        "s5_sudden_obstacle": S5SuddenObstacle(),
    }


def build_edge_scenarios() -> dict:
    """Instantiate the 5 edge case scenarios."""
    return {
        "ec1_dead_end":         EC1DeadEnd(),
        "ec2_oscillation":      EC2OscillationTrigger(),
        "ec3_tight_turn":       EC3TightTurn(),
        "ec4_late_braking":     EC4LateBraking(),
        "ec5_crowd":            EC5Crowd(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Core simulation loop — one run
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(
    scenario,
    controller,
    recorder:   MetricsRecorder,
    model:      BicycleModel,
    monitor:    HealthMonitor,
    costmap,
    viz=None,
    ctrl_name: str = "",
    scenario_name: str = "",
    run_id:    int = 0,
) -> MetricsRecorder:
    """
    Execute one episode: one controller on one scenario, full metrics recording.
    """
    state     = scenario.get_start_state().copy()
    max_steps = int(scenario.time_limit / config.DT)

    # Store scenario metadata on recorder if available
    if hasattr(scenario, "gap_width"):
        recorder.gap_width = getattr(scenario, "gap_width")

    for step_i in range(max_steps):
        t = step_i * config.DT

        # ── 1. Update scenario dynamics (moving obstacles / spawn) ──
        if hasattr(scenario, "_spawned") or isinstance(scenario, S5SuddenObstacle):
            scenario.update(t, robot_state=state)
            if hasattr(scenario, "spawn_timestep") and scenario.spawn_timestep is not None:
                recorder.record_spawn_event(scenario.spawn_timestep)
        else:
            scenario.update(t)

        obstacles = scenario.get_obstacles()
        # Re-stamp the costmap with current obstacle positions. On the S5 spawn
        # frame scenario.update() already did this once; the extra call here
        # guarantees the inflation layer is current for all other scenarios too.
        costmap.update(obstacles)

        # ── 2. Unified controller execution ────────────────────────
        path = scenario.get_path()
        goal = scenario.goal

        try:
            steer, v_cmd = controller.compute_control(
                state=state,
                path=path,
                goal=goal,
                costmap=costmap,
                obstacles=obstacles,
            )
        except TypeError:
            # Fallback for any legacy signature if present
            steer, v_cmd = _legacy_compute_control(controller, state, goal, path, obstacles)
        except Exception as exc:
            print(f"    [WARN] Controller error at t={t:.2f}s: {exc}")
            steer, v_cmd = 0.0, 0.0

        control = (steer, v_cmd)

        # ── 3. Health check ─────────────────────────────────────────
        status, msg = monitor.check(state, control, costmap)
        if status == "collision":
            recorder.finish(success=False, t=t, reason="collision")
            break
        elif status == "stuck":
            recorder.finish(success=False, t=t, reason="stuck")
            break
        elif status == "oscillating":
            recorder.oscillations += 1

        # ── 4. Record this timestep ─────────────────────────────────
        recorder.step(
            state=state,
            obstacles=obstacles,
            t=t,
            control=control,
            costmap=costmap,
        )

        # ── 5. Check termination conditions ─────────────────────────
        if recorder.failed:
            recorder.finish(success=False, t=t)
            break

        done, success, reason = scenario.is_complete(state, t)
        if done:
            recorder.finish(success=success, t=t, reason=reason)
            break

        # ── 6. Convert v_cmd -> acceleration and step robot ─────────
        a = _v_cmd_to_accel(v_cmd, float(state[3]))
        state = model.update(state, np.array([steer, a]))

        # ── 7. Live visualisation (optional) ───────────────────────
        if viz is not None:
            is_spawn_step = (
                hasattr(scenario, "spawn_timestep") and
                scenario.spawn_timestep is not None and
                step_i == scenario.spawn_timestep
            )
            keep_running = viz.render(
                state=state,
                obstacles=obstacles,
                path=scenario.get_path(),
                goal=scenario.goal,
                ctrl_name=ctrl_name,
                scenario_name=scenario_name,
                run_id=run_id,
                t=t,
                costmap=costmap,
                controller=controller,
                health_status=status,
                health_msg=msg,
                spawn_event=is_spawn_step,
            )
            if not keep_running:
                raise KeyboardInterrupt("Pygame window closed.")

    else:
        # Loop exhausted without goal/failure => timeout
        t_final = max_steps * config.DT
        recorder.mark_timeout(t_final)
        recorder.finish(success=False, t=t_final, reason="timeout")

    # Safety net — ensure finish() was called
    if not recorder.completed and not recorder.failed:
        recorder.finish(success=False, t=max_steps * config.DT)

    # Save structured per-step CSV
    recorder.save_timestep_csv(_RAW_DIR)

    return recorder


def _legacy_compute_control(controller, state, goal, path, obstacles):
    if isinstance(controller, RegulatedPurePursuit):
        return controller.compute_control(state, path, obstacles)
    elif isinstance(controller, DWA):
        return controller.compute_control(state, goal, obstacles)
    elif isinstance(controller, MPPI):
        delta, a = controller.compute_control(state, path, obstacles)
        v_next   = float(np.clip(state[3] + a * config.DT, 0.0, config.MAX_SPEED))
        return delta, v_next
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
    """
    rows = []
    for (ctrl, scen), grp in df.groupby(["controller", "scenario"]):
        n_total    = len(grp)
        n_complete = int(grp["completed"].sum())
        failures   = n_total - n_complete
        done       = grp[grp["completed"] == 1]

        react_mean = None
        if "reaction_time_s" in grp.columns:
            valid_react = grp["reaction_time_s"].dropna()
            if len(valid_react) > 0:
                react_mean = round(valid_react.mean(), 3)

        rows.append({
            "controller":      ctrl,
            "scenario":        scen,
            "completion_rate": f"{n_complete}/{n_total}",
            "avg_time_s":      round(done["completion_time"].mean(), 2) if len(done) else float("nan"),
            "std_time_s":      round(done["completion_time"].std(),  2) if len(done) else float("nan"),
            "min_obs_dist_m":  round(grp["min_obstacle_dist"].min(), 3),
            "peak_accel":      round(grp["peak_acceleration"].max(),  3),
            "peak_jerk":       round(grp["peak_jerk"].max(),          3),
            "reaction_time_s": react_mean if react_mean is not None else "-",
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
    parser.add_argument("--viz",        action="store_true",
                        help="Enable live pygame visualisation")
    parser.add_argument("--headless",   action="store_true",
                        help="Force headless mode (no pygame)")
    parser.add_argument("--runs",       type=int, default=config.RUNS_PER_COMBO,
                        help=f"Runs per controller x scenario combo (default: {config.RUNS_PER_COMBO})")
    parser.add_argument("--ctrl",       type=str, default=None,
                        help="Only run this controller: RPP | DWA | MPPI")
    parser.add_argument("--scenario",   type=str, default=None,
                        help="Only run scenarios whose name contains this substring")
    parser.add_argument("--edge-cases", action="store_true",
                        help="Include the 5 edge case scenarios in the simulation runs")
    parser.add_argument("--edge-only",  action="store_true",
                        help="Run ONLY the 5 edge case scenarios")
    parser.add_argument("--report",     action="store_true",
                        help="Generate 8-plot comparison report after runs")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip post-run matplotlib plot generation")
    args = parser.parse_args()

    use_viz = args.viz and not args.headless and not config.HEADLESS
    runs    = args.runs

    # ── Instantiate controllers & models ───────────────────────────
    model       = BicycleModel()
    monitor     = HealthMonitor()
    controllers = build_controllers()

    # Determine which scenarios to run
    if args.edge_only:
        scenarios = build_edge_scenarios()
        is_edge_experiment = True
    elif args.edge_cases:
        scenarios = {**build_core_scenarios(), **build_edge_scenarios()}
        is_edge_experiment = False
    else:
        scenarios = build_core_scenarios()
        is_edge_experiment = False

    # Filters
    if args.ctrl:
        key = args.ctrl.upper()
        if key not in controllers:
            print(f"[ERROR] Unknown controller '{key}'. Choose from: {list(controllers)}")
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

                    # Seed-based reproducibility (Phase G specification)
                    seed = config.BASE_SEED + run_id
                    scenario.reset(seed=seed)
                    controller.reset()
                    monitor.reset()
                    model.reset()
                    # Use the scenario's own costmap so that scenario.update()
                    # and the controller always see the same Costmap2D object.
                    costmap = scenario.costmap
                    if costmap is None:
                        costmap = scenario.build_costmap()
                        scenario.costmap = costmap

                    recorder = MetricsRecorder(ctrl_name, scenario_name, run_id)

                    t_run_wall = time.time()

                    run_simulation(
                        scenario=scenario,
                        controller=controller,
                        recorder=recorder,
                        model=model,
                        monitor=monitor,
                        costmap=costmap,
                        viz=viz,
                        ctrl_name=ctrl_name,
                        scenario_name=scenario_name,
                        run_id=run_id,
                    )

                    elapsed_wall = time.time() - t_run_wall

                    # Store
                    result = recorder.to_dict()
                    all_results.append(result)

                    # Save per-run legacy time-series CSV as well
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

    target_csv = _EDGE_ALL_CSV if is_edge_experiment else _ALL_CSV
    df.to_csv(target_csv, index=False)
    print(f"\n[SAVED] {len(df)} rows  ->  {target_csv}")

    summary = build_summary_table(df)
    summary_path = os.path.join(_EDGE_DIR, "edge_summary_table.csv") if is_edge_experiment else _SUMMARY_CSV
    summary.to_csv(summary_path, index=False)
    print(f"[SAVED] Summary      ->  {summary_path}")

    # ── Console summary ────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  RESULTS SUMMARY")
    print(f"{'='*62}")
    print(summary.to_string(index=False))
    print(f"{'='*62}\n")

    # ── Generate plots / reports ───────────────────────────────────
    if (args.report or not args.no_plots) and not is_edge_experiment:
        try:
            from metrics.report_generator import generate_report
            print("[REPORT] Generating all 8 comparison plots & summary...")
            generate_report(_ALL_CSV, _RAW_DIR, _PLOTS_DIR, _SUMMARY_CSV)
            print(f"[REPORT] Saved to {_PLOTS_DIR}/\n")
        except Exception as exc:
            print(f"[WARN] Report generation failed: {exc}")
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


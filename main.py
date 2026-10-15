"""
main.py
========
Entry point for the Campus Buggy Local Planner Comparison experiment.

Usage:
    python main.py                     # run all core combos, headless
    python main.py --viz               # run with live pygame visualisation
    python main.py --runs 3            # override RUNS_PER_COMBO (quick test)
    python main.py --ctrl MPPI         # explicit sole controller
    python main.py --scenario s1       # only run matching scenarios
    python main.py --scenarios s1 s3   # run multiple matching scenarios
    python main.py --headless          # explicit headless
    python main.py --edge-cases        # run edge case scenarios alongside core scenarios
    python main.py --edge-only         # run ONLY edge case scenarios
    python main.py --report            # automatically run report generator after simulation
    python main.py --no-plots          # skip post-run plot generation

Loop structure:
    validate()
    for scenario in SCENARIOS:
        for run_id in range(RUNS_PER_COMBO):
            reset MPPI, maneuver state machine, monitor, model, and scenario
            run one MPPI episode and record its metrics
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
from controllers.mppi import MPPI
from maneuvers.maneuver_state_machine import ManeuverStateMachine

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
    EC3TightTurn,
    EC4LateBraking,
    EC5Crowd,
    EC6WrongLaneEntry,
    EC7UTurnBlocked,
    EC8MissingLaneMark,
    EC9SensorNoise,
    EC10AllSamplesCollide,
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
        "ec3_tight_turn":       EC3TightTurn(),
        "ec4_late_braking":     EC4LateBraking(),
        "ec5_crowd":            EC5Crowd(),
        "ec6_wrong_lane_entry": EC6WrongLaneEntry(),
        "ec7_u_turn_blocked":   EC7UTurnBlocked(),
        "ec8_missing_lane_mark": EC8MissingLaneMark(),
        "ec9_sensor_noise":     EC9SensorNoise(),
        "ec10_all_samples_collide": EC10AllSamplesCollide(),
    }


def select_scenarios(scenarios: dict, requests: list) -> dict:
    """Select scenarios whose keys exactly match or contain a requested name."""
    selected_keys = set()
    for request in requests:
        matches = [key for key in scenarios if request.lower() in key.lower()]
        if not matches:
            raise ValueError(f"No scenario matches '{request}'.")
        selected_keys.update(matches)
    return {key: scenario for key, scenario in scenarios.items() if key in selected_keys}


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
    Execute one MPPI episode with lane, maneuver, and health monitoring.
    """
    state     = scenario.get_start_state().copy()
    max_steps = int(scenario.time_limit / config.DT)
    goal = tuple(scenario.goal)
    recorder.goal_adjusted = False
    if costmap is not None and costmap.query(*goal) >= config.INSCRIBED_COST:
        projected_goal = costmap.project_goal_to_free(*goal)
        if costmap.query(*projected_goal) >= config.INSCRIBED_COST:
            recorder.finish(success=False, t=0.0, reason="goal_unreachable")
            recorder.save_timestep_csv(_RAW_DIR)
            return recorder
        goal = projected_goal
        recorder.goal_adjusted = True

    maneuver  = ManeuverStateMachine(controller)
    maneuver.reset()
    controller.lane_guidance_enabled = not scenario.disable_lane_cost

    # Store scenario metadata on recorder if available
    if hasattr(scenario, "gap_width"):
        recorder.gap_width = getattr(scenario, "gap_width")

    for step_i in range(max_steps):
        t = step_i * config.DT

        # ── 1. Update scenario dynamics (moving obstacles / spawn) ──
        previously_active = {
            id(obstacle) for obstacle in scenario.get_obstacles()
            if obstacle.active
        }
        if hasattr(scenario, "_spawned") or isinstance(scenario, S5SuddenObstacle):
            scenario.update(t, robot_state=state)
            if hasattr(scenario, "spawn_timestep") and scenario.spawn_timestep is not None:
                recorder.record_spawn_event(scenario.spawn_timestep)
        else:
            scenario.update(t)

        recorder.spawn_adjustment_count += len(
            scenario.clear_spawn_overlaps(state, previously_active)
        )
        obstacles = scenario.get_obstacles()
        # Re-stamp the costmap with current obstacle positions. On the S5 spawn
        # frame scenario.update() already did this once; the extra call here
        # guarantees the inflation layer is current for all other scenarios too.
        costmap.update(obstacles)

        # ── 2. State estimation and maneuver update ────────────────
        controller_state = state.copy()
        if scenario.sensor_noise is not None:
            position_std, heading_std = scenario.sensor_noise
            controller_state[:2] += np.random.normal(0.0, position_std, size=2)
            controller_state[2] += np.random.normal(0.0, heading_std)

        path = scenario.get_path()
        scenario_command = scenario.get_maneuver_command(t, state)
        previous_maneuver_state = maneuver.state
        maneuver.update(controller_state, path, costmap, scenario_command, t)
        if previous_maneuver_state == "IN_UTURN" and maneuver.state == "RETURN_LEFT":
            recorder.record_maneuver_result("u_turn", True)
        elif previous_maneuver_state == "IN_LEFT" and maneuver.state == "NORMAL":
            recorder.record_maneuver_result("left_turn", True)
        elif previous_maneuver_state == "IN_RIGHT" and maneuver.state == "RETURN_LEFT":
            recorder.record_maneuver_result("right_turn", True)
        if maneuver._last_completed_maneuver is not None:
            recorder.record_maneuver_result(maneuver._last_completed_maneuver, True)
        if maneuver.failure_reason:
            recorder.record_maneuver_result(maneuver.failed_maneuver, False)
            recorder.finish(success=False, t=t, reason=maneuver.failure_reason)
            break

        # ── 3. MPPI controller execution ───────────────────────────
        steer, v_cmd = controller.compute_control(
            state=controller_state,
            path=path,
            goal=goal,
            costmap=costmap,
            obstacles=obstacles,
        )

        control = (steer, v_cmd)
        recorder.record_recovery_count_total(controller._recovery_count_total)
        if controller.emergency_stop:
            failure_reason = (
                "dead_end" if scenario.name == "ec1_dead_end"
                else "all_samples_collide"
            )
            recorder.finish(success=False, t=t, reason=failure_reason)
            break

        # ── 4. Health checks ───────────────────────────────────────
        lane_status, lane_msg = monitor.check_lane_discipline(
            state, controller.maneuver_mode, controller.lane_map
        )
        boundary_status, boundary_msg = monitor.check_boundary(state)
        status, msg = monitor.check(state, control, costmap)
        recovery_status, recovery_msg = monitor.check_mppi_recovery(
            controller._recovery_count
        )
        if boundary_status == "boundary_exit":
            recorder.finish(success=False, t=t, reason="boundary_exit")
            break
        if status == "collision":
            reason = (
                "dead_end"
                if scenario.name == "ec1_dead_end" and controller._recovery_count_total > 0
                else "collision"
            )
            recorder.finish(success=False, t=t, reason=reason)
            break
        elif status == "stuck":
            recorder.finish(success=False, t=t, reason="stuck")
            break

        # ── 5. Record this timestep ─────────────────────────────────
        recorder.step(
            state=state,
            obstacles=obstacles,
            t=t,
            control=control,
            costmap=costmap,
            lane_map=getattr(controller, "lane_map", None),
            maneuver_mode=getattr(controller, "maneuver_mode", "normal"),
            mppi_recovery_count=getattr(controller, "_recovery_count_total", 0),
            reference_path=path,
            lane_guidance_enabled=controller.lane_guidance_enabled,
            narrow_gap_violation=(
                recorder.gap_width is not None and lane_status == "lane_violation"
            ),
            weight_fallback_count=controller.weight_fallback_count,
            ackermann_clamp_count=controller.ackermann_clamp_count,
            estimated_state=controller_state,
        )

        # ── 6. Check termination conditions ─────────────────────────
        if recorder.failed:
            recorder.finish(success=False, t=t)
            break

        done, success, reason = scenario.is_complete(state, t, goal=goal)
        if done:
            recorder.finish(success=success, t=t, reason=reason)
            break

        # ── 7. Convert v_cmd -> acceleration and step robot ────────
        a = _v_cmd_to_accel(v_cmd, float(state[3]))
        state = model.update(state, np.array([steer, a]))

        # ── 8. Live visualisation (optional) ───────────────────────
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
                health_status=(
                    boundary_status if boundary_status != "ok" else
                    recovery_status if recovery_status != "ok" else
                    lane_status if lane_status != "ok" else status
                ),
                health_msg=(
                    boundary_msg if boundary_msg else
                    recovery_msg if recovery_msg else
                    lane_msg if lane_msg else msg
                ),
                spawn_event=is_spawn_step,
                maneuver_mode=controller.maneuver_mode,
                recovery_count=controller._recovery_count,
                lane_violations=monitor.lane_violation_count,
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

    active_maneuvers = {
        "IN_LEFT": "left_turn",
        "PRE_RIGHT": "right_turn",
        "IN_RIGHT": "right_turn",
        "OVERTAKING": "overtake",
        "IN_UTURN": "u_turn",
    }
    unfinished_maneuver = active_maneuvers.get(maneuver.state)
    if unfinished_maneuver and unfinished_maneuver not in recorder.maneuver_success:
        recorder.record_maneuver_result(unfinished_maneuver, False)

    # Save structured per-step CSV
    recorder.save_timestep_csv(_RAW_DIR)

    return recorder

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
        description="Campus Buggy — MPPI Autonomous Navigation Experiment"
    )
    parser.add_argument("--viz",        action="store_true",
                        help="Enable live pygame visualisation")
    parser.add_argument("--headless",   action="store_true",
                        help="Force headless mode (no pygame)")
    parser.add_argument("--runs",       type=int, default=config.RUNS_PER_COMBO,
                        help=f"Runs per scenario (default: {config.RUNS_PER_COMBO})")
    parser.add_argument("--ctrl",       type=str, default=None,
                        help="Only run this controller: MPPI")
    parser.add_argument("--scenario",   type=str, default=None,
                        help="Only run scenarios whose name contains this substring (legacy filter)")
    parser.add_argument("--scenarios",  nargs="+", default=None,
                        help="Run scenarios matching any of these names (e.g. s1 s3 s5)")
    parser.add_argument("--edge-cases", action="store_true",
                        help="Include the edge case scenarios in the simulation runs")
    parser.add_argument("--edge-only",  action="store_true",
                        help="Run ONLY the edge case scenarios")
    parser.add_argument("--report",     action="store_true",
                        help="Generate 8-plot comparison report after runs")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip post-run matplotlib plot generation")
    parser.add_argument("--vis-mppi-samples", action="store_true",
                        help="Show MPPI sampled trajectories in pygame visualization")
    args = parser.parse_args()

    use_viz = args.viz and not args.headless and not config.HEADLESS
    runs    = args.runs
    if runs < 1:
        parser.error("--runs must be at least 1")

    # ── Instantiate sole controller & models ───────────────────────
    model       = BicycleModel()
    monitor     = HealthMonitor()
    controller  = MPPI()
    ctrl_name   = controller.name

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

    if args.ctrl and args.ctrl.upper() != "MPPI":
        parser.error(f"Only MPPI is supported; received controller '{args.ctrl}'.")

    # Select specific scenarios, supporting short names such as "s1" and "ec7".
    if args.scenarios:
        try:
            scenarios = select_scenarios(scenarios, args.scenarios)
        except ValueError as exc:
            parser.error(str(exc))
    if args.scenario:
        scenarios = {k: v for k, v in scenarios.items() if args.scenario in k}
        if not scenarios:
            print(f"[ERROR] No scenario matching '{args.scenario}'.")
            sys.exit(1)

    edge_names = set(build_edge_scenarios())
    is_edge_experiment = bool(scenarios) and set(scenarios).issubset(edge_names)
    total_runs = len(scenarios) * runs

    print(f"\n{'='*62}")
    print(f"  Campus Buggy  --  MPPI Autonomous Navigation")
    print(f"{'='*62}")
    print(f"  Controller  : {ctrl_name}")
    print(f"  Scenarios   : {list(scenarios.keys())}")
    print(f"  Runs each   : {runs}")
    print(f"  Total runs  : {total_runs}")
    print(f"  Visualise   : {'YES (pygame)' if use_viz else 'NO (headless)'}")
    if args.vis_mppi_samples and not use_viz:
        print("  Note        : --vis-mppi-samples requires --viz; sample drawing is disabled.")
    print(f"{'='*62}\n")

    # ── Optional pygame init ───────────────────────────────────────
    viz = None
    if use_viz:
        try:
            from visualization.pygame_viz import PygameViz
            viz = PygameViz()
            viz.show_mppi_samples = args.vis_mppi_samples
        except Exception as e:
            print(f"[WARN] Pygame unavailable: {e}\n       Running headless.")
            viz = None

    # ── Main experiment loop ───────────────────────────────────────
    all_results = []
    run_count   = 0
    wall_start  = time.time()

    try:
        for scenario_name, scenario in scenarios.items():
            outcomes = []

            for run_id in range(runs):
                run_count += 1

                seed = config.BASE_SEED + run_id
                scenario.reset(seed=seed)
                controller.reset()
                monitor.reset()
                model.reset()
                costmap = scenario.costmap
                if costmap is None:
                    costmap = scenario.build_costmap()
                    scenario.costmap = costmap
                costmap.add_lane_boundaries()

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
                all_results.append(recorder.to_dict())

                raw_csv = os.path.join(
                    _RAW_DIR,
                    f"{ctrl_name}__{scenario_name}__run{run_id:02d}.csv"
                )
                recorder.save_csv(raw_csv)

                tag = "OK" if recorder.completed else (recorder.failure_reason or "?")[:4].upper()
                outcomes.append(tag)

                bar = _progress_bar(run_count, total_runs)
                print(
                    f"  {bar}  "
                    f"{ctrl_name:<4}|{scenario_name:<24}|"
                    f"run {run_id+1:2d}/{runs}  "
                    f"[{tag:<4}]  "
                    f"wall {elapsed_wall:.1f}s"
                )

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

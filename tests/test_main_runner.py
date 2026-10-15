"""Tests for scenario selection and runner integrations."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.mppi import MPPI
from main import run_simulation, select_scenarios
from maneuvers.maneuver_state_machine import ManeuverStateMachine
from maneuvers.u_turn import UTurnManeuver
from scenarios.edge_cases.ec4_late_braking import EC4LateBraking
from environment.costmap import Costmap2D
from environment.obstacle import Obstacle
from health.monitor import HealthMonitor
from metrics.recorder import MetricsRecorder
from robot.bicycle_model import BicycleModel
from scenarios.edge_cases.ec1_dead_end import EC1DeadEnd
from scenarios.s4_two_obstacles import S4TwoObstacles
from scenarios.s1_static_box import S1StaticBox
from scenarios.edge_cases.ec7_u_turn_blocked import EC7UTurnBlocked


def test_select_scenarios_accepts_multiple_short_names():
    scenarios = {
        "s1_static_box": object(),
        "s2_person_crossing": object(),
        "s3_narrow_gap": object(),
    }

    selected = select_scenarios(scenarios, ["s1", "s3"])

    assert list(selected) == ["s1_static_box", "s3_narrow_gap"]


def test_select_scenarios_rejects_unknown_names():
    try:
        select_scenarios({"s1_static_box": object()}, ["unknown"])
    except ValueError as exc:
        assert "No scenario matches 'unknown'." in str(exc)
    else:
        raise AssertionError("Unknown scenario name must be rejected")


def test_repeated_uturn_command_does_not_restart_maneuver():
    controller = MPPI()
    maneuver = ManeuverStateMachine(controller)
    path = [(float(x), config.LEFT_LANE_CENTER_Y) for x in range(20)]
    starting_state = np.array([10.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5])

    maneuver.update(starting_state, path, None, scenario_command="u_turn")
    initial_target = maneuver.target

    maneuver.update(starting_state, path, None, scenario_command="u_turn")
    assert maneuver.state == "IN_UTURN"
    assert maneuver.target == initial_target

    completed_state = np.array([initial_target[0], initial_target[1], initial_target[2], 0.5])
    maneuver.update(completed_state, path, None, scenario_command="u_turn")
    assert maneuver.state == "RETURN_LEFT"
    maneuver.update(completed_state, path, None, scenario_command="u_turn")
    assert maneuver.state == "RETURN_LEFT"
    assert controller.maneuver_mode == "return_left"


def test_uturn_infeasible_is_reported_and_aborted(monkeypatch):
    monkeypatch.setattr(UTurnManeuver, "feasible", staticmethod(lambda: False))
    controller = MPPI()
    maneuver = ManeuverStateMachine(controller)

    maneuver.update(
        np.array([10.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5]),
        [(float(x), config.LEFT_LANE_CENTER_Y) for x in range(20)],
        None,
        scenario_command="u_turn",
    )

    assert maneuver.state == "NORMAL"
    assert maneuver.failure_reason == "uturn_infeasible"
    assert maneuver.failed_maneuver == "u_turn"
    assert controller.maneuver_mode == "normal"


def test_uturn_target_with_no_free_projection_is_aborted():
    controller = MPPI()
    maneuver = ManeuverStateMachine(controller)
    costmap = Costmap2D()
    costmap.add_static_obstacle(
        Obstacle(
            x=config.MAP_W / 2,
            y=config.MAP_H / 2,
            width=config.MAP_W + 2.0,
            height=config.MAP_H + 2.0,
        )
    )

    maneuver.update(
        np.array([10.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5]),
        [(float(x), config.LEFT_LANE_CENTER_Y) for x in range(20)],
        costmap,
        scenario_command="u_turn",
    )

    assert maneuver.state == "NORMAL"
    assert maneuver.failure_reason == "maneuver_unreachable"


def test_new_obstacle_spawn_overlap_is_shifted_clear():
    scenario = EC4LateBraking()
    scenario.reset()
    scenario.obstacle.x = 2.0
    scenario.obstacle.y = 10.0
    scenario.obstacle.active = True

    adjusted = scenario.clear_spawn_overlaps(
        np.array([2.0, 10.0, 0.0, 0.0]),
        previously_active=set(),
    )

    assert adjusted == ["brake_hazard"]
    assert scenario.obstacle.dist_to_point(2.0, 10.0) >= config.ROBOT_RADIUS


def test_runner_projects_goal_out_of_obstacle_inflation(monkeypatch):
    scenario = EC1DeadEnd()
    scenario.reset(seed=1)
    scenario.goal = (12.0, 10.0)
    scenario.time_limit = 0.0
    costmap = scenario.costmap
    costmap.add_lane_boundaries()
    controller = MPPI()
    recorder = MetricsRecorder("MPPI", scenario.name, 0)
    monkeypatch.setattr(recorder, "save_timestep_csv", lambda *_: "")

    run_simulation(
        scenario,
        controller,
        recorder,
        BicycleModel(),
        HealthMonitor(),
        costmap,
    )

    assert recorder.goal_adjusted is True
    assert recorder.failure_reason == "timeout"


@pytest.mark.parametrize("scenario_type", [S1StaticBox, S4TwoObstacles])
def test_box_scenarios_reach_goal_without_collision(scenario_type, tmp_path, monkeypatch):
    scenario = scenario_type()
    scenario.time_limit = 25.0
    scenario.reset(seed=42)
    costmap = scenario.costmap
    costmap.add_lane_boundaries()
    controller = MPPI()
    controller.reset()
    recorder = MetricsRecorder("MPPI", scenario.name, 0)
    monitor = HealthMonitor()
    monitor.reset()
    model = BicycleModel()
    model.reset()
    monkeypatch.setattr("main._RAW_DIR", str(tmp_path))

    run_simulation(
        scenario,
        controller,
        recorder,
        model,
        monitor,
        costmap,
    )

    assert recorder.completed is True, recorder.failure_reason
    assert recorder.min_obstacle_dist > config.ROBOT_RADIUS
    assert recorder.completion_time < scenario.time_limit
    assert recorder.path_length < 1.5 * (
        np.hypot(
            scenario.goal[0] - scenario.start[0],
            scenario.goal[1] - scenario.start[1],
        )
    )


def test_ec7_switches_to_return_path_after_uturn():
    scenario = EC7UTurnBlocked()
    scenario.reset(seed=42)
    forward_path = list(scenario.get_path())

    command = scenario.get_maneuver_command(
        1.0,
        np.array([10.5, config.LEFT_LANE_CENTER_Y, 0.0, 1.0]),
    )
    assert command == "u_turn"
    assert scenario.get_path() == forward_path

    command = scenario.get_maneuver_command(
        2.0,
        np.array([10.5, 14.0, np.pi, 1.0]),
    )

    assert command == "u_turn"
    assert scenario.get_path() != forward_path
    assert scenario.get_path()[0] == (10.5, 14.0)
    assert scenario.get_path()[-1] == scenario.start[:2]

    scenario.reset(seed=43)
    assert scenario.get_path() == forward_path
    assert scenario._return_path_active is False


def test_ec7_stops_at_source_only_after_starting_the_return():
    scenario = EC7UTurnBlocked()
    scenario.reset(seed=42)
    source_state = np.array([scenario.start[0], scenario.start[1], 0.0, 0.0])

    assert scenario.is_complete(source_state, 0.0) == (False, False, "")

    scenario.get_maneuver_command(
        1.0,
        np.array([10.5, config.LEFT_LANE_CENTER_Y, 0.0, 1.0]),
    )
    assert scenario.is_complete(source_state, 1.0) == (False, False, "")

    scenario.get_maneuver_command(
        2.0,
        np.array([10.5, 14.0, np.pi, 1.0]),
    )
    assert scenario.is_complete(
        np.array([scenario.start[0] + config.GOAL_TOLERANCE, scenario.start[1], np.pi, 0.0]),
        2.0,
    ) == (False, False, "")
    assert scenario.is_complete(source_state, 2.0) == (True, True, "")


def test_ec7_uturn_returns_to_goal_without_following_old_forward_path(tmp_path, monkeypatch):
    scenario = EC7UTurnBlocked()
    scenario.reset(seed=42)
    scenario.time_limit = 35.0
    scenario.costmap.add_lane_boundaries()
    controller = MPPI()
    controller.reset()
    recorder = MetricsRecorder("MPPI", scenario.name, 0)
    monitor = HealthMonitor()
    monitor.reset()
    model = BicycleModel()
    model.reset()
    monkeypatch.setattr("main._RAW_DIR", str(tmp_path))
    np.random.seed(42)

    run_simulation(
        scenario,
        controller,
        recorder,
        model,
        monitor,
        scenario.costmap,
    )

    assert recorder.completed is True, recorder.failure_reason
    assert recorder.maneuver_success["u_turn"] is True
    assert "right_turn" not in recorder.maneuver_success
    assert recorder.completion_time < scenario.time_limit
    assert np.hypot(
        recorder.positions[-1][0] - scenario.start[0],
        recorder.positions[-1][1] - scenario.start[1],
    ) < config.GOAL_TOLERANCE

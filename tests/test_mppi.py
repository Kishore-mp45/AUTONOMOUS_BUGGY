"""
tests/test_mppi.py
==================
Unit tests for MPPI controller 5-critic cost, fallbacks, and control updates.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from controllers.mppi import MPPI
from environment.costmap import Costmap2D
from environment.lane_map import LaneMap
from environment.obstacle import Obstacle
from mppi_critics.lane_critic import LaneCritic
from mppi_critics.maneuver_critic import ManeuverCritic
from mppi_critics.obstacle_critic import ObstacleCritic


def _trajectory(y, theta=0.0, x=1.0, steps=10):
    states = np.zeros((steps, 4), dtype=float)
    states[:, 0] = x
    states[:, 1] = y
    states[:, 2] = theta
    return states


def _rollouts(*trajectories):
    return np.stack(trajectories, axis=0)


def test_produces_valid_control():
    """Output steering and velocity must strictly lie within vehicle limits."""
    mppi = MPPI()
    cm = Costmap2D()
    state = np.array([1.0, 10.0, 0.0, 0.5])
    path = [(x, 10.0) for x in np.linspace(1, 19, 20)]
    steer, speed = mppi.compute_control(state, path=path, goal=(19.0, 10.0), costmap=cm)
    assert -config.MAX_STEER <= steer <= config.MAX_STEER
    assert 0.0 <= speed <= config.MAX_SPEED


def test_fallback_on_all_collide():
    """All-collide: recovery widens sigma first, emergency stop after max attempts."""
    mppi = MPPI()
    cm = Costmap2D()
    # Fill costmap with lethal obstacle directly around vehicle
    wall = Obstacle(x=1.0, y=10.0, width=5.0, height=5.0)
    cm.add_static_obstacle(wall)
    state = np.array([1.0, 10.0, 0.0, 0.5])

    # First attempts: recovery creep with widened sigma
    steer, speed = mppi.compute_control(state, goal=(19.0, 10.0), costmap=cm)
    assert steer == 0.0
    assert speed == 0.1
    assert mppi._recovery_count == 1
    assert mppi._current_sigma[0] > config.MPPI_SIGMA_STEER

    # After max attempts: emergency stop
    for _ in range(config.MPPI_RECOVERY_MAX_ATTEMPTS - 1):
        steer, speed = mppi.compute_control(state, goal=(19.0, 10.0), costmap=cm)
    assert steer == 0.0
    assert speed == 0.0
    assert mppi._recovery_count == 0
    assert mppi.emergency_stop is True
    assert mppi._recovery_count_total == config.MPPI_RECOVERY_MAX_ATTEMPTS


def test_obstacle_critic_assigns_lethal_cost():
    costmap = Costmap2D()
    costmap.add_static_obstacle(Obstacle(x=2.0, y=10.0, width=1.0, height=1.0))
    trajectory = _trajectory(10.0, x=2.0)

    costs = ObstacleCritic().score(_rollouts(trajectory), costmap)

    assert costs[0] >= config.MPPI_OBS_COST


def test_obstacle_critic_respects_vehicle_footprint_clearance():
    costmap = Costmap2D()
    obstacle = Obstacle(x=2.0, y=10.0, width=1.0, height=1.0)
    near = _trajectory(10.83, x=2.0, steps=4)
    clear = _trajectory(11.2, x=2.0, steps=4)

    costs = ObstacleCritic().score(
        _rollouts(near, clear),
        costmap,
        obstacles=[obstacle],
    )

    assert costs[0] >= config.MPPI_OBS_COST
    assert costs[1] < config.MPPI_OBS_COST


def test_lane_critic_left_lane_is_free():
    critic = LaneCritic(LaneMap())
    trajectory = _trajectory(config.LEFT_LANE_CENTER_Y)

    costs = critic.score(_rollouts(trajectory), "normal")

    assert costs[0] == 0.0


def test_lane_critic_penalizes_right_lane():
    critic = LaneCritic(LaneMap())
    trajectory = _trajectory(config.RIGHT_LANE_CENTER_Y, steps=4)

    costs = critic.score(_rollouts(trajectory), "normal")

    assert costs[0] >= config.LANE_COST_RIGHT * trajectory.shape[0]


def test_lane_critic_assigns_lethal_boundary_cost():
    critic = LaneCritic(LaneMap())
    trajectory = _trajectory(config.ROAD_Y_MIN - 0.1, steps=4)

    costs = critic.score(_rollouts(trajectory), "normal")

    assert costs[0] == config.LANE_COST_BOUNDARY * trajectory.shape[0]


def test_maneuver_critic_penalizes_wrong_uturn_heading():
    critic = ManeuverCritic()
    aligned = _trajectory(config.RIGHT_LANE_CENTER_Y, theta=np.pi)
    wrong = _trajectory(config.RIGHT_LANE_CENTER_Y, theta=0.0)
    trajectories = _rollouts(aligned, wrong)

    costs = critic.score(
        trajectories,
        (1.0, config.RIGHT_LANE_CENTER_Y, np.pi),
        "u_turn",
    )

    assert costs[1] > costs[0]


def test_adaptive_lambda_uses_lower_value_near_obstacle():
    mppi = MPPI()
    costmap = Costmap2D()
    costmap.add_static_obstacle(Obstacle(x=2.0, y=10.0, width=1.0, height=1.0))
    state = np.array([2.0, 10.0, 0.0, 0.5])

    assert mppi._get_lambda(state, costmap) == config.MPPI_LAMBDA_NEAR_OBS
    assert mppi._get_lambda(state, costmap) < config.MPPI_LAMBDA_FREE


def test_ema_smooths_sudden_control_change(monkeypatch):
    mppi = MPPI()
    costmap = Costmap2D()
    state = np.array([1.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5])
    signs = iter((1.0, -1.0))
    monkeypatch.setattr(
        np.random,
        "randn",
        lambda *shape: np.full(shape, next(signs), dtype=float),
    )
    monkeypatch.setattr(mppi, "_compute_costs", lambda *args: np.zeros(config.MPPI_K))

    mppi.compute_control(state, goal=(19.0, config.LEFT_LANE_CENTER_Y), costmap=costmap)
    first_ema_steer = mppi._ema_u[0]
    steer_after_reverse_noise, _ = mppi.compute_control(
        state, goal=(19.0, config.LEFT_LANE_CENTER_Y), costmap=costmap
    )

    assert first_ema_steer > 0.0
    assert 0.0 < steer_after_reverse_noise < first_ema_steer


def test_recovery_widens_sampling_sigma():
    mppi = MPPI()
    costmap = Costmap2D()
    costmap.add_static_obstacle(Obstacle(x=1.0, y=10.0, width=5.0, height=5.0))
    state = np.array([1.0, 10.0, 0.0, 0.5])

    mppi.compute_control(state, goal=(19.0, 10.0), costmap=costmap)

    assert mppi._recovery_count == 1
    np.testing.assert_allclose(
        mppi._current_sigma,
        np.array([config.MPPI_SIGMA_STEER, config.MPPI_SIGMA_ACCEL])
        * config.MPPI_RECOVERY_SIGMA_MULT,
    )


def test_warm_start_shifts_nominal_sequence(monkeypatch):
    mppi = MPPI()
    cm = Costmap2D()
    state = np.array([1.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5])
    sequence = np.column_stack((
        np.linspace(-0.2, 0.2, mppi.T),
        np.linspace(-0.5, 0.5, mppi.T),
    ))
    mppi.U[:] = sequence
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.zeros(shape))
    previous_second_control = mppi.U[1].copy()

    mppi.compute_control(
        state,
        goal=(19.0, config.LEFT_LANE_CENTER_Y),
        costmap=cm,
    )

    np.testing.assert_array_equal(mppi.U[0], previous_second_control)
    assert np.all(mppi.U[-1] == 0.0)


def test_narrow_context_detects_blockage_on_both_sides():
    mppi = MPPI()
    costmap = Costmap2D()
    state = np.array([5.0, 10.0, 0.0, 0.5])
    costmap.add_static_obstacle(Obstacle(x=5.0, y=9.0, width=0.4, height=0.4))
    costmap.add_static_obstacle(Obstacle(x=5.0, y=11.0, width=0.4, height=0.4))

    assert mppi._is_narrow_context(costmap, state) is True


def test_narrow_context_requires_blockage_on_both_sides():
    mppi = MPPI()
    costmap = Costmap2D()
    state = np.array([5.0, 10.0, 0.0, 0.5])
    costmap.add_static_obstacle(Obstacle(x=5.0, y=9.0, width=0.4, height=0.4))

    assert mppi._is_narrow_context(costmap, state) is False


def test_invalid_weight_sum_uses_uniform_viable_fallback(monkeypatch):
    mppi = MPPI()
    state = np.array([1.0, config.LEFT_LANE_CENTER_Y, 0.0, 0.5])
    monkeypatch.setattr(mppi, "_compute_costs", lambda *args: np.zeros(config.MPPI_K))
    monkeypatch.setattr(np, "exp", lambda values: np.zeros_like(values))

    mppi.compute_control(
        state,
        goal=(19.0, config.LEFT_LANE_CENTER_Y),
        costmap=Costmap2D(),
    )

    np.testing.assert_allclose(mppi.last_weights, np.full(config.MPPI_K, 1.0 / config.MPPI_K))
    assert mppi.last_weight_fallback is True
    assert mppi.weight_fallback_count == 1


def test_rollout_clamps_steering_to_ackermann_minimum_radius(monkeypatch):
    mppi = MPPI()
    state = np.array([1.0, 10.0, 0.0, 1.0])
    requested_radius = 2.0
    monkeypatch.setattr(config, "MIN_TURN_RADIUS", requested_radius)
    controls = np.array([[[config.MAX_STEER, 0.0]]], dtype=float)

    trajectory = mppi._rollout(state, controls, 1)
    heading_change = trajectory[0, 0, 2]
    effective_steer = np.arctan2(
        heading_change * config.WHEELBASE,
        config.DT * state[3],
    )
    effective_radius = config.WHEELBASE / abs(np.tan(effective_steer))

    assert effective_radius >= requested_radius - 1e-9


def test_applied_steering_obeys_ackermann_minimum_radius(monkeypatch):
    mppi = MPPI()
    state = np.array([1.0, config.LEFT_LANE_CENTER_Y, 0.0, 1.0])
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.zeros(shape))
    monkeypatch.setattr(mppi, "_compute_costs", lambda *args: np.zeros(config.MPPI_K))
    monkeypatch.setattr(config, "MPPI_EMA_ALPHA", 0.0)
    requested_radius = 4.0
    monkeypatch.setattr(config, "MIN_TURN_RADIUS", requested_radius)
    mppi.U[0, 0] = config.MAX_STEER

    steer, _ = mppi.compute_control(
        state,
        goal=(19.0, config.LEFT_LANE_CENTER_Y),
        costmap=Costmap2D(),
    )

    applied_radius = config.WHEELBASE / abs(np.tan(steer))
    assert applied_radius >= requested_radius - 1e-9
    assert mppi.ackermann_clamp_count > 0


def test_cost_decreases_with_clear_path():
    """Sample rollout costs on a clear path must be significantly lower than when blocked."""
    mppi = MPPI()
    cm_clear = Costmap2D()
    cm_blocked = Costmap2D()
    cm_blocked.add_static_obstacle(Obstacle(x=2.0, y=10.0, width=2.0, height=2.0))

    state = np.array([1.0, 10.0, 0.0, 1.0])
    T = 20
    rng = np.random.default_rng(0)
    U_noisy = rng.normal(0, 0.1, size=(10, T, 2))
    trajs = mppi._rollout(state, U_noisy, T)

    costs_clear = mppi._compute_costs(trajs, goal=(19.0, 10.0), path=None, costmap=cm_clear)
    costs_blocked = mppi._compute_costs(trajs, goal=(19.0, 10.0), path=None, costmap=cm_blocked)
    assert np.mean(costs_clear) < np.mean(costs_blocked)

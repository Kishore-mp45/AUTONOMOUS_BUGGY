"""
mppi_critics/obstacle_critic.py
===============================
Penalize trajectories that pass through or too close to obstacles.
Clearance includes the buggy footprint and a one-step motion margin.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ObstacleCritic:
    """
    Queries costmap at each trajectory point.
    Obstacle overlap or insufficient footprint clearance → MPPI_OBS_COST.
    Points near obstacle → proportional soft penalty (MPPI_OBSTACLE_WEIGHT).
    """

    name = "obstacle"

    def score(self, trajectories: np.ndarray, costmap, obstacles=None) -> np.ndarray:
        """
        trajectories : shape (K, T, 4)
        costmap      : Costmap2D (uses query_batch for vectorised lookup)
        obstacles    : active obstacle objects used for footprint clearance
        Returns      : shape (K,) cost contribution
        """
        if costmap is None:
            return np.zeros(trajectories.shape[0])

        xs = trajectories[:, :, 0]                  # (K, T)
        ys = trajectories[:, :, 1]
        cell_costs = costmap.query_batch(xs, ys)

        lethal = cell_costs >= config.LETHAL_COST
        if obstacles:
            collision_clearance = config.ROBOT_RADIUS + config.MAX_SPEED * config.DT
            for obstacle in obstacles:
                if not getattr(obstacle, "active", True):
                    continue

                dx = np.abs(xs - float(obstacle.x))
                dy = np.abs(ys - float(obstacle.y))
                radius = getattr(obstacle, "radius", None)
                width = getattr(obstacle, "width", None)
                height = getattr(obstacle, "height", None)
                if radius is not None:
                    surface_dist = np.maximum(
                        0.0,
                        np.hypot(dx, dy) - float(radius),
                    )
                elif width is not None and height is not None:
                    outside_x = np.maximum(dx - float(width) / 2.0, 0.0)
                    outside_y = np.maximum(dy - float(height) / 2.0, 0.0)
                    surface_dist = np.hypot(outside_x, outside_y)
                else:
                    surface_dist = np.hypot(dx, dy)

                lethal |= surface_dist < collision_clearance

        lethal_costs = lethal.any(axis=1).astype(float) * config.MPPI_OBS_COST
        soft_costs = config.MPPI_OBSTACLE_WEIGHT * np.where(
            lethal, 0.0, cell_costs
        ).sum(axis=1)
        return lethal_costs + soft_costs


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from environment.costmap import Costmap2D
    from environment.obstacle import Obstacle

    cm = Costmap2D()
    cm.add_static_obstacle(Obstacle(10.0, 10.0, width=1.5, height=1.5))

    crit = ObstacleCritic()
    # One trajectory straight through the box (x: 9->12 at y=10),
    # one that stops well before it (x: 1->8 at y=10)
    T = 20
    t = np.linspace(0, 1, T)
    hit = np.stack([9.0 + 3.0 * t, np.full(T, 10.0), np.zeros(T), np.ones(T)], axis=-1)[None]
    safe = np.stack([1.0 + 7.0 * t, np.full(T, 10.0), np.zeros(T), np.ones(T)], axis=-1)[None]
    trajs = np.concatenate([hit, safe])
    costs = crit.score(trajs, cm)
    assert costs[0] >= config.MPPI_OBS_COST, "trajectory inside box must be lethal"
    assert costs[1] < config.MPPI_OBS_COST, "clear trajectory must stay cheap"
    print(f"[PASS] ObstacleCritic: hit={costs[0]:.3e}, safe={costs[1]:.3f}")

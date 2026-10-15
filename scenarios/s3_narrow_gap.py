"""
scenarios/s3_narrow_gap.py
===========================
Scenario 3 — Narrow Corridor / Gap

Map layout (20 x 20 m):
    Start : (1, 10)   heading East
    Goal  : (19, 10)
    Path  : straight horizontal line at y = 10

    Two wall segments straddle x = 10, leaving a 3.0 m gap centred on y = 10:

        Wall 1 (lower):  x=10,  y from 0    to 8.5   (below the gap)
        Wall 2 (upper):  x=10,  y from 11.5 to 20    (above the gap)

        Gap: y = 8.5 → 11.5  =  3.0 m wide
        Robot width ~ 0.6 m, so the gap is ~5× wider than the robot
        but still requires precise alignment.

    Walls are represented as thin rectangles (thickness 0.3 m).

Challenge: Robot must thread the gap precisely. Tests lateral precision
           under kinematic constraints.
"""

import copy
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
import config


# Gap geometry
_WALL_X         = 10.0
_WALL_THICKNESS =  0.3    # metres
_GAP_LOW_Y      =  8.5    # gap starts here (top of lower wall)
_GAP_HIGH_Y     = 11.5    # gap ends here   (bottom of upper wall)
_GAP_WIDTH      = _GAP_HIGH_Y - _GAP_LOW_Y   # 3.0 m


def _wall_rect(x_centre, y_start, y_end, thickness):
    """
    Return an obstacle dict for a vertical wall segment.
    x_centre   : horizontal centre of the wall
    y_start    : bottom y of the wall
    y_end      : top    y of the wall
    thickness  : wall width in x direction
    """
    cy = (y_start + y_end) / 2.0
    h  = y_end - y_start
    return {
        "type": "wall",
        "x": x_centre,
        "y": cy,
        "w": thickness,
        "h": h,
        "label": f"wall_y{y_start:.1f}-{y_end:.1f}",
    }


class S3NarrowGap(ScenarioBase):
    """
    Scenario 3: Two vertical wall segments create a narrow gap at x=10.
    Robot must align itself to pass through the 3 m opening centred at y=10.
    """

    def __init__(self):
        super().__init__()
        self.name = "s3_narrow_gap"

        self.start = (1.0, 10.0, 0.0)
        self.goal  = (19.0, 10.0)
        self.global_path = straight_path(1.0, 10.0, 19.0, 10.0, n_pts=40)

        _obstacles = [
            # Lower wall: x=10, y from 0 to 8.5
            _wall_rect(_WALL_X, 0.0, _GAP_LOW_Y,  _WALL_THICKNESS),
            # Upper wall: x=10, y from 11.5 to 20
            _wall_rect(_WALL_X, _GAP_HIGH_Y, 20.0, _WALL_THICKNESS),
        ]
        self._initial_obstacles = copy.deepcopy(_obstacles)
        self.obstacles          = copy.deepcopy(_obstacles)

    # update() inherited — walls are static, no motion needed


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = S3NarrowGap()
    print(s)
    for obs in s.obstacles:
        print(f"  {obs['label']:30s}  centre=({obs['x']:.1f},{obs['y']:.2f})  "
              f"w={obs['w']:.2f} h={obs['h']:.2f}")

    import numpy as np
    # Robot aligned on path centre should not collide with walls
    mid_state = np.array([10.0, 10.0, 0.0, 1.0])   # in the gap
    assert not s.collision_detected(mid_state), "Centre of gap should be clear"

    # Robot touching the lower wall should collide
    # Lower wall top edge = 8.5, robot radius = 0.3 → collision if y < 8.5 + 0.15 + 0.3
    danger_state = np.array([10.0, 8.6, 0.0, 1.0])
    dist = s.min_obstacle_dist(danger_state)
    print(f"  Distance at y=8.6 (near lower wall): {dist:.3f} m")
    assert dist < config.ROBOT_RADIUS * 3, "Should be close to lower wall"

    s.reset()
    assert len(s.obstacles) == 2
    print("[PASS]  S3NarrowGap OK")

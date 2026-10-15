"""
mppi_critics/__init__.py
========================
Critic plugin package for the MPPI controller (Nav2-style architecture).

Each critic is a small class exposing:
    name : str
    score(...) -> np.ndarray of shape (K,)
where K is the number of sampled trajectories, shape (K, T, 4) with state
[x, y, theta, v] per step.

MPPI sums all critic outputs into the total trajectory cost.
"""

from .obstacle_critic  import ObstacleCritic
from .goal_critic      import GoalCritic
from .path_critic      import PathCritic
from .jerk_critic      import JerkCritic
from .speed_critic     import SpeedCritic
from .lane_critic      import LaneCritic
from .maneuver_critic  import ManeuverCritic

__all__ = [
    "ObstacleCritic",
    "GoalCritic",
    "PathCritic",
    "JerkCritic",
    "SpeedCritic",
    "LaneCritic",
    "ManeuverCritic",
]

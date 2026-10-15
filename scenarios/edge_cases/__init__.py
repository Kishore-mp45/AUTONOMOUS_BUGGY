"""
scenarios/edge_cases/__init__.py
================================
Package for Edge Case Scenarios:
  - EC1:  Dead End / U-Turn
  - EC3:  Tight 90-degree Turn
  - EC4:  Late Braking
  - EC5:  Moving Crowd (Multi-Pedestrian)
  - EC6:  Wrong Lane Entry (left-lane block, overtake + return)
  - EC7:  U-Turn Blocked Road (goal behind start)
  - EC8:  Missing Lane Markings (lane guidance disabled)
  - EC9:  Sensor Noise (noisy state feedback)
  - EC10: All Samples Collide (recovery / sigma expansion)

(EC2 oscillation trigger removed — it targeted the deleted RPP controller.)
"""

from .ec1_dead_end import EC1DeadEnd
from .ec3_tight_turn import EC3TightTurn
from .ec4_late_braking import EC4LateBraking
from .ec5_crowd import EC5Crowd
from .ec6_wrong_lane_entry import EC6WrongLaneEntry
from .ec7_u_turn_blocked import EC7UTurnBlocked
from .ec8_missing_lane_mark import EC8MissingLaneMark
from .ec9_sensor_noise import EC9SensorNoise
from .ec10_all_samples_collide import EC10AllSamplesCollide

__all__ = [
    "EC1DeadEnd",
    "EC3TightTurn",
    "EC4LateBraking",
    "EC5Crowd",
    "EC6WrongLaneEntry",
    "EC7UTurnBlocked",
    "EC8MissingLaneMark",
    "EC9SensorNoise",
    "EC10AllSamplesCollide",
]

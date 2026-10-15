"""
scenarios/edge_cases/__init__.py
================================
Package for Phase D6 Edge Case Scenarios:
  - EC1: Dead End / U-Turn
  - EC2: Oscillation Trigger (High Speed Straight)
  - EC3: Tight 90-degree Turn
  - EC4: Late Braking
  - EC5: Moving Crowd (Multi-Pedestrian)
"""

from .ec1_dead_end import EC1DeadEnd
from .ec2_oscillation_trigger import EC2OscillationTrigger
from .ec3_tight_turn import EC3TightTurn
from .ec4_late_braking import EC4LateBraking
from .ec5_crowd import EC5Crowd

__all__ = [
    "EC1DeadEnd",
    "EC2OscillationTrigger",
    "EC3TightTurn",
    "EC4LateBraking",
    "EC5Crowd",
]

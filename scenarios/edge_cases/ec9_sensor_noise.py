"""
scenarios/edge_cases/ec9_sensor_noise.py
=========================================
Edge Case 9 — Sensor Noise / State Uncertainty.

Gaussian noise is added to the vehicle state before it is passed to the
MPPI controller (main.py reads scenario.sensor_noise).

  pos_noise     ~ N(0, SENSOR_NOISE_POS)   [metres]
  heading_noise ~ N(0, SENSOR_NOISE_HDG)   [radians]

Expected: MPPI is inherently robust to moderate state noise (sampling is
stochastic); the run tracks cross-track error variance.

Metric: cross-track error variance across runs (recorder).
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scenarios.scenario_base import ScenarioBase, straight_path
from environment.obstacle import Obstacle
from config import LEFT_LANE_CENTER_Y, SENSOR_NOISE_POS, SENSOR_NOISE_HDG
import config


class EC9SensorNoise(ScenarioBase):
    """
    EC9: Standard static-box scenario driven with noisy state feedback.
    """

    def __init__(self):
        super().__init__()
        self.name = "ec9_sensor_noise"

        self.start = (1.0, LEFT_LANE_CENTER_Y, 0.0)
        self.goal  = (19.0, LEFT_LANE_CENTER_Y)
        self.global_path = straight_path(1.0, LEFT_LANE_CENTER_Y,
                                         19.0, LEFT_LANE_CENTER_Y, n_pts=40)

        self.obstacles = [
            Obstacle(x=10.0, y=LEFT_LANE_CENTER_Y, width=1.5, height=1.5,
                     label="noisy_box"),
        ]
        self.static_obstacles = self.obstacles

        # EC9 hook: main.py injects this noise into the controller's state
        self.sensor_noise = (SENSOR_NOISE_POS, SENSOR_NOISE_HDG)

    def _on_reset(self, seed: Optional[int] = None):
        for obs in self.obstacles:
            obs.reset()
        self.costmap = self.build_costmap()


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== EC9SensorNoise self-test ===")
    scen = EC9SensorNoise()
    scen.reset(seed=42)
    assert scen.sensor_noise == (config.SENSOR_NOISE_POS, config.SENSOR_NOISE_HDG)
    assert len(scen.obstacles) == 1
    print(f"Sensor noise: pos~N(0,{scen.sensor_noise[0]}), hdg~N(0,{scen.sensor_noise[1]})")
    print("[PASS] EC9SensorNoise verified.")

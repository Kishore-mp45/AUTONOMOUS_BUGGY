"""
environment/costmap.py
======================
2D Occupancy Grid Costmap with static, obstacle, and inflation layers.
Shared across all controllers (RPP, DWA, MPPI).
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    COSTMAP_RESOLUTION,
    INFLATION_RADIUS,
    INFLATION_DECAY,
    LETHAL_COST,
    INSCRIBED_COST,
    MAP_W,
    MAP_H,
    ROAD_Y_MIN,
    ROAD_Y_MAX,
)
from environment.obstacle import Obstacle


class Costmap2D:
    """
    2D occupancy grid with 3 layers:
      Layer 1 — Static:    walls, fixed structures
      Layer 2 — Obstacle:  dynamic obstacles stamped each timestep
      Layer 3 — Inflation: cost gradient around all obstacles

    All controllers call costmap.query(x, y) to get the cost at a world point.
    """

    def __init__(
        self,
        res: float = COSTMAP_RESOLUTION,
        map_w: float = MAP_W,
        map_h: float = MAP_H,
        inflation_radius: float = INFLATION_RADIUS,
        inflation_decay: float = INFLATION_DECAY,
        lethal_cost: float = LETHAL_COST,
        inscribed_cost: float = INSCRIBED_COST,
    ):
        self.res              = float(res)
        self.map_w            = float(map_w)
        self.map_h            = float(map_h)
        self.inflation_radius = float(inflation_radius)
        self.inflation_decay  = float(inflation_decay)
        self.lethal_cost      = float(lethal_cost)
        self.inscribed_cost   = float(inscribed_cost)

        self.cols = int(self.map_w / self.res)
        self.rows = int(self.map_h / self.res)

        self.static_layer   = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.obstacle_layer = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.inflated       = np.zeros((self.rows, self.cols), dtype=np.float32)

    # ── World <-> Grid conversion ─────────────────────────
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (x, y) to (row, col)."""
        col = int(np.clip(x / self.res, 0, self.cols - 1))
        row = int(np.clip(y / self.res, 0, self.rows - 1))
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert (row, col) cell index to center world coordinates (x, y)."""
        x = col * self.res + self.res / 2.0
        y = row * self.res + self.res / 2.0
        return float(x), float(y)

    # ── Static layer ────────────────────────────────────
    def add_static_obstacle(self, obstacle: Obstacle):
        """Call once at scenario setup for walls and fixed boxes."""
        self._stamp_obstacle(self.static_layer, obstacle)
        self._inflate()

    def clear_static_layer(self):
        """Clear all static obstacles."""
        self.static_layer[:] = 0.0
        self._inflate()

    # ── Lane boundaries (Indian left-hand road) ────────
    def add_lane_boundaries(self):
        """
        Stamp road boundaries (y <= ROAD_Y_MIN and y >= ROAD_Y_MAX) as lethal.
        These are static — added once at scenario setup (called from main.py).
        The road centerline is NOT stamped here: lane discipline is a soft
        cost handled by LaneMap / LaneCritic, not a lethal wall.
        """
        row_centers = np.arange(self.rows) * self.res + self.res / 2.0
        outside_rows = (row_centers <= ROAD_Y_MIN) | (row_centers >= ROAD_Y_MAX)
        self.static_layer[outside_rows, :] = self.lethal_cost
        self._inflate()

    # ── Dynamic update (called every timestep) ──────────
    def update(self, obstacles: List[Obstacle]):
        """
        Clear obstacle layer, re-stamp all active obstacles,
        then recompute inflation. Called once per simulation step.
        """
        self.obstacle_layer[:] = 0.0
        for obs in obstacles:
            if obs.active:
                self._stamp_obstacle(self.obstacle_layer, obs)
        self._inflate()

    def _stamp_obstacle(self, layer: np.ndarray, obs: Obstacle):
        """Mark cells occupied by this obstacle as LETHAL_COST."""
        if not obs.active:
            return

        if obs.radius is not None:
            # Circular obstacle: stamp in bounding box around circle
            cx, cy = obs.x, obs.y
            r_cells = int(obs.radius / self.res) + 1
            row0, col0 = self.world_to_cell(cx, cy)
            r_min = max(0, row0 - r_cells)
            r_max = min(self.rows, row0 + r_cells + 1)
            c_min = max(0, col0 - r_cells)
            c_max = min(self.cols, col0 + r_cells + 1)

            # Generate grid coordinates in local slice
            rs, cs = np.meshgrid(np.arange(r_min, r_max), np.arange(c_min, c_max), indexing="ij")
            wxs = cs * self.res + self.res / 2.0
            wys = rs * self.res + self.res / 2.0
            dists = np.hypot(wxs - cx, wys - cy)
            mask = dists <= obs.radius
            layer[rs[mask], cs[mask]] = self.lethal_cost
        elif obs.width is not None and obs.height is not None:
            # Rectangular obstacle: compute bounding cell range
            left   = obs.x - obs.width / 2.0
            right  = obs.x + obs.width / 2.0
            bottom = obs.y - obs.height / 2.0
            top    = obs.y + obs.height / 2.0

            r_start, c_start = self.world_to_cell(left, bottom)
            r_end,   c_end   = self.world_to_cell(right, top)

            r_min = min(r_start, r_end)
            r_max = max(r_start, r_end) + 1
            c_min = min(c_start, c_end)
            c_max = max(c_start, c_end) + 1

            layer[r_min:r_max, c_min:c_max] = self.lethal_cost

    def _inflate(self):
        """
        Inflate combined (static + obstacle) grid using distance transform.
        Cost falls off exponentially from obstacle surface.
        """
        combined = np.maximum(self.static_layer, self.obstacle_layer)
        free_mask = (combined < self.lethal_cost).astype(np.float32)

        # Distance transform in cell units
        dist_cells = distance_transform_edt(free_mask)
        dist_m = dist_cells * self.res

        # Exponential decay outside the obstacle surface
        decay_cost = 100.0 * np.exp(-self.inflation_decay * dist_m)
        self.inflated = np.where(combined >= self.lethal_cost, self.lethal_cost, decay_cost).astype(np.float32)

    # ── Query (called by all controllers) ───────────────
    def query(self, x: float, y: float) -> float:
        """Return cost [0–100] at world position (x, y). 100 = lethal."""
        r, c = self.world_to_cell(x, y)
        return float(self.inflated[r, c])

    def query_batch(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorised query for arrays of world positions.
        xs, ys: numpy arrays of identical shape.
        Returns: numpy array of same shape containing costs.
        """
        cols = np.clip((xs / self.res).astype(int), 0, self.cols - 1)
        rows = np.clip((ys / self.res).astype(int), 0, self.rows - 1)
        return self.inflated[rows, cols]

    def is_free(self, x: float, y: float) -> bool:
        """Check if world position (x, y) is below lethal cost threshold."""
        return self.query(x, y) < self.lethal_cost

    def nearest_obstacle_dist(self, x: float, y: float, obstacles: List[Obstacle]) -> float:
        """Fast exact distance to nearest obstacle surface (for metrics)."""
        active_obs = [obs for obs in obstacles if obs.active]
        if not active_obs:
            return float('inf')
        return min(obs.dist_to_point(x, y) for obs in active_obs)

    # ── Goal safety check (Edge Case 8) ─────────────────
    def project_goal_to_free(self, gx: float, gy: float, search_radius: float = 2.0) -> Tuple[float, float]:
        """
        If goal is inside inflation radius / obstacle, find nearest free cell.
        Returns (gx, gy) unchanged if already free.
        """
        if self.query(gx, gy) < INSCRIBED_COST:
            return float(gx), float(gy)

        best_dist = float('inf')
        best = (float(gx), float(gy))
        steps = int(search_radius / self.res)
        gr, gc = self.world_to_cell(gx, gy)

        for dr in range(-steps, steps + 1):
            for dc in range(-steps, steps + 1):
                r, c = gr + dr, gc + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.inflated[r, c] < INSCRIBED_COST:
                        wx, wy = self.cell_to_world(r, c)
                        d = float(np.hypot(wx - gx, wy - gy))
                        if d < best_dist:
                            best_dist = d
                            best = (wx, wy)

        print(f"[costmap] Goal adjusted from ({gx:.2f}, {gy:.2f}) to ({best[0]:.2f}, {best[1]:.2f})")
        return best

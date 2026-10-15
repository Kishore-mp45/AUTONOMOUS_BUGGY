"""
environment/obstacle.py
=======================
Unified obstacle class representing static and dynamic obstacles.
All scenarios instantiate Obstacle objects, which are then rendered into Costmap2D.
"""

from typing import Optional
import numpy as np


class Obstacle:
    """
    Single obstacle — static or dynamic.
    All scenarios create Obstacle objects; costmap and metrics read them.
    """
    def __init__(
        self,
        x: float,
        y: float,
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        dynamic: bool = False,
        vx: float = 0.0,
        vy: float = 0.0,
        label: str = "",
    ):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius) if radius is not None else None   # circular obstacle (person, bike)
        self.width  = float(width)  if width  is not None else None   # rectangular obstacle (box, wall)
        self.height = float(height) if height is not None else None
        self.dynamic = bool(dynamic)
        self.vx = float(vx)                                           # velocity (dynamic only)
        self.vy = float(vy)
        self.label = str(label)                                       # "person" | "box" | "wall" | "bike"
        self.active = True                                            # False = not yet spawned (Scenario 5)

        # Store initial positions for easy reset
        self._init_x = self.x
        self._init_y = self.y
        self._init_vx = self.vx
        self._init_vy = self.vy
        self._init_active = self.active

    def reset(self):
        """Restore initial position and state."""
        self.x = self._init_x
        self.y = self._init_y
        self.vx = self._init_vx
        self.vy = self._init_vy
        self.active = self._init_active

    def step(self, dt: float):
        """Move dynamic obstacle one timestep."""
        if self.dynamic and self.active:
            self.x += self.vx * dt
            self.y += self.vy * dt

    def dist_to_point(self, px: float, py: float) -> float:
        """Minimum distance from point (px, py) to this obstacle surface."""
        if not self.active:
            return float('inf')

        if self.radius is not None:
            return max(0.0, float(np.hypot(px - self.x, py - self.y) - self.radius))
        elif self.width is not None and self.height is not None:
            # Rectangle: clamp point to rectangle, measure distance
            cx = float(np.clip(px, self.x - self.width / 2.0, self.x + self.width / 2.0))
            cy = float(np.clip(py, self.y - self.height / 2.0, self.y + self.height / 2.0))
            return float(np.hypot(px - cx, py - cy))
        else:
            return float(np.hypot(px - self.x, py - self.y))

    def to_dict(self) -> dict:
        """Compatibility dict format for visualization and legacy modules."""
        d = {
            "type": "dynamic" if self.dynamic else ("circle" if self.radius is not None else "static"),
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "active": self.active,
        }
        if self.radius is not None:
            d["radius"] = self.radius
        if self.width is not None:
            d["w"] = self.width
        if self.height is not None:
            d["h"] = self.height
        if self.dynamic:
            d["vx"] = self.vx
            d["vy"] = self.vy
        return d

    def __repr__(self) -> str:
        kind = "circ" if self.radius is not None else "rect"
        dyn = "dyn" if self.dynamic else "stat"
        return f"Obstacle({kind}, {dyn}, pos=({self.x:.2f},{self.y:.2f}), active={self.active})"

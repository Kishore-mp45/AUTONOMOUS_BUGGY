"""
visualization/pygame_viz.py
===========================
Live 2D pygame simulation visualization for Campus Buggy Local Planner Comparison.

Renders:
  - Coordinate system mapping: 20x20 m world mapped to 800x800 px window (or configurable)
  - Boundary walls & campus asphalt grid
  - Global nominal path (grey dashed / solid line with waypoint markers)
  - Dynamic & static obstacles (red rectangles, circular pedestrian with walking direction, gap walls)
  - Goal target (green animated glowing circle with flag/target icon)
  - Buggy body (blue car rectangle with heading indicator, front steering wheels, rear axle)
  - Trajectory trail (cyan dots / smooth fading ribbon)
  - Modern Glassmorphism-style HUD overlay:
      * Controller name, Scenario name, Run # / Total runs
      * Elapsed simulation time, current speed, steering angle
      * Distance to goal, min distance to nearest obstacle
      * Real-time status / outcome badge (e.g., RUNNING, GOAL, COLLISION)
"""

import math
import sys
import os
from typing import List, Dict, Any, Tuple, Optional

import pygame
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Color Palette (Modern Dark / Neon Theme) ──────────────────────────────────
COLOR_BG          = (18, 22, 32)       # Sleek dark navy/slate
COLOR_GRID        = (28, 35, 50)       # Subtle grid lines
COLOR_BORDER      = (45, 55, 75)       # Boundary frame
COLOR_PATH        = (90, 105, 130)     # Waypoint reference path
COLOR_WAYPOINT    = (120, 140, 170)    # Small waypoint dots
COLOR_TRAIL       = (0, 220, 255)      # Cyan trajectory trail
COLOR_GOAL        = (46, 213, 115)     # Vibrant emerald green
COLOR_GOAL_RING   = (46, 213, 115, 80) # Glowing outer ring
COLOR_OBSTACLE    = (235, 77, 75)      # Red/Coral obstacle
COLOR_OBS_OUTLINE = (255, 121, 121)    # Crisp obstacle border
COLOR_PEDESTRIAN  = (240, 147, 43)     # Warm amber for moving person
COLOR_BUGGY       = (52, 152, 219)     # Electric blue buggy
COLOR_BUGGY_ROOF  = (41, 128, 185)     # Accent roof
COLOR_BUGGY_WHEEL = (20, 20, 25)       # Dark tire color
COLOR_BUGGY_FRONT = (255, 234, 167)    # Headlight / front indicator

COLOR_HUD_BG      = (12, 16, 24, 210)  # Glassmorphic semi-transparent HUD
COLOR_HUD_BORDER  = (60, 75, 100)
COLOR_TEXT_MAIN   = (240, 245, 250)
COLOR_TEXT_MUTED  = (140, 155, 175)
COLOR_TEXT_ACCENT = (0, 210, 255)


class PygameViz:
    """
    Live 2D visualization renderer using Pygame.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        map_w: float = config.MAP_W,
        map_h: float = config.MAP_H,
        fps: int = 60,
    ):
        self.width = width
        self.height = height
        self.map_w = map_w
        self.map_h = map_h
        self.fps = fps

        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Campus Buggy Sim — Local Planner Comparison")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True)
        self.font_body  = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13)
        self.font_badge = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12, bold=True)

        # Persistent history trail for current run
        self.trail: List[Tuple[float, float]] = []
        self._last_scenario: Optional[str] = None
        self._last_ctrl: Optional[str] = None
        self._last_run_id: Optional[int] = None

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (x: 0..map_w, y: 0..map_h) with (0,0) at bottom-left
        to screen coordinates (px: 0..width, py: 0..height) with (0,0) at top-left.
        """
        px = int((x / self.map_w) * self.width)
        py = int(self.height - (y / self.map_h) * self.height)
        return px, py

    def scale_dist(self, d: float) -> int:
        """Convert a distance in metres to pixels (using width scale)."""
        return max(1, int((d / self.map_w) * self.width))

    def render(
        self,
        state: np.ndarray,
        obstacles: List[Dict[str, Any]],
        path: Optional[List[Tuple[float, float]]] = None,
        goal: Optional[Tuple[float, float]] = None,
        ctrl_name: str = "",
        scenario_name: str = "",
        run_id: int = 0,
        t: float = 0.0,
    ) -> bool:
        """
        Render a single frame.

        Parameters
        ----------
        state : np.ndarray
            [x, y, theta, v]
        obstacles : list of dict
            Obstacles in scenario format
        path : list of (x, y), optional
            Global reference path waypoints
        goal : (x, y), optional
            Goal location
        ctrl_name : str
            Active controller identifier (RPP, DWA, MPPI)
        scenario_name : str
            Active scenario identifier
        run_id : int
            Current run number (0-indexed)
        t : float
            Simulation time in seconds

        Returns
        -------
        keep_running : bool
            False if user requested to quit (closed window or pressed Esc/Q)
        """
        # Handle Pygame window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False

        # Detect new run to reset trail
        if (scenario_name != self._last_scenario or
            ctrl_name != self._last_ctrl or
            run_id != self._last_run_id):
            self.trail.clear()
            self._last_scenario = scenario_name
            self._last_ctrl = ctrl_name
            self._last_run_id = run_id

        # Update trail
        rx, ry, rtheta, rv = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        if not self.trail or math.hypot(rx - self.trail[-1][0], ry - self.trail[-1][1]) > 0.08:
            self.trail.append((rx, ry))
            if len(self.trail) > 500:
                self.trail.pop(0)

        # ── 1. Clear background & draw grid ───────────────────────────
        self.screen.fill(COLOR_BG)
        self._draw_grid()

        # ── 2. Draw Global Path ───────────────────────────────────────
        if path and len(path) >= 2:
            pts = [self.world_to_screen(pt[0], pt[1]) for pt in path]
            pygame.draw.lines(self.screen, COLOR_PATH, False, pts, 2)
            for pt in pts:
                pygame.draw.circle(self.screen, COLOR_WAYPOINT, pt, 2)

        # ── 3. Draw Trajectory Trail ──────────────────────────────────
        if len(self.trail) >= 2:
            trail_pts = [self.world_to_screen(pt[0], pt[1]) for pt in self.trail]
            pygame.draw.lines(self.screen, COLOR_TRAIL, False, trail_pts, 2)

        # ── 4. Draw Goal ──────────────────────────────────────────────
        if goal is not None:
            gx_px, gy_px = self.world_to_screen(goal[0], goal[1])
            # Animated pulse effect based on time t
            pulse = math.sin(t * 5.0) * 2.0
            g_radius = max(3, self.scale_dist(config.GOAL_TOLERANCE) + int(pulse))

            # Outer ring surface with alpha
            ring_surf = pygame.Surface((g_radius * 2 + 4, g_radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (46, 213, 115, 60), (g_radius + 2, g_radius + 2), g_radius)
            self.screen.blit(ring_surf, (gx_px - g_radius - 2, gy_px - g_radius - 2))

            # Inner solid circle & bullseye
            pygame.draw.circle(self.screen, COLOR_GOAL, (gx_px, gy_px), 6)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx_px, gy_px), 2)

        # ── 5. Draw Obstacles ─────────────────────────────────────────
        for obs in obstacles:
            self._draw_obstacle(obs)

        # ── 6. Draw Buggy ─────────────────────────────────────────────
        self._draw_buggy(rx, ry, rtheta, rv)

        # ── 7. Draw HUD Overlay ───────────────────────────────────────
        self._draw_hud(ctrl_name, scenario_name, run_id, t, rx, ry, rv, goal, obstacles)

        # Display update
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_grid(self):
        """Draw 1-meter and 5-meter grid lines across the 20x20m map."""
        for m in range(0, int(self.map_w) + 1):
            x_px, _ = self.world_to_screen(m, 0)
            color = COLOR_BORDER if m in (0, int(self.map_w)) else (COLOR_BORDER if m % 5 == 0 else COLOR_GRID)
            width = 2 if m in (0, int(self.map_w)) or m % 5 == 0 else 1
            pygame.draw.line(self.screen, color, (x_px, 0), (x_px, self.height), width)

        for m in range(0, int(self.map_h) + 1):
            _, y_px = self.world_to_screen(0, m)
            color = COLOR_BORDER if m in (0, int(self.map_h)) else (COLOR_BORDER if m % 5 == 0 else COLOR_GRID)
            width = 2 if m in (0, int(self.map_h)) or m % 5 == 0 else 1
            pygame.draw.line(self.screen, color, (0, y_px), (self.width, y_px), width)

    def _draw_obstacle(self, obs: Dict[str, Any]):
        """Render a single obstacle (static rect, dynamic person, or wall)."""
        obs_type = obs.get("type", "static")

        if obs_type == "circle" or "radius" in obs:
            cx, cy = obs["x"], obs["y"]
            r_m = obs.get("radius", 0.4)
            px, py = self.world_to_screen(cx, cy)
            r_px = self.scale_dist(r_m)

            color_fill = COLOR_PEDESTRIAN if obs_type == "dynamic" else COLOR_OBSTACLE
            pygame.draw.circle(self.screen, color_fill, (px, py), r_px)
            pygame.draw.circle(self.screen, COLOR_OBS_OUTLINE, (px, py), r_px, 2)

            # Walking indicator line if moving
            vx = obs.get("vx", 0.0)
            vy = obs.get("vy", 0.0)
            if abs(vx) > 1e-3 or abs(vy) > 1e-3:
                norm = math.hypot(vx, vy)
                nx, ny = vx / norm, vy / norm
                head_x, head_y = self.world_to_screen(cx + nx * r_m * 1.5, cy + ny * r_m * 1.5)
                pygame.draw.line(self.screen, (255, 255, 255), (px, py), (head_x, head_y), 2)

        else:
            # Rectangular obstacle (box, wall)
            ox, oy = obs["x"], obs["y"]
            w_m = obs.get("w", 1.0)
            h_m = obs.get("h", 1.0)

            # World bounding corners
            left   = ox - w_m / 2.0
            top    = oy + h_m / 2.0  # world top
            px, py = self.world_to_screen(left, top)
            w_px   = self.scale_dist(w_m)
            h_px   = self.scale_dist(h_m)

            rect = pygame.Rect(px, py, w_px, h_px)
            pygame.draw.rect(self.screen, COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, COLOR_OBS_OUTLINE, rect, 2)

    def _draw_buggy(self, rx: float, ry: float, theta: float, v: float):
        """
        Draw realistic car-like campus buggy oriented along theta.
        """
        L = config.WHEELBASE          # 0.5m
        W = 0.4                       # buggy width in metres
        length = 0.7                  # total length in metres

        # Buggy corners relative to robot center/rear axle
        # Let state (rx, ry) be rear axle centre or centre
        # Footprint polygon in local frame (x forward, y left)
        half_w = W / 2.0
        front_x = length - 0.15
        rear_x  = -0.15

        corners_local = [
            (front_x,  half_w),   # front-left
            (front_x, -half_w),   # front-right
            (rear_x,  -half_w),   # rear-right
            (rear_x,   half_w),   # rear-left
        ]

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        poly_screen = []
        for lx, ly in corners_local:
            wx = rx + (lx * cos_t - ly * sin_t)
            wy = ry + (lx * sin_t + ly * cos_t)
            poly_screen.append(self.world_to_screen(wx, wy))

        # Main chassis
        pygame.draw.polygon(self.screen, COLOR_BUGGY, poly_screen)
        pygame.draw.polygon(self.screen, (255, 255, 255), poly_screen, 2)

        # Front headlight / heading beam
        front_mid_wx = rx + front_x * cos_t
        front_mid_wy = ry + front_x * sin_t
        fx_px, fy_px = self.world_to_screen(front_mid_wx, front_mid_wy)
        tip_wx = rx + (front_x + 0.3) * cos_t
        tip_wy = ry + (front_x + 0.3) * sin_t
        tx_px, ty_px = self.world_to_screen(tip_wx, tip_wy)
        pygame.draw.line(self.screen, COLOR_BUGGY_FRONT, (fx_px, fy_px), (tx_px, ty_px), 3)

        # Collision circle (safety radius outline)
        cx_px, cy_px = self.world_to_screen(rx, ry)
        r_rad_px = self.scale_dist(config.ROBOT_RADIUS)
        r_surf = pygame.Surface((r_rad_px * 2 + 2, r_rad_px * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(r_surf, (0, 210, 255, 45), (r_rad_px + 1, r_rad_px + 1), r_rad_px)
        pygame.draw.circle(r_surf, (0, 210, 255, 120), (r_rad_px + 1, r_rad_px + 1), r_rad_px, 1)
        self.screen.blit(r_surf, (cx_px - r_rad_px - 1, cy_px - r_rad_px - 1))

    def _draw_hud(
        self,
        ctrl_name: str,
        scenario_name: str,
        run_id: int,
        t: float,
        rx: float,
        ry: float,
        rv: float,
        goal: Optional[Tuple[float, float]],
        obstacles: List[Dict[str, Any]],
    ):
        """Draw semi-transparent HUD dashboard in the upper-left corner."""
        hud_w, hud_h = 320, 160
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        pygame.draw.rect(hud_surf, COLOR_HUD_BG, (0, 0, hud_w, hud_h), border_radius=8)
        pygame.draw.rect(hud_surf, COLOR_HUD_BORDER, (0, 0, hud_w, hud_h), width=1, border_radius=8)

        # Goal dist
        d_goal = math.hypot(goal[0] - rx, goal[1] - ry) if goal else 0.0

        # Min obs dist
        min_obs = 999.0
        for obs in obstacles:
            if obs.get("type") == "circle" or "radius" in obs:
                d = max(0.0, math.hypot(rx - obs["x"], ry - obs["y"]) - obs.get("radius", 0.4))
            else:
                ox, oy = obs["x"], obs["y"]
                hw = obs.get("w", 1.0) / 2.0
                hh = obs.get("h", 1.0) / 2.0
                cx = max(ox - hw, min(rx, ox + hw))
                cy = max(oy - hh, min(ry, oy + hh))
                d = math.hypot(rx - cx, ry - cy)
            if d < min_obs:
                min_obs = d

        # Header: Controller & Scenario
        title_txt = self.font_title.render(f"{ctrl_name.upper()}", True, COLOR_TEXT_ACCENT)
        run_txt   = self.font_body.render(f"Run #{run_id + 1} | {scenario_name}", True, COLOR_TEXT_MAIN)
        hud_surf.blit(title_txt, (14, 10))
        hud_surf.blit(run_txt, (14, 34))

        # Divider
        pygame.draw.line(hud_surf, COLOR_HUD_BORDER, (14, 56), (hud_w - 14, 56), 1)

        # Stats
        lines = [
            f"Time: {t:5.2f}s  |  Speed: {rv:4.2f} m/s",
            f"Pose: ({rx:4.1f}, {ry:4.1f}) m  |  Dist to Goal: {d_goal:4.1f}m",
            f"Nearest Obstacle: {min_obs:4.2f}m",
        ]
        y_offset = 64
        for line in lines:
            txt = self.font_body.render(line, True, COLOR_TEXT_MUTED)
            hud_surf.blit(txt, (14, y_offset))
            y_offset += 20

        # Danger indicator if close to obstacle
        status_color = (46, 213, 115) if min_obs >= 0.5 else ((240, 147, 43) if min_obs >= config.ROBOT_RADIUS else (235, 77, 75))
        status_text  = "CLEAR" if min_obs >= 0.5 else ("CAUTION" if min_obs >= config.ROBOT_RADIUS else "COLLISION DANGER")
        badge_txt    = self.font_badge.render(f"STATUS: {status_text}", True, status_color)
        hud_surf.blit(badge_txt, (14, 130))

        self.screen.blit(hud_surf, (16, 16))

    def close(self):
        """Cleanly close Pygame."""
        pygame.quit()


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== PygameViz Self-Test ===")
    viz = PygameViz(width=800, height=800)
    fake_state = np.array([2.0, 10.0, 0.0, 1.2])
    fake_obs = [
        {"type": "static", "x": 10.0, "y": 10.0, "w": 1.5, "h": 1.5},
        {"type": "dynamic", "x": 10.0, "y": 7.0, "radius": 0.4, "vx": 0.0, "vy": 0.5},
    ]
    fake_path = [(x, 10.0) for x in np.linspace(1, 19, 20)]

    for step in range(60):
        fake_state[0] += 0.05
        keep = viz.render(
            state=fake_state,
            obstacles=fake_obs,
            path=fake_path,
            goal=(19.0, 10.0),
            ctrl_name="RPP",
            scenario_name="s1_static_box",
            run_id=0,
            t=step * 0.05,
        )
        if not keep:
            break
    viz.close()
    print("[PASS] PygameViz initialized and rendered frames cleanly.")

"""
mppi_critics/maneuver_critic.py
===============================
Guide MPPI during specific maneuvers (turns, U-turn, reverse).
Active only during maneuver modes; replaces goal direction with the
maneuver's intermediate target waypoint.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class ManeuverCritic:
    """
    Active only during maneuver modes (mode != "normal", target given).
    Adds cost for deviation from the maneuver's intermediate waypoint
    (x, y, theta), which the ManeuverStateMachine provides each step.
    """

    name = "maneuver"

    def score(
        self,
        trajectories: np.ndarray,
        maneuver_target=None,      # (tx, ty, t_theta) or None
        maneuver_mode: str = "normal",
    ) -> np.ndarray:
        """
        trajectories    : shape (K, T, 4)
        maneuver_target : (tx, ty, t_theta) or None
        maneuver_mode   : current maneuver mode string
        Returns         : shape (K,)
        """
        K = trajectories.shape[0]
        if maneuver_target is None or maneuver_mode == "normal":
            return np.zeros(K)

        tx, ty, t_theta = maneuver_target

        # Position cost to maneuver target (terminal, heaviest here)
        term_dist = np.hypot(trajectories[:, -1, 0] - tx,
                             trajectories[:, -1, 1] - ty)
        costs = 3.0 * term_dist

        # Heading cost to target heading (for U-turn completion check)
        if maneuver_mode == "u_turn":
            final_theta = trajectories[:, -1, 2]
            heading_err = np.abs(
                np.arctan2(np.sin(final_theta - t_theta),
                           np.cos(final_theta - t_theta))
            )
            costs += 2.0 * heading_err

        return costs


# ── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    crit = ManeuverCritic()
    T = 10
    t = np.arange(T) * 0.5
    trajs = np.stack([t, np.zeros(T), np.zeros(T), np.ones(T)], axis=-1)[None]

    # Inactive in normal mode or without target
    assert np.all(crit.score(trajs, None, "normal") == 0.0)
    assert np.all(crit.score(trajs, (5.0, 0.0, 0.0), "normal") == 0.0)

    # Active with target: closer trajectory is cheaper
    trajs2 = trajs.copy(); trajs2[:, -1, 0] = 5.0   # ends at target
    c_far, c_near = crit.score(np.concatenate([trajs, trajs2]), (5.0, 0.0, 0.0), "left_turn")
    assert c_near < c_far

    # U-turn mode: at the target, aligned heading is cheaper than wrong heading
    trajs3 = trajs.copy(); trajs3[:, -1, 0] = 5.0; trajs3[:, -1, 2] = np.pi   # aligned
    trajs4 = trajs.copy(); trajs4[:, -1, 0] = 5.0; trajs4[:, -1, 2] = 0.0     # wrong heading
    c_align, c_wrong = crit.score(np.concatenate([trajs3, trajs4]), (5.0, 0.0, np.pi), "u_turn")
    assert c_align < c_wrong
    print(f"[PASS] ManeuverCritic: far={c_far:.2f}, near={c_near:.2f}, aligned={c_align:.2f}, wrong-hdg={c_wrong:.2f}")

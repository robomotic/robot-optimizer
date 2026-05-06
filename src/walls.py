"""
Wall geometry and differentiable wall constraint functions.

All walls are in simulation coordinates [0, 1] × [0, 1].
Internal walls are converted from models/maze.xml (maze coords [-1, 1]) via:
    sim_center = (maze_center + 1) / 2
    sim_half   = maze_half / 2
"""

from typing import Tuple

import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Wall geometry (static Python constants — evaluated at JAX trace time)
# ---------------------------------------------------------------------------

# Internal rectangular walls as (cx, cy, hx, hy) in sim coords [0, 1].
# Source: models/maze.xml
#   wall_v1: maze pos=( 0.00, -0.40), half=(0.05, 0.40) → sim (0.500, 0.300), half (0.025, 0.200)
#   wall_h1: maze pos=(-0.40,  0.50), half=(0.40, 0.05) → sim (0.300, 0.750), half (0.200, 0.025)
#   wall_v2: maze pos=( 0.50,  0.60), half=(0.05, 0.30) → sim (0.750, 0.800), half (0.025, 0.150)
INTERNAL_WALLS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.500, 0.300, 0.025, 0.200),  # wall_v1: vertical, centre
    (0.300, 0.750, 0.200, 0.025),  # wall_h1: horizontal, upper-left
    (0.750, 0.800, 0.025, 0.150),  # wall_v2: vertical, upper-right
)


# ---------------------------------------------------------------------------
# Primitive: rectangle SDF
# ---------------------------------------------------------------------------

def rect_sdf(
    px: Array, py: Array,
    cx: float, cy: float, hx: float, hy: float,
) -> Array:
    """
    Signed distance from (px, py) to an axis-aligned rectangle.
    Positive outside, negative inside.
    Differentiable everywhere except the four corners (measure-zero set).
    """
    qx = jnp.abs(px - cx) - hx
    qy = jnp.abs(py - cy) - hy
    # Add 1e-12 inside sqrt to avoid 0/0 NaN in gradient when both clamped terms are zero
    # (robot outside the rectangle but exactly at a corner, or deep inside).
    outside = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2 + 1e-12)
    inside  = jnp.minimum(jnp.maximum(qx, qy), 0.0)
    return outside + inside


def _rect_outward_normal(
    px: Array, py: Array,
    cx: float, cy: float, hx: float, hy: float,
) -> Tuple[Array, Array]:
    """
    Outward unit normal of the nearest rectangle surface point to (px, py).
    Handled analytically for both the exterior and interior cases.
    """
    # --- Exterior: unit vector from nearest surface point to robot ---
    sx = jnp.clip(px, cx - hx, cx + hx)
    sy = jnp.clip(py, cy - hy, cy + hy)
    nx_out = px - sx
    ny_out = py - sy
    len_out = jnp.sqrt(nx_out ** 2 + ny_out ** 2) + 1e-8
    nx_out = nx_out / len_out
    ny_out = ny_out / len_out

    # --- Interior: push toward the nearest face ---
    # jnp.sign(0) = 0 which would zero the normal at exact wall centre;
    # use jnp.where so the direction is always ±1, never 0.
    pen_x = hx - jnp.abs(px - cx)   # penetration depth to nearest x-face
    pen_y = hy - jnp.abs(py - cy)   # penetration depth to nearest y-face
    nx_in = jnp.where(pen_x <= pen_y, jnp.where(px >= cx, 1.0, -1.0), 0.0)
    ny_in = jnp.where(pen_x <= pen_y, 0.0, jnp.where(py >= cy, 1.0, -1.0))

    is_outside = (jnp.abs(px - cx) > hx) | (jnp.abs(py - cy) > hy)
    return (
        jnp.where(is_outside, nx_out, nx_in),
        jnp.where(is_outside, ny_out, ny_in),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_wall_correction(
    px: Array, py: Array, robot_radius: float
) -> Tuple[Array, Array]:
    """
    Push (px, py) out of any penetrated wall using differentiable SDF correction.

    Replaces jnp.clip (which saturates gradients to zero at walls) with
    jnp.maximum-based (ReLU) penetration depth, so gradients remain nonzero
    when the robot is at or inside a wall surface.

    Applied to both outer boundary (half-planes) and internal rectangular walls.
    Internal walls are processed sequentially so corrections compose correctly
    in corners.

    Args:
        px, py: Raw position in sim coords [0, 1] (before correction).
        robot_radius: Robot bounding-circle radius in metres.

    Returns:
        (corrected_px, corrected_py) guaranteed to be outside all walls.
    """
    r = robot_radius

    # Outer half-plane walls — simple 1-D penetration correction
    cx = px + jnp.maximum(0.0, r - px)            # left   (x = 0)
    cx = cx - jnp.maximum(0.0, r - (1.0 - cx))    # right  (x = 1)
    cy = py + jnp.maximum(0.0, r - py)             # bottom (y = 0)
    cy = cy - jnp.maximum(0.0, r - (1.0 - cy))    # top    (y = 1)

    # Internal rectangular walls — SDF-based push-out
    for (wcx, wcy, whx, why) in INTERNAL_WALLS:
        sdf = rect_sdf(cx, cy, wcx, wcy, whx, why)
        penetration = jnp.maximum(0.0, r - sdf)
        nx, ny = _rect_outward_normal(cx, cy, wcx, wcy, whx, why)
        cx = cx + penetration * nx
        cy = cy + penetration * ny

    return cx, cy


def wall_collision_loss(
    raw_px: Array, raw_py: Array, robot_radius: float, margin: float
) -> Array:
    """
    Smooth quadratic penalty for proximity / penetration of any wall.

    Deliberately uses the *raw* (pre-correction) position so that the gradient
    flows back through velocity → control → design_params even when the position
    correction has already clamped the robot to the wall surface.

    Args:
        raw_px, raw_py: Position before apply_wall_correction.
        robot_radius: Robot bounding-circle radius (unused here — margin controls
                      the onset of the penalty instead, matching existing behaviour).
        margin: Penalty activates when SDF < margin.

    Returns:
        Scalar non-negative loss value.
    """
    # Outer walls: clearance to each boundary edge
    clearances = jnp.array([raw_px, 1.0 - raw_px, raw_py, 1.0 - raw_py])
    loss = jnp.sum(jnp.maximum(0.0, margin - clearances) ** 2)

    # Internal walls: SDF-based quadratic penalty
    for (wcx, wcy, whx, why) in INTERNAL_WALLS:
        sdf = rect_sdf(raw_px, raw_py, wcx, wcy, whx, why)
        loss = loss + jnp.maximum(0.0, margin - sdf) ** 2

    return loss


def ray_wall_distance(
    px: Array, py: Array,
    dx: Array, dy: Array,
    max_range: float = 2.0,
) -> Array:
    """
    Distance from (px, py) along direction (dx, dy) to the nearest wall surface.

    Checks the 4 outer boundary planes and all 4 edges of each internal rectangle.
    Uses only jnp.where / jnp.minimum — fully JAX-compatible (grad / jit / vmap / scan).

    Args:
        px, py: Ray origin in sim coords [0, 1].
        dx, dy: Ray direction components (need not be unit length).
        max_range: Distance returned when no intersection found in front of ray.

    Returns:
        Scalar distance clipped to [0, max_range].
    """
    eps = 1e-9
    t = max_range  # running minimum hit distance

    # --- Outer boundary planes ---
    t = jnp.minimum(t, jnp.where(dx < -eps, (0.0 - px) / dx, max_range))
    t = jnp.minimum(t, jnp.where(dx >  eps, (1.0 - px) / dx, max_range))
    t = jnp.minimum(t, jnp.where(dy < -eps, (0.0 - py) / dy, max_range))
    t = jnp.minimum(t, jnp.where(dy >  eps, (1.0 - py) / dy, max_range))

    # --- Internal rectangular walls: 4 edges each ---
    for (wcx, wcy, whx, why) in INTERNAL_WALLS:
        x_lo, x_hi = wcx - whx, wcx + whx
        y_lo, y_hi = wcy - why, wcy + why

        # Left vertical edge (x = x_lo)
        t_v = (x_lo - px) / jnp.where(jnp.abs(dx) > eps, dx, eps)
        y_h = py + t_v * dy
        t = jnp.minimum(t, jnp.where(
            (t_v > eps) & (y_h >= y_lo) & (y_h <= y_hi), t_v, max_range))

        # Right vertical edge (x = x_hi)
        t_v = (x_hi - px) / jnp.where(jnp.abs(dx) > eps, dx, eps)
        y_h = py + t_v * dy
        t = jnp.minimum(t, jnp.where(
            (t_v > eps) & (y_h >= y_lo) & (y_h <= y_hi), t_v, max_range))

        # Bottom horizontal edge (y = y_lo)
        t_h = (y_lo - py) / jnp.where(jnp.abs(dy) > eps, dy, eps)
        x_h = px + t_h * dx
        t = jnp.minimum(t, jnp.where(
            (t_h > eps) & (x_h >= x_lo) & (x_h <= x_hi), t_h, max_range))

        # Top horizontal edge (y = y_hi)
        t_h = (y_hi - py) / jnp.where(jnp.abs(dy) > eps, dy, eps)
        x_h = px + t_h * dx
        t = jnp.minimum(t, jnp.where(
            (t_h > eps) & (x_h >= x_lo) & (x_h <= x_hi), t_h, max_range))

    return jnp.clip(t, 0.0, max_range)

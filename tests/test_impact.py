"""
Tests for differentiable wall impact behaviour.

Verifies that:
  1. apply_wall_correction stops the robot at every wall surface.
  2. Gradients through the collision loss are nonzero when the robot is near/
     inside a wall — unlike jnp.clip which saturates them to zero.
  3. Gradient direction (into the wall) and magnitude scale correctly with
     penetration depth and impact geometry.
  4. Internal maze walls (wall_v1, wall_h1, wall_v2) are enforced as well as
     the outer boundary.
  5. The full simulation objective retains nonzero gradients (lax.scan survives).

Gradient sign convention
------------------------
wall_collision_loss is a *cost to minimise*.  Its gradient w.r.t. raw position
points **into** the wall (increasing the loss).  The gradient-descent step
``params -= lr * grad`` therefore moves the robot **away** from the wall.

  Left wall (x=0) penetration:   gx < 0  → step moves robot in +x (rightward)
  Right wall (x=1) penetration:  gx > 0  → step moves robot in -x (leftward)
  Bottom wall (y=0) penetration: gy < 0  → step moves robot in +y (upward)
  Top wall (y=1) penetration:    gy > 0  → step moves robot in -y (downward)
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from walls import (
    apply_wall_correction,
    wall_collision_loss,
    rect_sdf,
)
from simulation import DifferentiableSimulation, RolloutConfig
from motor_model import MotorModel
from control_policy import WallFollowerPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROBOT_RADIUS = 0.05   # matches RolloutConfig.wall_margin


def _loss_grad(raw_x: float, raw_y: float, margin: float = ROBOT_RADIUS) -> tuple:
    """Return (dloss/draw_x, dloss/draw_y) for wall_collision_loss."""
    g = grad(
        lambda pos: wall_collision_loss(pos[0], pos[1], ROBOT_RADIUS, margin)
    )(jnp.array([raw_x, raw_y]))
    return float(g[0]), float(g[1])


# ---------------------------------------------------------------------------
# 1. Correction correctness
# ---------------------------------------------------------------------------

class TestWallCorrectionPosition:
    """apply_wall_correction must keep the robot outside every wall surface."""

    @pytest.mark.parametrize("raw_x,raw_y", [
        (-0.10,  0.50),   # deeply into left wall
        ( 1.10,  0.50),   # deeply into right wall
        ( 0.50, -0.10),   # deeply into bottom wall
        ( 0.50,  1.10),   # deeply into top wall
    ])
    def test_corrected_outside_outer_boundary(self, raw_x, raw_y):
        cx, cy = apply_wall_correction(
            jnp.array(raw_x), jnp.array(raw_y), ROBOT_RADIUS
        )
        assert float(cx) >= ROBOT_RADIUS - 1e-5, f"x={cx:.4f} inside left wall"
        assert float(cx) <= 1.0 - ROBOT_RADIUS + 1e-5, f"x={cx:.4f} inside right wall"
        assert float(cy) >= ROBOT_RADIUS - 1e-5, f"y={cy:.4f} inside bottom wall"
        assert float(cy) <= 1.0 - ROBOT_RADIUS + 1e-5, f"y={cy:.4f} inside top wall"

    # Use off-centre positions: exact centre has degenerate normal (symmetry)
    # so we displace slightly in the dimension with smallest wall half-extent.
    @pytest.mark.parametrize("raw_x,raw_y,wcx,wcy,whx,why", [
        (0.51,  0.30, 0.500, 0.300, 0.025, 0.200),   # wall_v1, +x offset
        (0.30,  0.76, 0.300, 0.750, 0.200, 0.025),   # wall_h1, +y offset
        (0.76,  0.80, 0.750, 0.800, 0.025, 0.150),   # wall_v2, +x offset
    ])
    def test_corrected_outside_internal_walls(self, raw_x, raw_y, wcx, wcy, whx, why):
        """Robot placed inside each internal wall must be pushed out."""
        cx, cy = apply_wall_correction(
            jnp.array(raw_x), jnp.array(raw_y), ROBOT_RADIUS
        )
        sdf = float(rect_sdf(cx, cy, wcx, wcy, whx, why))
        assert sdf >= ROBOT_RADIUS - 1e-4, (
            f"Robot sdf={sdf:.4f} still inside wall "
            f"({wcx},{wcy}) after correction"
        )

    def test_free_space_unchanged(self):
        """Position far from all walls must not be moved by the correction."""
        # (0.5, 0.6) is above wall_v1 (top edge at y=0.5) and clear of all
        # other walls by > robot_radius.
        raw_x, raw_y = 0.5, 0.6
        cx, cy = apply_wall_correction(
            jnp.array(raw_x), jnp.array(raw_y), ROBOT_RADIUS
        )
        assert abs(float(cx) - raw_x) < 1e-5
        assert abs(float(cy) - raw_y) < 1e-5


# ---------------------------------------------------------------------------
# 2. Gradient nonzero at wall contact
# ---------------------------------------------------------------------------

class TestGradientAtWallImpact:
    """
    Collision-loss gradient must be nonzero near any wall, and must point
    INTO the wall (so that gradient descent moves the robot away from it).
    """

    def test_loss_grad_nonzero_left_wall(self):
        gx, gy = _loss_grad(raw_x=0.02, raw_y=0.5)
        assert abs(gx) > 1e-4, "gradient w.r.t. raw_x should be nonzero at left wall"
        assert gx < 0, "gradient points into left wall (−x); descent moves robot right"
        assert abs(gy) < 1e-5, "y-gradient should be zero for axis-aligned left wall"

    def test_loss_grad_nonzero_right_wall(self):
        gx, gy = _loss_grad(raw_x=0.98, raw_y=0.5)
        assert abs(gx) > 1e-4
        assert gx > 0, "gradient points into right wall (+x); descent moves robot left"

    def test_loss_grad_nonzero_bottom_wall(self):
        gx, gy = _loss_grad(raw_x=0.5, raw_y=0.02)
        assert abs(gy) > 1e-4
        assert gy < 0, "gradient points into bottom wall (−y); descent moves robot up"

    def test_loss_grad_nonzero_top_wall(self):
        gx, gy = _loss_grad(raw_x=0.5, raw_y=0.98)
        assert abs(gy) > 1e-4
        assert gy > 0, "gradient points into top wall (+y); descent moves robot down"

    def test_no_loss_grad_in_free_space(self):
        """Far from all walls the collision loss gradient must be zero."""
        # (0.5, 0.6): above wall_v1's top edge, clear of wall_h1 and wall_v2
        gx, gy = _loss_grad(raw_x=0.5, raw_y=0.6)
        assert abs(gx) < 1e-5
        assert abs(gy) < 1e-5

    @pytest.mark.parametrize("raw_x,raw_y,wcx,wcy", [
        (0.51,  0.30, 0.500, 0.300),   # inside wall_v1, right of centre
        (0.30,  0.76, 0.300, 0.750),   # inside wall_h1, above centre
        (0.76,  0.80, 0.750, 0.800),   # inside wall_v2, right of centre
    ])
    def test_loss_grad_nonzero_internal_walls(self, raw_x, raw_y, wcx, wcy):
        """Robot inside each internal wall: collision loss gradient must be nonzero."""
        gx, gy = _loss_grad(raw_x=raw_x, raw_y=raw_y)
        magnitude = (gx ** 2 + gy ** 2) ** 0.5
        assert magnitude > 1e-4, (
            f"Loss gradient magnitude {magnitude:.6f} too small "
            f"inside internal wall centred at ({wcx},{wcy})"
        )


# ---------------------------------------------------------------------------
# 3. Impact angle affects gradient direction and magnitude
# ---------------------------------------------------------------------------

class TestImpactAngle:
    """
    The gradient of wall_collision_loss must point into the nearest wall face,
    and must scale correctly with penetration depth.
    """

    @pytest.mark.parametrize("raw_x,raw_y,exp_gx_sign,exp_gy_sign,label", [
        (0.02, 0.50,  -1,  0, "head-on left wall"),
        (0.98, 0.50,  +1,  0, "head-on right wall"),
        (0.50, 0.02,   0, -1, "head-on bottom wall"),
        (0.50, 0.98,   0, +1, "head-on top wall"),
        (0.02, 0.02,  -1, -1, "oblique corner bottom-left"),
    ])
    def test_gradient_direction_at_impact(
        self, raw_x, raw_y, exp_gx_sign, exp_gy_sign, label
    ):
        """
        Loss gradient must point into the wall (sign convention: see module docstring).
        A zero exp_*_sign means that axis should carry no gradient.
        """
        gx, gy = _loss_grad(raw_x, raw_y)
        if exp_gx_sign != 0:
            assert int(jnp.sign(gx)) == exp_gx_sign, (
                f"{label}: gx={gx:.4f}, expected sign {exp_gx_sign:+d}"
            )
        else:
            assert abs(gx) < 1e-5, f"{label}: gx={gx:.6f} should be zero"
        if exp_gy_sign != 0:
            assert int(jnp.sign(gy)) == exp_gy_sign, (
                f"{label}: gy={gy:.4f}, expected sign {exp_gy_sign:+d}"
            )
        else:
            assert abs(gy) < 1e-5, f"{label}: gy={gy:.6f} should be zero"

    def test_gradient_magnitude_scales_with_penetration(self):
        """Deeper penetration must produce a larger gradient magnitude."""
        depths = [0.01, 0.03, 0.06, 0.10]
        magnitudes = []
        for d in depths:
            raw_x = ROBOT_RADIUS - d   # penetrating left wall by depth d
            gx, _ = _loss_grad(raw_x, 0.5)
            magnitudes.append(abs(gx))

        for i in range(len(magnitudes) - 1):
            assert magnitudes[i + 1] > magnitudes[i], (
                f"Gradient should grow with penetration: "
                f"depths={depths}, magnitudes={[f'{m:.4f}' for m in magnitudes]}"
            )

    def test_perpendicular_vs_grazing_impact(self):
        """
        Head-on impact (large x-penetration) must produce a larger x-gradient
        than a grazing approach (tiny x-penetration at the same distance from corner).
        """
        gx_headon, _ = _loss_grad(ROBOT_RADIUS - 0.02, 0.5)   # deep penetration
        gx_grazing, _ = _loss_grad(ROBOT_RADIUS - 0.002, 0.5)  # near-surface approach

        assert abs(gx_headon) > abs(gx_grazing), (
            f"Head-on |gx|={abs(gx_headon):.4f} should exceed "
            f"grazing |gx|={abs(gx_grazing):.4f}"
        )

    def test_collision_loss_gradient_scales_linearly_with_lambda(self):
        """
        The gradient of lambda * wall_collision_loss scales linearly with lambda.
        This is tested directly (not through the scan) for precision.
        """
        pos = jnp.array([0.01, 0.5])   # clearly penetrating left wall

        g1  = grad(lambda p: 1.0  * wall_collision_loss(p[0], p[1], ROBOT_RADIUS, ROBOT_RADIUS))(pos)
        g50 = grad(lambda p: 50.0 * wall_collision_loss(p[0], p[1], ROBOT_RADIUS, ROBOT_RADIUS))(pos)

        ratio = float(jnp.linalg.norm(g50) / jnp.linalg.norm(g1))
        assert abs(ratio - 50.0) < 0.1, (
            f"Gradient should scale 50x with lambda; actual ratio={ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# 4. End-to-end: gradients survive lax.scan
# ---------------------------------------------------------------------------

class TestEndToEndGradient:
    """
    The full simulation objective (inside lax.scan + jax.grad) must retain
    nonzero, finite gradients after the wall-correction fix.
    """

    @pytest.fixture(scope="class")
    def sim(self):
        config = RolloutConfig(n_steps=50, lambda_collision=10.0)
        return DifferentiableSimulation(MotorModel(), WallFollowerPolicy(), config)

    def test_gradient_nonzero_and_finite(self, sim):
        params = jnp.array([0.05, 1.0, 1.0])
        loss, grads = value_and_grad(sim.objective)(params)
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        assert not jnp.any(jnp.isnan(grads)), f"NaN in gradients: {grads}"
        assert jnp.any(grads != 0.0), f"All gradients are zero: {grads}"

    def test_gradient_differs_between_param_sets(self, sim):
        """
        Two very different parameter sets must produce different gradients,
        confirming the optimization landscape is not trivially flat.
        """
        params_slow = jnp.array([0.05, 1.0, 1.0])
        params_fast = jnp.array([0.12, 2.0, 1.0])

        _, g_slow = value_and_grad(sim.objective)(params_slow)
        _, g_fast = value_and_grad(sim.objective)(params_fast)

        assert not jnp.allclose(g_slow, g_fast, atol=1e-6), (
            f"Gradient should differ between param sets "
            f"(g_slow={g_slow}, g_fast={g_fast})"
        )

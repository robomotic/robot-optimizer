"""
Differentiable rollout for co-design optimization.

Implements the main objective function using jax.lax.scan for efficient
unrolling of a simplified differentiable physics simulation.

This is the core of the end-to-end differentiable pipeline.
"""

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, Array

from motor_model import MotorModel
from control_policy import compute_control, WallFollowerPolicy
from walls import apply_wall_correction, wall_collision_loss, ray_wall_distance


class RolloutCarry(NamedTuple):
    """Carry state for differentiable rollout scan."""
    position: Array  # Robot position [x, y, z]
    velocity: Array  # Robot velocity [vx, vy, vz]
    orientation: Array  # Robot orientation (quaternion or angle)
    total_loss: Array  # Accumulated loss
    total_energy: Array  # Accumulated energy cost
    step_count: Array  # Current step count


class RolloutConfig(NamedTuple):
    """Configuration for differentiable rollout."""
    n_steps: int = 500
    goal_pos: Array = jnp.array([0.8, 0.2, 0.04])  # East side, away from walls
    lambda_energy: float = 0.01  # Weight for energy penalty
    lambda_collision: float = 10.0  # Weight for collision penalty
    dt: float = 0.002  # Simulation timestep (s)
    wheel_base: float = 0.08  # Distance between wheels (m)
    max_speed: float = 1.0  # Maximum wheel linear speed (m/s)
    wall_margin: float = 0.05  # Hard collision margin from arena walls (m)


def _compute_sensor_readings(position: Array, orientation: Array, max_range: float = 2.0) -> Array:
    """
    Compute 5 rangefinder distances based on robot pose using raycasting.

    Sensor order matches WallFollowerPolicy:
      [front, front-left (45°), left (90°), front-right (-45°), right (-90°)]

    Checks the outer boundary AND all internal maze walls via walls.ray_wall_distance.
    All operations are JAX-compatible for grad / jit / vmap / scan.
    """
    px, py = position[0], position[1]
    rel = jnp.array([0.0, jnp.pi / 4.0, jnp.pi / 2.0, -jnp.pi / 4.0, -jnp.pi / 2.0])
    abs_angles = orientation + rel
    dxs = jnp.cos(abs_angles)
    dys = jnp.sin(abs_angles)

    # Vectorise ray_wall_distance over the 5 sensor directions
    return jax.vmap(
        lambda dx, dy: ray_wall_distance(px, py, dx, dy, max_range)
    )(dxs, dys)


class DifferentiableSimulation:
    """
    End-to-end differentiable simulation pipeline using simplified physics.
    
    Combines:
    - Design parameter morphing (wheel radius affects max speed/torque)
    - Control policy (soft wall-follower)
    - Simplified motor dynamics (RLC model)
    - Kinematic physics step (differential drive)
    - Loss computation (distance, energy, collision)
    
    All operations are JAX-compatible for automatic differentiation.
    """
    
    def __init__(
        self,
        motor_model: Optional[MotorModel] = None,
        control_policy: Optional[WallFollowerPolicy] = None,
        config: Optional[RolloutConfig] = None,
    ):
        """
        Initialize simulation.
        
        Args:
            motor_model: Motor dynamics model. If None, uses default.
            control_policy: Control policy. If None, uses default.
            config: Rollout configuration. If None, uses defaults.
        """
        self.motor_model = motor_model
        self.control_policy = control_policy
        self.config = config or RolloutConfig()
    
    def step_fn(
        self,
        carry: RolloutCarry,
        args: Tuple[Array],
    ) -> Tuple[RolloutCarry, None]:
        """
        Single simulation step within jax.lax.scan.
        
        Args:
            carry: RolloutCarry with current state and accumulated loss.
            args: Tuple containing (design_params).
        
        Returns:
            Updated carry, None (scan produces no per-step output).
        """
        design_params = args[0]
        position, velocity, orientation, total_loss, total_energy, step_count = carry

        wheel_radius = design_params[0]
        motor_kt = design_params[1]
        battery_rho = design_params[2]

        # --- Sensors: orientation-aware raycasting against arena walls ---
        sensor_readings = _compute_sensor_readings(position, orientation)

        # --- Control policy ---
        control_output = compute_control(
            sensor_readings,
            wheel_radius,
            motor_kt,
            policy=self.control_policy,
            position=position,
            orientation=orientation,
            goal=self.config.goal_pos,
        )

        # --- Differential-drive kinematics ---
        # battery_rho scales achievable peak speed: higher capacity → less voltage sag.
        # Normalized so rho=1.0 gives speed_scale=1.0 (no change from baseline).
        # tanh / tanh(1) bounds the multiplier in (0, ~1.31] as rho → ∞.
        speed_scale = jnp.tanh(battery_rho) / jnp.tanh(jnp.array(1.0))
        v_left  = control_output.left_wheel  * self.config.max_speed * speed_scale
        v_right = control_output.right_wheel * self.config.max_speed * speed_scale

        v_linear  = (v_left + v_right) / 2.0
        v_angular = (v_right - v_left) / self.config.wheel_base

        new_orientation = orientation + v_angular * self.config.dt

        raw_x = position[0] + v_linear * jnp.cos(new_orientation) * self.config.dt
        raw_y = position[1] + v_linear * jnp.sin(new_orientation) * self.config.dt

        # --- Differentiable wall correction (outer + internal walls) ---
        m = self.config.wall_margin
        corrected_x, corrected_y = apply_wall_correction(raw_x, raw_y, m)
        new_position = jnp.array([corrected_x, corrected_y, position[2]])

        new_velocity = jnp.array([
            v_linear * jnp.cos(new_orientation),
            v_linear * jnp.sin(new_orientation),
            0.0,
        ])

        # --- Loss ---
        # 1. Distance-to-goal (quadratic, 2-D)
        delta_xy = new_position[:2] - self.config.goal_pos[:2]
        loss_dist = 0.5 * jnp.sum(delta_xy ** 2)

        # 2. Energy: I²R proxy using control signal magnitude
        energy_cost = (
            jnp.abs(control_output.left_wheel) ** 2
            + jnp.abs(control_output.right_wheel) ** 2
        ) * self.config.dt
        loss_energy = energy_cost

        # 3. Wall-collision: soft penalty for all walls (outer + internal).
        #    Uses raw_x/y (pre-correction) so gradient flows even when the
        #    correction has already clamped the robot at the wall surface.
        loss_collision = wall_collision_loss(raw_x, raw_y, m, m)

        step_loss = (
            loss_dist
            + self.config.lambda_energy    * loss_energy
            + self.config.lambda_collision * loss_collision
        )

        new_carry = RolloutCarry(
            position=new_position,
            velocity=new_velocity,
            orientation=new_orientation,
            total_loss=total_loss + step_loss,
            total_energy=total_energy + energy_cost,
            step_count=step_count + 1,
        )

        return new_carry, None
    
    def objective(
        self,
        design_params: Array,
    ) -> Array:
        """
        Compute objective function (total loss over rollout).
        
        This is the main function called by the optimizer.
        Gradients flow through this via jax.grad().
        
        Args:
            design_params: Design parameters [wheel_radius, motor_kt, battery_rho].
        
        Returns:
            Scalar loss value.
        """
        # Initialize simulation state
        init_position = jnp.array([0.1, 0.1, 0.04])  # Start near bottom-left
        init_velocity = jnp.zeros(3)
        init_orientation = jnp.array(0.0)  # Facing positive x
        
        # Initialize carry
        carry_init = RolloutCarry(
            position=init_position,
            velocity=init_velocity,
            orientation=init_orientation,
            total_loss=jnp.array(0.0),
            total_energy=jnp.array(0.0),
            step_count=jnp.array(0),
        )
        
        # Differentiable rollout using scan.
        # jax.checkpoint (rematerialisation) trades compute for memory: activations from
        # each step are recomputed during the backward pass rather than stored, so the
        # gradient graph stays O(1) in n_steps instead of O(n_steps).  This allows
        # arbitrarily long rollouts without GPU OOM during XLA compilation.
        scan_args = (design_params,)

        def step_fn_partial(carry, unused):
            return self.step_fn(carry, scan_args)

        final_carry, _ = lax.scan(
            jax.checkpoint(step_fn_partial),
            carry_init,
            None,
            length=self.config.n_steps,
        )
        
        return final_carry.total_loss
    
    def rollout_with_visualization(
        self,
        design_params: Array,
    ) -> Dict[str, Array]:
        """
        Run a rollout and return trajectory data for visualization/analysis.
        
        Note: This is NOT differentiable (uses non-scan loop). Use only for
        evaluation, not for optimization.
        
        Args:
            design_params: Design parameters.
        
        Returns:
            Dictionary with trajectory data:
            - 'positions': Nx3 array of robot positions
            - 'distances': N array of distances to goal
            - 'total_loss': scalar loss
        """
        # Initialize state
        position    = jnp.array([0.1, 0.1, 0.04])
        velocity    = jnp.zeros(3)
        orientation = jnp.array(0.0)

        positions = []
        distances = []

        m = self.config.wall_margin

        for _ in range(self.config.n_steps):
            positions.append(position)
            distances.append(jnp.linalg.norm(position[:2] - self.config.goal_pos[:2]))

            # Orientation-aware sensor readings
            sensor_readings = _compute_sensor_readings(position, orientation)

            control = compute_control(
                sensor_readings,
                design_params[0],
                design_params[1],
                policy=self.control_policy,
                position=position,
                orientation=orientation,
                goal=self.config.goal_pos,
            )

            # Correct differential-drive kinematics
            speed_scale = float(jnp.tanh(design_params[2]))
            v_left  = control.left_wheel  * self.config.max_speed * speed_scale
            v_right = control.right_wheel * self.config.max_speed * speed_scale

            v_linear  = (v_left + v_right) / 2.0
            v_angular = (v_right - v_left) / self.config.wheel_base

            orientation = orientation + v_angular * self.config.dt
            raw_x = position[0] + v_linear * jnp.cos(orientation) * self.config.dt
            raw_y = position[1] + v_linear * jnp.sin(orientation) * self.config.dt

            # Differentiable wall correction — outer + internal walls
            corrected_x, corrected_y = apply_wall_correction(raw_x, raw_y, m)
            position = jnp.array([corrected_x, corrected_y, position[2]])
            velocity = jnp.array([
                v_linear * jnp.cos(orientation),
                v_linear * jnp.sin(orientation),
                0.0,
            ])

        return {
            "positions": jnp.array(positions),
            "distances": jnp.array(distances),
            "total_loss": self.objective(design_params),
        }


if __name__ == "__main__":
    print("Differentiable Simulation Module")
    print("=" * 50)
    print("This module provides the core optimization pipeline.")
    print("It is intended to be imported and used by scripts/train.py")

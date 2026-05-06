"""
Differentiable control policy for maze navigation.

Implements a soft wall-follower strategy using sigmoid-based smooth switching.
The policy is conditioned on morphology parameters (wheel_radius, motor_kt)
to enable learning of parameter-dependent strategies.

All operations use JAX primitives for automatic differentiation.
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array


class ControlOutput(NamedTuple):
    """Control output from the policy."""
    left_wheel: Array  # Left wheel command [-1, +1]
    right_wheel: Array  # Right wheel command [-1, +1]
    steer: Array  # Steering signal (raw, before symmetry breaking)


class WallFollowerPolicy:
    """
    Goal-seeking wall-follower control policy.

    Uses sigmoid-based smooth transitions to avoid hard conditionals,
    enabling gradient flow through the control law.

    The policy implements:
    1. Front obstacle avoidance (sigmoid-based turn weight)
    2. Goal-heading: steers toward goal when path is clear
    3. Wall-following fallback: side-distance proportional control
    4. Parameter-conditioned behavior (adapts to wheel radius, motor constant)
    """

    def __init__(
        self,
        forward_speed: float = 0.8,
        turn_gain: float = 5.0,
        turn_threshold: float = 0.3,
        side_target_distance: float = 0.4,
        side_gain: float = 2.0,
        max_turn: float = 0.5,
        goal_gain: float = 1.5,
        goal_mix: float = 0.7,
    ):
        """
        Initialize wall-follower policy.
        
        Args:
            forward_speed: Base forward speed (-1 to +1).
            turn_gain: Sensitivity of turn_weight to front distance (higher = sharper).
            turn_threshold: Distance threshold for triggering turns (m).
            side_target_distance: Target distance from side wall (m).
            side_gain: Proportional control gain for side distance error.
            max_turn: Maximum steering magnitude when turning (0 to 1).
        """
        self.forward_speed = float(forward_speed)
        self.turn_gain = float(turn_gain)
        self.turn_threshold = float(turn_threshold)
        self.side_target_distance = float(side_target_distance)
        self.side_gain = float(side_gain)
        self.max_turn = float(max_turn)
        self.goal_gain = float(goal_gain)
        self.goal_mix = float(goal_mix)
    
    def extract_rangefinder_readings(
        self,
        sensor_data: Array,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Extract individual rangefinder distances.
        
        Assumes sensor data is ordered as:
        [front, front_left, left, front_right, right, ...]
        
        Args:
            sensor_data: Flattened sensor reading array from MJX.
        
        Returns:
            Tuple of (front, front_left, left, front_right, right) distances.
        """
        # Assume first 5 sensors are rangefinders in order
        front = sensor_data[0] if len(sensor_data) > 0 else jnp.array(1.0)
        front_left = sensor_data[1] if len(sensor_data) > 1 else jnp.array(1.0)
        left = sensor_data[2] if len(sensor_data) > 2 else jnp.array(1.0)
        front_right = sensor_data[3] if len(sensor_data) > 3 else jnp.array(1.0)
        right = sensor_data[4] if len(sensor_data) > 4 else jnp.array(1.0)
        
        return front, front_left, left, front_right, right
    
    def compute_turn_weight(
        self,
        front_distance: Array,
    ) -> Array:
        r"""
        Compute soft turn weight based on front obstacle distance.
        
        Uses sigmoid for smooth switching:
        
        .. math::
            w_{turn} = \sigma(g \cdot (d_{threshold} - d_{front}))
        
        where :math:`\sigma` is the sigmoid function.
        
        - If front_distance < turn_threshold: w_turn → 1 (turn strongly)
        - If front_distance > turn_threshold: w_turn → 0 (go forward)
        
        Args:
            front_distance: Distance to front obstacle (m).
        
        Returns:
            Turn weight in [0, 1].
        """
        exponent = self.turn_gain * (self.turn_threshold - front_distance)
        turn_weight = jax.nn.sigmoid(exponent)
        return turn_weight
    
    def compute_steering(
        self,
        turn_weight: Array,
        left_distance: Array,
        right_distance: Array,
        position: Array,
        orientation: Array,
        goal: Array,
    ) -> Array:
        """
        Compute steering command blending goal-seeking, wall-following, and obstacle avoidance.

        When the path is clear (turn_weight ≈ 0):
          steer = goal_mix * goal_steer + (1 - goal_mix) * wall_following_steer
        When obstacle is close (turn_weight ≈ 1):
          steer = turn toward open space
        """
        # Goal-heading: signed angle from current heading to bearing of goal.
        # arctan2(sin(Δ), cos(Δ)) wraps cleanly to [-π, π] — fully differentiable.
        dx = goal[0] - position[0]
        dy = goal[1] - position[1]
        goal_bearing = jnp.arctan2(dy, dx)
        heading_error = goal_bearing - orientation
        heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))
        goal_steer = jnp.clip(self.goal_gain * heading_error, -self.max_turn, self.max_turn)

        # Wall-following: maintain target gap from the left wall.
        left_error = left_distance - self.side_target_distance
        wall_following_steer = jnp.clip(
            self.side_gain * left_error, -self.max_turn, self.max_turn
        )

        # Obstacle avoidance: turn toward the side with more space.
        # Use tanh instead of sign to keep gradient nonzero.
        turn_direction = jnp.tanh(10.0 * (left_distance - right_distance))
        turn_steer = turn_direction * self.max_turn

        # When path clear: blend goal-seeking with wall-following.
        # When blocked: hard turn away from obstacle.
        clear_steer = self.goal_mix * goal_steer + (1.0 - self.goal_mix) * wall_following_steer
        steer = turn_weight * turn_steer + (1.0 - turn_weight) * clear_steer

        return jnp.clip(steer, -self.max_turn, self.max_turn)
    
    def compute_wheel_commands(
        self,
        steer: Array,
        wheel_radius: Array,
        motor_kt: Array,
        goal_brake: Array = None,
    ) -> Tuple[Array, Array]:
        r"""
        Convert steering command to left/right wheel commands.

        Args:
            steer: Steering command in [-max_turn, +max_turn].
            wheel_radius: Wheel radius (m), used to condition steering response.
            motor_kt: Motor torque constant (Nm/A), used to condition steering.
            goal_brake: Speed multiplier in (0, 1] from goal-proximity braking.

        Returns:
            Tuple of (left_wheel_command, right_wheel_command), each in [-1, +1].
        """
        # Base forward speed, reduced near goal so the robot doesn't overshoot
        if goal_brake is not None:
            forward = self.forward_speed * goal_brake
        else:
            forward = self.forward_speed

        # Condition steering on morphology
        # This allows the optimizer to learn that different parameters need
        # different steering behavior.
        
        # Larger wheels may affect how steering commands translate to wheel speeds.
        radius_factor = 1.0 + 0.2 * jnp.tanh((wheel_radius - 0.05) / 0.05)
        
        # Motor constant affects how much steering can be realized for a given torque.
        kt_factor = 1.0 + 0.1 * jnp.tanh((motor_kt - 1.0) / 1.0)
        
        # Scaled steering
        conditioned_steer = steer * radius_factor * kt_factor
        conditioned_steer = jnp.clip(conditioned_steer, -self.max_turn, self.max_turn)

        # Add a small morphology-dependent offset to drive variation even when
        # steering commands are symmetric.
        radius_offset = 0.05 * jnp.tanh((wheel_radius - 0.05) / 0.05)
        
        # Differential drive: left and right wheel speeds
        left_wheel = forward - conditioned_steer + radius_offset
        right_wheel = forward + conditioned_steer - radius_offset
        
        # Normalize to [-1, +1]
        max_speed = jnp.maximum(jnp.abs(left_wheel), jnp.abs(right_wheel))
        max_speed = jnp.maximum(max_speed, 1.0)  # Avoid division by zero
        
        left_wheel = left_wheel / max_speed
        right_wheel = right_wheel / max_speed
        
        return left_wheel, right_wheel
    
    def __call__(
        self,
        sensor_data: Array,
        wheel_radius: Array,
        motor_kt: Array,
        position: Array = None,
        orientation: Array = None,
        goal: Array = None,
    ) -> ControlOutput:
        """
        Compute control output from sensor readings and robot state.

        Args:
            sensor_data: Array of rangefinder distances [front, f-left, left, f-right, right].
            wheel_radius: Wheel radius (m).
            motor_kt: Motor torque constant (Nm/A).
            position: Robot (x, y, z) in sim coords. If None, goal-seeking is disabled.
            orientation: Robot heading angle (rad). If None, goal-seeking is disabled.
            goal: Goal (x, y, z) in sim coords. If None, goal-seeking is disabled.

        Returns:
            ControlOutput with left_wheel, right_wheel, steer commands.
        """
        front, _fl, left, _fr, right = self.extract_rangefinder_readings(sensor_data)

        turn_weight = self.compute_turn_weight(front)

        if position is not None and orientation is not None and goal is not None:
            steer = self.compute_steering(turn_weight, left, right, position, orientation, goal)
            # Smooth braking: full speed when > brake_dist from goal, zero at goal.
            # tanh(x/brake_dist): at x=brake_dist → tanh(1)=0.76; at x=0 → 0.
            goal_dist = jnp.sqrt(jnp.sum((position[:2] - goal[:2]) ** 2) + 1e-8)
            goal_brake = jnp.tanh(goal_dist / 0.15)
        else:
            # Fallback: pure wall-following (no goal info)
            left_error = left - self.side_target_distance
            wall_steer = jnp.clip(
                self.side_gain * left_error, -self.max_turn, self.max_turn
            )
            turn_direction = jnp.tanh(10.0 * (left - right))
            turn_steer = turn_direction * self.max_turn
            steer = turn_weight * turn_steer + (1.0 - turn_weight) * wall_steer
            steer = jnp.clip(steer, -self.max_turn, self.max_turn)
            goal_brake = None

        left_wheel, right_wheel = self.compute_wheel_commands(steer, wheel_radius, motor_kt, goal_brake)

        return ControlOutput(
            left_wheel=left_wheel,
            right_wheel=right_wheel,
            steer=steer,
        )


# Global policy instance
_policy = WallFollowerPolicy()


def compute_control(
    sensor_data: Array,
    wheel_radius: Array,
    motor_kt: Array,
    policy: WallFollowerPolicy = None,
    position: Array = None,
    orientation: Array = None,
    goal: Array = None,
) -> ControlOutput:
    """
    Compute control output from sensor data and optional robot state.

    Args:
        sensor_data: Rangefinder sensor array [front, f-left, left, f-right, right].
        wheel_radius: Wheel radius (m).
        motor_kt: Motor torque constant (Nm/A).
        policy: WallFollowerPolicy instance. If None, uses global default.
        position: Robot (x, y, z) in sim coords — enables goal-seeking when provided.
        orientation: Robot heading angle (rad) — enables goal-seeking when provided.
        goal: Goal (x, y, z) in sim coords — enables goal-seeking when provided.

    Returns:
        ControlOutput with wheel commands and steering signal.
    """
    if policy is None:
        policy = _policy
    return policy(sensor_data, wheel_radius, motor_kt, position, orientation, goal)


if __name__ == "__main__":
    import jax
    
    # Test the policy
    policy = WallFollowerPolicy()
    
    # Simulate rangefinder readings (5 sensors)
    # Front=0.3m (close), Left=0.5m, Right=0.6m (both good)
    sensor_data = jnp.array([0.3, 0.4, 0.5, 0.45, 0.6])
    
    # Design parameters
    wheel_radius = jnp.array(0.05)
    motor_kt = jnp.array(1.0)
    
    print("Wall-Follower Policy Test")
    print("=" * 50)
    print(f"Sensor readings (front, f-left, left, f-right, right):")
    print(f"  {sensor_data}")
    print(f"Wheel radius: {wheel_radius} m")
    print(f"Motor constant (kt): {motor_kt} Nm/A")
    print()
    
    # Compute control
    control = policy(sensor_data, wheel_radius, motor_kt)
    
    print(f"Left wheel command: {control.left_wheel:.4f}")
    print(f"Right wheel command: {control.right_wheel:.4f}")
    print(f"Steering signal: {control.steer:.4f}")
    print()
    
    # Test with different morphology
    wheel_radius_large = jnp.array(0.1)
    control_large = policy(sensor_data, wheel_radius_large, motor_kt)
    
    print(f"With larger wheels ({wheel_radius_large} m):")
    print(f"Left wheel command: {control_large.left_wheel:.4f}")
    print(f"Right wheel command: {control_large.right_wheel:.4f}")
    print(f"Steering signal: {control_large.steer:.4f}")
    print()
    
    # Test JAX differentiability
    def objective(radius):
        """Objective: maximize forward speed with small wheels."""
        control = policy(sensor_data, radius, motor_kt)
        return control.left_wheel + control.right_wheel
    
    grad_radius = jax.grad(objective)(wheel_radius)
    print(f"Gradient of speed w.r.t. wheel_radius: {grad_radius:.6f}")
    print("✓ Gradients computed successfully (JAX differentiable)")

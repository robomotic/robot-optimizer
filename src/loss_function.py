"""
Loss function for co-design optimization.

Computes loss as a weighted sum of:
1. Distance to goal (primary task objective)
2. Energy consumption (efficiency penalty)
3. Collision forces (safety penalty)

All components are differentiable for gradient-based optimization.
"""

from typing import Optional

import jax.numpy as jnp
from jax import Array
from mujoco_compat import mjx


def compute_distance_loss(
    robot_pos: Array,
    goal_pos: Array,
) -> Array:
    r"""
    Compute distance-to-goal loss (quadratic).
    
    .. math::
        L_{dist} = \frac{1}{2} \|\mathbf{p}_{robot} - \mathbf{p}_{goal}\|_2^2
    
    This encourages the robot to reach the goal position.
    
    Args:
        robot_pos: Current robot position [x, y, z].
        goal_pos: Goal position [x, y, z].
    
    Returns:
        Distance loss (scalar).
    """
    delta = robot_pos - goal_pos
    distance = jnp.sqrt(jnp.sum(delta[:2] ** 2))  # Use only x, y (2D distance)
    loss = 0.5 * distance ** 2
    return loss


def compute_energy_loss(
    motor_current_left: Array,
    motor_current_right: Array,
    motor_resistance: float = 5.0,
    dt: float = 0.001,
) -> Array:
    r"""
    Compute energy dissipation loss (resistive heating).
    
    .. math::
        L_{energy} = (I_{left}^2 + I_{right}^2) \cdot R \cdot \Delta t
    
    This encourages the optimizer to find efficient motor control and morphologies.
    
    Args:
        motor_current_left: Left motor current (A).
        motor_current_right: Right motor current (A).
        motor_resistance: Motor winding resistance (Ohms).
        dt: Timestep (seconds).
    
    Returns:
        Energy loss (scalar, Joules per timestep).
    """
    i_squared_total = motor_current_left ** 2 + motor_current_right ** 2
    energy_loss = i_squared_total * motor_resistance * dt
    return energy_loss


def compute_collision_loss(
    data: mjx.Data,
    contact_penalty_scale: float = 1.0,
) -> Array:
    r"""
    Compute collision penalty loss.
    
    .. math::
        L_{collision} = \sum_i \max(0, -d_i)^2
    
    where :math:`d_i` are contact distances (positive = separated, negative = penetration).
    
    This discourages wall collisions and pushing.
    
    Args:
        data: MJX data structure with contact information.
        contact_penalty_scale: Scaling factor for penalty.
    
    Returns:
        Collision loss (scalar).
    """
    # Contact distances: positive = separated, negative = penetrated
    if not hasattr(data, "contact") or data.ncon == 0:
        return jnp.array(0.0)
    
    # Get contact distances
    contact_dist = data.contact.dist if hasattr(data.contact, "dist") else jnp.zeros(data.ncon)
    
    # Penalize penetrations (negative distances)
    penetrations = jnp.maximum(0.0, -contact_dist)
    
    # Sum squared penetrations
    collision_loss = contact_penalty_scale * jnp.sum(penetrations ** 2)
    
    return collision_loss


def compute_step_loss(
    data: mjx.Data,
    robot_pos: Array,
    goal_pos: Array,
    lambda_energy: float = 0.01,
    lambda_collision: float = 10.0,
    motor_current_left: float = 0.0,
    motor_current_right: float = 0.0,
    motor_resistance: float = 5.0,
    dt: float = 0.001,
) -> Array:
    r"""
    Compute total loss for a single timestep.
    
    .. math::
        L = L_{dist} + \lambda_1 L_{energy} + \lambda_2 L_{collision}
    
    Args:
        data: MJX data at current step.
        robot_pos: Current robot position.
        goal_pos: Goal position.
        lambda_energy: Weight for energy penalty.
        lambda_collision: Weight for collision penalty.
        motor_current_left: Left motor current (A).
        motor_current_right: Right motor current (A).
        motor_resistance: Motor resistance (Ohms).
        dt: Timestep (seconds).
    
    Returns:
        Total loss (scalar).
    """
    # Distance loss (primary objective)
    loss_dist = compute_distance_loss(robot_pos, goal_pos)
    
    # Energy loss (efficiency)
    loss_energy = compute_energy_loss(
        motor_current_left,
        motor_current_right,
        motor_resistance=motor_resistance,
        dt=dt,
    )
    
    # Collision loss (safety)
    loss_collision = compute_collision_loss(data)
    
    # Weighted sum
    total_loss = loss_dist + lambda_energy * loss_energy + lambda_collision * loss_collision
    
    return total_loss


def compute_final_loss(
    final_distance: Array,
    total_energy: Array,
    total_collisions: Array,
    lambda_energy: float = 0.01,
    lambda_collision: float = 10.0,
) -> Array:
    r"""
    Compute final loss from accumulated statistics.
    
    Alternative to per-step losses. Can use this for trajectory-level objectives.
    
    .. math::
        L = d_{final} + \lambda_1 E_{total} + \lambda_2 P_{collision}
    
    Args:
        final_distance: Final distance to goal (m).
        total_energy: Total energy consumed (J).
        total_collisions: Total collision penalty.
        lambda_energy: Weight for energy.
        lambda_collision: Weight for collisions.
    
    Returns:
        Total loss.
    """
    loss = (
        final_distance
        + lambda_energy * total_energy
        + lambda_collision * total_collisions
    )
    return loss


class LossConfig:
    """Configuration for loss function weighting."""
    
    def __init__(
        self,
        lambda_energy: float = 0.01,
        lambda_collision: float = 10.0,
        motor_resistance: float = 5.0,
        dt: float = 0.001,
    ):
        """
        Initialize loss configuration.
        
        Args:
            lambda_energy: Weight for energy penalty.
            lambda_collision: Weight for collision penalty.
            motor_resistance: Motor winding resistance (Ohms).
            dt: Simulation timestep (seconds).
        """
        self.lambda_energy = lambda_energy
        self.lambda_collision = lambda_collision
        self.motor_resistance = motor_resistance
        self.dt = dt


if __name__ == "__main__":
    print("Loss Function Module")
    print("=" * 50)
    
    # Test individual loss components
    robot_pos = jnp.array([0.0, 0.0, 0.04])
    goal_pos = jnp.array([0.8, 0.8, 0.04])
    
    dist_loss = compute_distance_loss(robot_pos, goal_pos)
    print(f"Distance loss (robot at origin, goal at NE corner): {dist_loss:.4f}")
    
    energy_loss = compute_energy_loss(
        motor_current_left=jnp.array(1.5),
        motor_current_right=jnp.array(1.5),
    )
    print(f"Energy loss (1.5A per motor): {energy_loss:.6f}")
    
    # Test combined loss
    print("\nWeighted loss example:")
    total_loss = dist_loss + 0.01 * energy_loss
    print(f"  Distance: {dist_loss:.4f}")
    print(f"  Energy (λ=0.01): {0.01 * energy_loss:.6f}")
    print(f"  Total: {total_loss:.4f}")

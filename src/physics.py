"""
MJX physics simulation utilities and model morphing.

Provides:
- Model morphing to apply design parameters (wheel radius, motor constant, battery factor)
- Differentiable state initialization
- Utilities for extracting robot state and computing distances
"""

from typing import Dict, Optional

import jax.numpy as jnp
import mujoco
from mujoco_compat import mjx

from jax import Array

from environment import load_robot_model


# Design parameter scaling factors
WHEEL_RADIUS_SCALE = 0.05  # Base wheel radius (m), default is 5cm
MOTOR_KT_SCALE = 1.0  # Base motor constant
BATTERY_MASS_SCALE = 0.5  # Base battery mass relative to chassis


def _get_raw_model(base_model: mjx.Model) -> mujoco.MjModel:
    """
    Return the underlying MuJoCo model from a MJX model.
    """
    if hasattr(base_model, "_model"):
        return base_model._model
    if hasattr(base_model, "model"):
        return base_model.model
    raise AttributeError("Unable to access underlying MuJoCo model from MJX Model")


def morph_model(
    base_model: mjx.Model,
    design_params: Array,
) -> mjx.Model:
    r"""
    Apply design parameters to the MJX model using immutable `.replace()`.
    
    Design parameters control:
    1. Wheel radius: affects wheel geometry and contact dynamics
    2. Motor constant: affects motor torque (handled in motor_model.py)
    3. Battery mass factor: affects total body mass and internal resistance
    
    .. math::
        wheel\_radius &= \text{params}[0] \\
        motor\_kt &= \text{params}[1] \\
        battery\_rho &= \text{params}[2]
    
    Args:
        base_model: Base MJX model with nominal parameters.
        design_params: Design parameters array [wheel_radius, motor_kt, battery_rho].
                       (motor_kt is handled in motor_model; battery_rho affects mass here)
    
    Returns:
        Morphed MJX model with updated geometry and mass.
    
    Note:
        motor_kt does not directly change the model geometry.
        Scaling happens in the electrical motor model (motor_model.py).
        battery_rho scales the battery mass contribution.
    """
    if len(design_params) < 3:
        raise ValueError(f"Expected 3 design parameters, got {len(design_params)}")
    
    wheel_radius = design_params[0]
    # motor_kt = design_params[1]  # Not used for geometry morphing
    battery_rho = design_params[2]
    
    # Extract geometry indices for wheels
    # Assume left and right wheels are named 'left_wheel_geom' and 'right_wheel_geom'
    try:
        raw_model = _get_raw_model(base_model)
        left_wheel_id = mujoco.mj_name2id(
            raw_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "left_wheel_geom"
        )
        right_wheel_id = mujoco.mj_name2id(
            raw_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "right_wheel_geom"
        )
    except Exception as e:
        raise ValueError(f"Failed to find wheel geometries: {e}")
    
    if left_wheel_id < 0 or right_wheel_id < 0:
        raise ValueError("Wheel geometries not found in model")
    
    # Update wheel radii in geom_size
    # Cylinders have size [radius, length] in the first two components
    new_geom_size = base_model.geom_size.copy()
    new_geom_size = new_geom_size.at[left_wheel_id, 0].set(wheel_radius)
    new_geom_size = new_geom_size.at[right_wheel_id, 0].set(wheel_radius)
    
    # Update body mass for battery
    # Increase the chassis mass by battery factor
    robot_body_id = mujoco.mj_name2id(
        raw_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "robot"
    )
    
    if robot_body_id < 0:
        raise ValueError("Robot body not found in model")
    
    new_body_mass = base_model.body_mass.copy()
    base_chassis_mass = new_body_mass[robot_body_id]
    battery_mass_addition = BATTERY_MASS_SCALE * battery_rho
    new_body_mass = new_body_mass.at[robot_body_id].set(base_chassis_mass + battery_mass_addition)
    
    # Apply morphing via immutable .replace()
    morphed_model = base_model.replace(
        geom_size=new_geom_size,
        body_mass=new_body_mass,
    )
    
    return morphed_model


def init_simulation_state(
    model: mjx.Model,
    start_pos: Optional[Array] = None,
    start_quat: Optional[Array] = None,
) -> mjx.Data:
    """
    Initialize MJX simulation state with sensible defaults.
    
    Args:
        model: MJX model.
        start_pos: Starting position [x, y, z]. If None, uses (0, 0, 0).
        start_quat: Starting quaternion [w, x, y, z]. If None, uses identity.
    
    Returns:
        Initialized MJX data structure ready for stepping.
    """
    data = mjx.Data(model)
    
    # Set robot body position
    if start_pos is None:
        start_pos = jnp.array([-0.8, -0.8, 0.04])  # Southwest corner of maze
    
    if start_quat is None:
        start_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    # Get robot body ID
    raw_model = _get_raw_model(model)
    robot_body_id = mujoco.mj_name2id(
        raw_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "robot"
    )
    
    if robot_body_id >= 0:
        # Update position and orientation
        updated_qpos = data.qpos.at[0:3].set(start_pos).at[3:7].set(start_quat)
        data = data.replace(qpos=updated_qpos)
    
    # Zero out velocities (start from rest)
    data = data.replace(
        qvel=jnp.zeros_like(data.qvel),
    )
    
    return data


def get_robot_state(
    data: mjx.Data,
    model: mjx.Model,
) -> Dict[str, Array]:
    """
    Extract current robot state from MJX data.
    
    Args:
        data: MJX data structure.
        model: MJX model.
    
    Returns:
        Dictionary with:
        - 'position': robot XYZ position [x, y, z]
        - 'velocity': robot linear velocity [vx, vy, vz]
        - 'quat': robot orientation quaternion [w, x, y, z]
        - 'ang_vel': robot angular velocity
        - 'left_wheel_vel': left wheel angular velocity
        - 'right_wheel_vel': right wheel angular velocity
    """
    # Get sensor data if available
    state = {}
    
    # Try to extract position from xpos sensors or qpos
    if len(data.qpos) >= 7:
        state["position"] = data.qpos[0:3]
        state["quat"] = data.qpos[3:7]
    
    if len(data.qvel) >= 6:
        state["velocity"] = data.qvel[0:3]
        state["ang_vel"] = data.qvel[3:6]
    
    # Wheel velocities (joint 0 = left wheel, joint 1 = right wheel)
    if len(data.qvel) >= 8:
        state["left_wheel_vel"] = data.qvel[6]
        state["right_wheel_vel"] = data.qvel[7]
    
    return state


def compute_distance_to_goal(
    robot_pos: Array,
    goal_pos: Array,
) -> Array:
    """
    Compute Euclidean distance from robot to goal.
    
    Args:
        robot_pos: Robot position [x, y, z].
        goal_pos: Goal position [x, y, z].
    
    Returns:
        Distance (scalar).
    """
    delta = robot_pos - goal_pos
    distance = jnp.sqrt(jnp.sum(delta ** 2))
    return distance


def compute_contact_penalty(
    data: mjx.Data,
    contact_threshold: float = 0.0,
) -> Array:
    """
    Compute penalty for collisions.
    
    Penalizes contact forces (pushing into walls is expensive).
    
    Args:
        data: MJX data structure.
        contact_threshold: Contact distance below which penalty applies.
    
    Returns:
        Sum of squared contact forces (scalar).
    """
    # Contact data: positive distance = separation, negative = penetration
    # We penalize penetration (negative distances)
    
    # For now, use a simple approximation:
    # Sum any negative contact distances
    if hasattr(data, "contact"):
        contact_dist = data.contact.dist
        penetration = jnp.maximum(0.0, -contact_dist)  # Only negative distances
        penalty = jnp.sum(penetration ** 2)
    else:
        penalty = jnp.array(0.0)
    
    return penalty


def compute_energy_cost(
    current_left: Array,
    current_right: Array,
    motor_resistance: float = 5.0,
    dt: float = 0.001,
) -> Array:
    r"""
    Compute energy dissipated as heat.
    
    .. math::
        E = I_{left}^2 R \Delta t + I_{right}^2 R \Delta t
    
    Args:
        current_left: Left motor current (A).
        current_right: Right motor current (A).
        motor_resistance: Motor winding resistance (Ohms).
        dt: Timestep (seconds).
    
    Returns:
        Energy cost (scalar).
    """
    energy_left = current_left ** 2 * motor_resistance * dt
    energy_right = current_right ** 2 * motor_resistance * dt
    total_energy = energy_left + energy_right
    return total_energy


def validate_simulation_state(data: mjx.Data) -> bool:
    """
    Check if simulation state is valid (no NaNs, reasonable values).
    
    Args:
        data: MJX data to validate.
    
    Returns:
        True if valid, False otherwise.
    """
    # Check for NaNs
    if jnp.any(jnp.isnan(data.qpos)):
        return False
    if jnp.any(jnp.isnan(data.qvel)):
        return False
    
    # Check position bounds (roughly within simulation area)
    pos = data.qpos[0:3]
    if jnp.any(jnp.abs(pos[0:2]) > 2.0):  # x, y should be within -2 to +2
        return False
    if pos[2] < -1.0 or pos[2] > 1.0:  # z should be roughly above ground
        return False
    
    # Check velocity bounds (no crazy speeds)
    vel = data.qvel[0:3] if len(data.qvel) >= 3 else jnp.array([0, 0, 0])
    if jnp.any(jnp.abs(vel) > 100.0):
        return False
    
    return True


if __name__ == "__main__":
    print("Physics utilities module for MJX integration")
    print("=" * 50)
    
    # Load base robot model
    print("Loading robot model...")
    try:
        base_model = load_robot_model()
        print(f"✓ Loaded robot model: {base_model.name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        exit(1)
    
    # Convert to MJX
    print("Converting to MJX...")
    try:
        mjx_model = mjx.Model.from_mujoco(base_model)
        print(f"✓ Converted to MJX model")
    except Exception as e:
        print(f"✗ Failed to convert: {e}")
        exit(1)
    
    # Test morphing
    print("\nTesting model morphing...")
    design_params = jnp.array([0.05, 1.0, 1.0])  # wheel_radius, kt, battery_rho
    try:
        morphed_model = morph_model(mjx_model, design_params)
        print(f"✓ Model morphed successfully")
        print(f"  Original geom_size: {mjx_model.geom_size[0]}")
        print(f"  Morphed geom_size: {morphed_model.geom_size[0]}")
    except Exception as e:
        print(f"✗ Morphing failed: {e}")
    
    # Test initialization
    print("\nTesting simulation state initialization...")
    try:
        data = init_simulation_state(mjx_model)
        print(f"✓ Simulation state initialized")
        print(f"  Robot position: {data.qpos[0:3]}")
        print(f"  Valid state: {validate_simulation_state(data)}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")

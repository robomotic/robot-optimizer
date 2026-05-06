"""
Environment utilities for loading and parsing MJCF models.

Provides functions to:
- Load robot and maze models from MJCF files
- Extract sensor and actuator indices
- Initialize simulation states
- Parse model metadata for differentiable simulation
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco


def get_model_dir() -> Path:
    """Get the models directory path."""
    return Path(__file__).parent.parent / "models"


def load_robot_model() -> mujoco.MjModel:
    """
    Load the robot MJCF model.
    
    Returns:
        mujoco.MjModel: Compiled MuJoCo model of the robot.
    
    Raises:
        FileNotFoundError: If robot.xml is not found.
    """
    model_path = get_model_dir() / "robot.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"Robot model not found at {model_path}")
    
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model


def load_maze_model() -> mujoco.MjModel:
    """
    Load the maze MJCF model.
    
    Returns:
        mujoco.MjModel: Compiled MuJoCo model of the maze (static walls).
    
    Raises:
        FileNotFoundError: If maze.xml is not found.
    """
    model_path = get_model_dir() / "maze.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"Maze model not found at {model_path}")
    
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model


def load_combined_model(include_maze: bool = True) -> mujoco.MjModel:
    """
    Load a combined robot + maze model by merging XML.
    
    Args:
        include_maze: If True, include maze walls in the model.
    
    Returns:
        mujoco.MjModel: Combined model.
    
    Note:
        This is a placeholder. For now, we'll use the robot model with
        assumptions about the maze being in a separate scene.
    """
    if include_maze:
        # For now, load robot model. Future: merge robot + maze XML
        return load_robot_model()
    else:
        return load_robot_model()


def get_actuator_info(model: mujoco.MjModel) -> Dict[str, Any]:
    """
    Extract actuator information from the model.
    
    Args:
        model: MuJoCo model.
    
    Returns:
        Dictionary with actuator names and indices.
    
    Example:
        >>> info = get_actuator_info(model)
        >>> info['names']  # ['left_wheel_motor', 'right_wheel_motor']
        >>> info['indices']  # {0: 'left_wheel_motor', 1: 'right_wheel_motor'}
    """
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(model.nu)]
    
    return {
        "names": actuator_names,
        "indices": {i: name for i, name in enumerate(actuator_names)},
        "count": model.nu,
    }


def get_sensor_info(model: mujoco.MjModel) -> Dict[str, Any]:
    """
    Extract sensor information from the model.
    
    Args:
        model: MuJoCo model.
    
    Returns:
        Dictionary with sensor names, types, and indices.
    """
    sensor_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                    for i in range(model.nsensor)]
    
    sensors = {}
    for i, name in enumerate(sensor_names):
        sensor = model.sensor(name)
        sensors[name] = {
            "index": i,
            "type": mujoco.mjtSensor(sensor.type).name,
            "objtype": mujoco.mjtObj(sensor.objtype).name,
            "objname": mujoco.mj_id2name(model, sensor.objtype, sensor.objid),
        }
    
    return sensors


def get_rangefinder_sensors(model: mujoco.MjModel) -> Dict[str, int]:
    """
    Extract rangefinder sensor indices (distance sensors).
    
    Args:
        model: MuJoCo model.
    
    Returns:
        Dictionary mapping sensor names to their data indices.
    
    Note:
        Assumes rangefinder sensors are named 'sensor_front', 'sensor_left', etc.
    """
    rangefinder_names = [
        "sensor_front",
        "sensor_front_left",
        "sensor_left",
        "sensor_front_right",
        "sensor_right",
    ]
    
    rangefinders = {}
    for i, name in enumerate(rangefinder_names):
        try:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id >= 0:
                # Find the data index for this sensor
                # Sensor data starts at adr_nrange for distance sensors
                rangefinders[name] = i
        except (ValueError, AttributeError):
            pass
    
    if not rangefinders:
        raise ValueError(f"No rangefinder sensors found. Expected one of: {rangefinder_names}")
    
    return rangefinders


def get_body_info(model: mujoco.MjModel, body_name: str) -> Dict[str, Any]:
    """
    Extract body information.
    
    Args:
        model: MuJoCo model.
        body_name: Name of the body (e.g., "robot", "left_wheel").
    
    Returns:
        Dictionary with body properties.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model")
    
    return {
        "name": body_name,
        "id": body_id,
        "mass": model.body_mass[body_id],
    }


def get_geom_info(model: mujoco.MjModel, geom_name: str) -> Dict[str, Any]:
    """
    Extract geometry information.
    
    Args:
        model: MuJoCo model.
        geom_name: Name of the geometry (e.g., "left_wheel_geom").
    
    Returns:
        Dictionary with geom properties (size, type, etc.).
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"Geometry '{geom_name}' not found in model")
    
    geom = model.geom(geom_name)
    return {
        "name": geom_name,
        "id": geom_id,
        "type": mujoco.mjtGeom(geom.type).name,
        "size": geom.size[:3],  # size array
        "pos": geom.pos,
        "body_id": geom.bodyid,
    }


def validate_model_structure(model: mujoco.MjModel) -> bool:
    """
    Validate that the model has the expected structure.
    
    Checks for:
    - 2 actuators (left and right wheels)
    - Expected joint types
    - Expected sensor names
    
    Args:
        model: MuJoCo model to validate.
    
    Returns:
        True if valid, False otherwise.
    
    Raises:
        ValueError: If critical components are missing.
    """
    # Check actuators
    if model.nu < 2:
        raise ValueError(f"Expected at least 2 actuators, found {model.nu}")
    
    actuator_info = get_actuator_info(model)
    expected_actuators = ["left_wheel_motor", "right_wheel_motor"]
    for expected in expected_actuators:
        if expected not in actuator_info["names"]:
            raise ValueError(f"Expected actuator '{expected}' not found")
    
    # Check rangefinder sensors
    try:
        rangefinders = get_rangefinder_sensors(model)
        if len(rangefinders) < 3:
            raise ValueError(f"Expected at least 3 rangefinders, found {len(rangefinders)}")
    except ValueError as e:
        raise ValueError(f"Rangefinder validation failed: {e}")
    
    # Check bodies
    try:
        get_body_info(model, "robot")
        get_body_info(model, "left_wheel")
        get_body_info(model, "right_wheel")
    except ValueError as e:
        raise ValueError(f"Body validation failed: {e}")
    
    return True


def print_model_info(model: mujoco.MjModel, verbose: bool = False) -> None:
    """
    Print comprehensive model information.
    
    Args:
        model: MuJoCo model.
        verbose: If True, print detailed sensor and actuator info.
    """
    print(f"Model: {model.name}")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of geoms: {model.ngeom}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of sensors: {model.nsensor}")
    print(f"Number of contacts: {model.nconmax}")
    
    if verbose:
        print("\nActuators:")
        actuator_info = get_actuator_info(model)
        for name in actuator_info["names"]:
            print(f"  - {name}")
        
        print("\nSensors:")
        sensor_info = get_sensor_info(model)
        for name, info in sensor_info.items():
            print(f"  - {name}: {info['type']}")
        
        print("\nRangefinders:")
        try:
            rangefinders = get_rangefinder_sensors(model)
            for name in rangefinders:
                print(f"  - {name}")
        except ValueError:
            print("  (none found)")


if __name__ == "__main__":
    # Test loading and validation
    robot = load_robot_model()
    print("Robot model loaded successfully")
    print_model_info(robot, verbose=True)
    
    print("\nValidating model structure...")
    if validate_model_structure(robot):
        print("✓ Model structure is valid")

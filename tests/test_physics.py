"""
Unit tests for physics utilities and MJX model morphing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import jax.numpy as jnp
import mujoco
from mujoco_compat import mjx

from environment import load_robot_model
from physics import morph_model, init_simulation_state, compute_distance_to_goal


@pytest.fixture
def mjx_model():
    base_model = load_robot_model()
    return mjx.Model.from_mujoco(base_model)


def test_morph_model_changes_wheel_radius(mjx_model):
    params = jnp.array([0.08, 1.0, 1.0])
    morphed = morph_model(mjx_model, params)

    assert morphed.geom_size.shape == mjx_model.geom_size.shape
    assert morphed.body_mass.shape == mjx_model.body_mass.shape

    left_id = mujoco.mj_name2id(
        morphed.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_wheel_geom",
    )
    right_id = mujoco.mj_name2id(
        morphed.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "right_wheel_geom",
    )

    assert left_id >= 0
    assert right_id >= 0
    assert morphed.geom_size[left_id, 0] == params[0]
    assert morphed.geom_size[right_id, 0] == params[0]


def test_init_simulation_state(mjx_model):
    data = init_simulation_state(mjx_model)
    assert data.qvel is not None
    assert data.qpos is not None
    assert len(data.qpos) >= 7
    assert len(data.qvel) >= 8


def test_distance_to_goal():
    pos = jnp.array([0.0, 0.0, 0.04])
    goal = jnp.array([0.8, 0.8, 0.04])
    dist = compute_distance_to_goal(pos, goal)
    assert dist > 0.0
    assert pytest.approx(dist, rel=1e-3) == jnp.sqrt(0.8**2 + 0.8**2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

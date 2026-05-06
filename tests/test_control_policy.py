"""
Unit tests for control policy.

Verifies policy output shape, smooth blending, and gradient flow.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from control_policy import WallFollowerPolicy, compute_control


class TestControlPolicy:
    @pytest.fixture
    def policy(self):
        return WallFollowerPolicy()

    def test_control_output_shape(self, policy):
        sensor_data = jnp.array([0.5, 0.5, 0.4, 0.5, 0.4])
        wheel_radius = jnp.array(0.05)
        motor_kt = jnp.array(1.0)

        output = policy(sensor_data, wheel_radius, motor_kt)
        assert hasattr(output, "left_wheel")
        assert hasattr(output, "right_wheel")
        assert hasattr(output, "steer")
        assert output.left_wheel.shape == ()
        assert output.right_wheel.shape == ()
        assert output.steer.shape == ()

    def test_forward_speed_behavior(self, policy):
        sensor_data = jnp.array([1.0, 1.0, 0.8, 1.0, 0.8])
        wheel_radius = jnp.array(0.05)
        motor_kt = jnp.array(1.0)

        output = policy(sensor_data, wheel_radius, motor_kt)
        assert -1.0 <= output.left_wheel <= 1.0
        assert -1.0 <= output.right_wheel <= 1.0

    def test_turn_weight_when_obstacle_close(self, policy):
        sensor_data = jnp.array([0.1, 0.2, 0.3, 0.2, 0.4])
        wheel_radius = jnp.array(0.05)
        motor_kt = jnp.array(1.0)

        output = policy(sensor_data, wheel_radius, motor_kt)
        assert jnp.abs(output.steer) <= policy.max_turn

    def test_larger_wheel_radius_changes_output(self, policy):
        sensor_data = jnp.array([0.3, 0.3, 0.4, 0.3, 0.4])
        small_radius = jnp.array(0.03)
        large_radius = jnp.array(0.08)
        motor_kt = jnp.array(1.0)

        output_small = policy(sensor_data, small_radius, motor_kt)
        output_large = policy(sensor_data, large_radius, motor_kt)

        assert output_small.left_wheel != output_large.left_wheel or output_small.right_wheel != output_large.right_wheel

    def test_gradient_flow(self, policy):
        sensor_data = jnp.array([0.3, 0.35, 0.4, 0.35, 0.45])
        wheel_radius = jnp.array(0.05)
        motor_kt = jnp.array(1.0)

        def loss_fn(params):
            output = policy(sensor_data, params[0], params[1])
            return output.left_wheel + output.right_wheel

        grads = grad(loss_fn)(jnp.array([wheel_radius, motor_kt]))
        assert grads.shape == (2,)
        assert not jnp.any(jnp.isnan(grads))

    def test_compute_control_convenience(self):
        sensor_data = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        wheel_radius = jnp.array(0.04)
        motor_kt = jnp.array(0.8)

        output = compute_control(sensor_data, wheel_radius, motor_kt)
        assert -1.0 <= output.left_wheel <= 1.0
        assert -1.0 <= output.right_wheel <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

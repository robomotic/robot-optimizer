"""
Unit tests for motor model.

Tests Back-EMF calculation, current computation, torque generation,
and gradient flow for automatic differentiation.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motor_model import MotorModel, compute_motor_torque, compute_current, compute_back_emf


class TestMotorModel:
    """Test suite for MotorModel class."""
    
    @pytest.fixture
    def motor(self):
        """Create motor model instance."""
        return MotorModel()
    
    def test_initialization(self, motor):
        """Test motor initialization with default parameters."""
        assert motor.battery_voltage == 12.0
        assert motor.gear_ratio == 5.0
        assert motor.motor_resistance == 5.0
        assert motor.motor_max_current == 2.0
    
    def test_back_emf_at_zero_speed(self, motor):
        """Back-EMF should be zero at zero angular velocity."""
        omega = jnp.array(0.0)
        kt = jnp.array(1.0)
        v_emf = motor.compute_back_emf(omega, kt)
        assert jnp.allclose(v_emf, 0.0)
    
    def test_back_emf_proportional_to_speed(self, motor):
        """Back-EMF should be proportional to angular velocity."""
        omega1 = jnp.array(10.0)
        omega2 = jnp.array(20.0)
        kt = jnp.array(1.0)
        
        v_emf1 = motor.compute_back_emf(omega1, kt)
        v_emf2 = motor.compute_back_emf(omega2, kt)
        
        # Ratio should be approximately 1:2
        ratio = v_emf2 / v_emf1
        assert jnp.allclose(ratio, 2.0, rtol=0.01)
    
    def test_current_at_zero_speed(self, motor):
        """At zero speed, current should depend only on control and resistance."""
        control = jnp.array(0.5)
        omega = jnp.array(0.0)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        current = motor.compute_current(control, omega, kt, rho)
        
        # Should be positive
        assert current > 0.0
        # Should be less than max
        assert current < motor.motor_max_current
    
    def test_current_decreases_with_speed(self, motor):
        """Current should decrease as Back-EMF increases with speed."""
        control = jnp.array(0.5)
        omega_low = jnp.array(0.0)
        omega_high = jnp.array(50.0)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        current_low = motor.compute_current(control, omega_low, kt, rho)
        current_high = motor.compute_current(control, omega_high, kt, rho)
        
        # High speed should have lower current (Back-EMF opposes)
        assert current_low > current_high
    
    def test_current_saturation(self, motor):
        """Current should saturate near motor_max_current."""
        control = jnp.array(1.0)  # Full throttle
        omega = jnp.array(0.0)  # At zero speed (max current)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        current = motor.compute_current(control, omega, kt, rho)
        
        # Should be close to but not exceed max
        assert current <= motor.motor_max_current
        assert current >= 0.8 * motor.motor_max_current  # Close to saturation
    
    def test_torque_proportional_to_current(self, motor):
        """Torque should be proportional to current and kt."""
        control = jnp.array(0.5)
        omega = jnp.array(0.0)
        kt1 = jnp.array(1.0)
        kt2 = jnp.array(2.0)
        rho = jnp.array(1.0)
        
        tau1 = motor.compute_torque(control, omega, kt1, rho)
        tau2 = motor.compute_torque(control, omega, kt2, rho)
        
        # tau2 should be roughly 2x tau1
        ratio = tau2 / tau1
        assert jnp.allclose(ratio, 2.0, rtol=0.1)
    
    def test_negative_control(self, motor):
        """Negative control should produce negative torque."""
        control_pos = jnp.array(0.5)
        control_neg = jnp.array(-0.5)
        omega = jnp.array(0.0)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        tau_pos = motor.compute_torque(control_pos, omega, kt, rho)
        tau_neg = motor.compute_torque(control_neg, omega, kt, rho)
        
        assert tau_pos > 0.0
        assert tau_neg < 0.0
        assert jnp.allclose(jnp.abs(tau_pos), jnp.abs(tau_neg))
    
    def test_gradient_flow(self, motor):
        """Gradients should flow through motor torque computation."""
        control = jnp.array(0.5)
        omega = jnp.array(10.0)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        # Define loss as torque (for testing)
        def loss_fn(params):
            return motor.compute_torque(control, omega, params[0], params[1])
        
        params = jnp.array([kt, rho])
        grads = grad(loss_fn)(params)
        
        # Gradients should exist and be non-zero
        assert not jnp.any(jnp.isnan(grads))
        assert jnp.abs(grads[0]) > 0.0  # Gradient w.r.t. kt
    
    def test_internal_resistance_scaling(self, motor):
        """Higher battery factor should reduce internal resistance."""
        omega = jnp.array(0.0)
        control = jnp.array(0.5)
        kt = jnp.array(1.0)
        
        rho_low = jnp.array(0.5)
        rho_high = jnp.array(2.0)
        
        current_low_rho = motor.compute_current(control, omega, kt, rho_low)
        current_high_rho = motor.compute_current(control, omega, kt, rho_high)
        
        # Higher rho (lower resistance) should allow higher current
        assert current_high_rho > current_low_rho


class TestModuleFunctions:
    """Test module-level convenience functions."""
    
    def test_compute_motor_torque_function(self):
        """Test module-level compute_motor_torque function."""
        control = jnp.array(0.5)
        omega = jnp.array(10.0)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        torque = compute_motor_torque(control, omega, kt, rho)
        
        assert isinstance(torque, jax.Array)
        assert not jnp.isnan(torque)
    
    def test_compute_current_function(self):
        """Test module-level compute_current function."""
        control = jnp.array(0.5)
        omega = jnp.array(10.0)
        kt = jnp.array(1.0)
        rho = jnp.array(1.0)
        
        current = compute_current(control, omega, kt, rho)
        
        assert isinstance(current, jax.Array)
        assert current > 0.0
    
    def test_compute_back_emf_function(self):
        """Test module-level compute_back_emf function."""
        omega = jnp.array(50.0)
        kt = jnp.array(1.0)
        
        v_emf = compute_back_emf(omega, kt)
        
        assert isinstance(v_emf, jax.Array)
        assert jnp.allclose(v_emf, 50.0, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

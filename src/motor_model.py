"""
Electrical motor model with RLC dynamics.

Implements a differentiable motor model that computes motor torque from:
- Motor control input (normalized -1 to +1)
- Angular velocity (for Back-EMF feedback)
- Design parameters (motor constant, internal resistance)

All operations use JAX primitives for automatic differentiation.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import Array


# Motor constants (tuned for 2-wheel robot)
BATTERY_VOLTAGE = 12.0  # Nominal battery voltage (V)
GEAR_RATIO = 5.0  # Motor gear reduction ratio
MOTOR_RESISTANCE = 5.0  # Base motor winding resistance (Ohms)
BACK_EMF_CONSTANT_RATIO = 1.0  # k_e / k_t ratio (typically ~1.0)
MOTOR_MAX_CURRENT = 2.0  # Maximum motor current (A) before saturation


class MotorModel:
    """
    Differentiable RLC motor model for JAX.
    
    The motor model computes achievable torque based on:
    1. Back-EMF voltage from current velocity
    2. Voltage sag from internal battery resistance
    3. Current saturation (smooth approximation)
    4. Torque scaling with motor constant and gear ratio
    
    All operations are differentiable through JAX operations.
    """
    
    def __init__(
        self,
        battery_voltage: float = BATTERY_VOLTAGE,
        gear_ratio: float = GEAR_RATIO,
        motor_resistance: float = MOTOR_RESISTANCE,
        motor_max_current: float = MOTOR_MAX_CURRENT,
    ):
        """
        Initialize motor model parameters.
        
        Args:
            battery_voltage: Nominal battery voltage (V).
            gear_ratio: Motor gear reduction ratio.
            motor_resistance: Base motor winding resistance (Ohms).
            motor_max_current: Maximum current before saturation (A).
        """
        self.battery_voltage = float(battery_voltage)
        self.gear_ratio = float(gear_ratio)
        self.motor_resistance = float(motor_resistance)
        self.motor_max_current = float(motor_max_current)
    
    def compute_back_emf(
        self,
        omega: Array,
        motor_kt: Array,
    ) -> Array:
        r"""
        Compute Back-EMF voltage.
        
        .. math::
            V_{emf} = k_e \cdot \omega
        
        where :math:`k_e \approx k_t` (motor torque constant).
        
        Args:
            omega: Angular velocity (rad/s).
            motor_kt: Motor torque constant (Nm/A).
        
        Returns:
            Back-EMF voltage (V).
        """
        ke = motor_kt * BACK_EMF_CONSTANT_RATIO  # k_e ≈ k_t
        v_emf = ke * omega
        return v_emf
    
    def compute_internal_resistance(
        self,
        battery_rho: Array,
    ) -> Array:
        r"""
        Compute internal battery resistance (scales with battery factor).
        
        .. math::
            R_{internal}(\rho) = R_{base} \cdot (1 + \delta(\rho))
        
        where :math:`\rho` is the battery mass factor.
        Higher :math:`\rho` allows higher current capability.
        
        Args:
            battery_rho: Battery mass factor (dimensionless).
        
        Returns:
            Internal resistance (Ohms).
        """
        # Inverse relationship: higher rho -> lower internal resistance.
        # The scale factor keeps the circuit responsive enough to reach
        # near-saturation current at full throttle.
        r_internal = self.motor_resistance * 0.2 / (battery_rho + 0.1)
        return r_internal
    
    def compute_current(
        self,
        control: Array,
        omega: Array,
        motor_kt: Array,
        battery_rho: Array,
    ) -> Array:
        r"""
        Compute motor current with voltage sag and saturation.
        
        .. math::
            V_{eff} &= (V_{bat} \cdot u) - V_{emf} \\
            I &= V_{eff} / (R_{motor} + R_{internal}(\rho))
        
        Current is saturated smoothly using :math:`\tanh` to keep gradients flowing.
        
        Args:
            control: Motor control input (normalized, -1 to +1).
            omega: Angular velocity (rad/s).
            motor_kt: Motor torque constant (Nm/A).
            battery_rho: Battery mass factor (dimensionless).
        
        Returns:
            Motor current (A), clipped to [-I_max, +I_max].
        """
        # Back-EMF opposes the applied voltage
        v_emf = self.compute_back_emf(omega, motor_kt)
        
        # Effective voltage across motor resistance.
        # The control signal is amplified to reflect motor drive behavior
        # before voltage sag and Back-EMF.
        v_control = self.battery_voltage * control * 2.0
        v_eff = v_control - v_emf
        
        # Total resistance (motor + internal)
        r_internal = self.compute_internal_resistance(battery_rho)
        r_total = self.motor_resistance + r_internal
        
        # Current (Ohm's law)
        # Avoid division by zero
        current = v_eff / (r_total + 1e-6)
        
        # Smooth saturation using tanh
        # I_sat = I_max * tanh(I / I_sat)
        # This keeps gradients flowing for optimization
        saturation_factor = jnp.tanh(current / (self.motor_max_current + 1e-6))
        current_saturated = self.motor_max_current * saturation_factor
        
        return current_saturated
    
    def compute_torque(
        self,
        control: Array,
        omega: Array,
        motor_kt: Array,
        battery_rho: Array,
    ) -> Array:
        r"""
        Compute final motor torque.
        
        .. math::
            \tau = I \cdot k_t \cdot \text{GearRatio}
        
        Args:
            control: Motor control input (normalized, -1 to +1).
            omega: Angular velocity (rad/s).
            motor_kt: Motor torque constant (Nm/A).
            battery_rho: Battery mass factor (dimensionless).
        
        Returns:
            Motor torque (Nm).
        """
        current = self.compute_current(control, omega, motor_kt, battery_rho)
        torque = current * motor_kt * self.gear_ratio
        return torque
    
    def __call__(
        self,
        control: Array,
        omega: Array,
        motor_kt: Array,
        battery_rho: Array,
    ) -> Array:
        """
        Compute motor torque (wrapper for compute_torque).
        
        Args:
            control: Motor control input (-1 to +1).
            omega: Angular velocity (rad/s).
            motor_kt: Motor torque constant (Nm/A).
            battery_rho: Battery mass factor (dimensionless).
        
        Returns:
            Motor torque (Nm).
        """
        return self.compute_torque(control, omega, motor_kt, battery_rho)


# Global motor model instance
_motor_model = MotorModel()


def compute_motor_torque(
    control: Array,
    omega: Array,
    motor_kt: Array,
    battery_rho: Array,
    motor_model: MotorModel = None,
) -> Array:
    """
    Compute motor torque from control and state.
    
    Convenience function for differentiable simulation.
    
    Args:
        control: Motor control input (normalized, -1 to +1).
        omega: Angular velocity (rad/s).
        motor_kt: Motor torque constant (Nm/A).
        battery_rho: Battery mass factor (dimensionless).
        motor_model: MotorModel instance. If None, uses global default.
    
    Returns:
        Motor torque (Nm).
    """
    if motor_model is None:
        motor_model = _motor_model
    return motor_model.compute_torque(control, omega, motor_kt, battery_rho)


def compute_current(
    control: Array,
    omega: Array,
    motor_kt: Array,
    battery_rho: Array,
    motor_model: MotorModel = None,
) -> Array:
    """
    Compute motor current from control and state.
    
    Args:
        control: Motor control input (normalized, -1 to +1).
        omega: Angular velocity (rad/s).
        motor_kt: Motor torque constant (Nm/A).
        battery_rho: Battery mass factor (dimensionless).
        motor_model: MotorModel instance. If None, uses global default.
    
    Returns:
        Motor current (A).
    """
    if motor_model is None:
        motor_model = _motor_model
    return motor_model.compute_current(control, omega, motor_kt, battery_rho)


def compute_back_emf(
    omega: Array,
    motor_kt: Array,
    motor_model: MotorModel = None,
) -> Array:
    """
    Compute Back-EMF voltage.
    
    Args:
        omega: Angular velocity (rad/s).
        motor_kt: Motor torque constant (Nm/A).
        motor_model: MotorModel instance. If None, uses global default.
    
    Returns:
        Back-EMF voltage (V).
    """
    if motor_model is None:
        motor_model = _motor_model
    return motor_model.compute_back_emf(omega, motor_kt)


if __name__ == "__main__":
    # Simple test
    motor = MotorModel()
    
    # Test case: motor at rest (omega=0), 50% throttle, base parameters
    control = jnp.array(0.5)
    omega = jnp.array(0.0)
    motor_kt = jnp.array(1.0)
    battery_rho = jnp.array(1.0)
    
    print("Motor Model Test")
    print("=" * 50)
    print(f"Control: {control}")
    print(f"Angular velocity: {omega} rad/s")
    print(f"Motor constant (kt): {motor_kt} Nm/A")
    print(f"Battery factor (rho): {battery_rho}")
    print()
    
    v_emf = motor.compute_back_emf(omega, motor_kt)
    current = motor.compute_current(control, omega, motor_kt, battery_rho)
    torque = motor.compute_torque(control, omega, motor_kt, battery_rho)
    
    print(f"Back-EMF: {v_emf:.4f} V")
    print(f"Current: {current:.4f} A")
    print(f"Torque: {torque:.4f} Nm")
    print()
    
    # Test with high speed (Back-EMF reduces current)
    omega_high = jnp.array(50.0)
    current_high = motor.compute_current(control, omega_high, motor_kt, battery_rho)
    torque_high = motor.compute_torque(control, omega_high, motor_kt, battery_rho)
    
    print(f"At high speed ({omega_high} rad/s):")
    print(f"Current: {current_high:.4f} A")
    print(f"Torque: {torque_high:.4f} Nm")
    print()
    
    # Test JAX differentiability
    import jax
    
    def objective(kt):
        """Simple objective: maximize torque at control=0.5, omega=10."""
        return motor.compute_torque(
            jnp.array(0.5),
            jnp.array(10.0),
            kt,
            jnp.array(1.0)
        )
    
    grad_kt = jax.grad(objective)(motor_kt)
    print(f"Gradient of torque w.r.t. kt: {grad_kt:.6f}")
    print("✓ Gradients computed successfully (JAX differentiable)")

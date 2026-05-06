#!/usr/bin/env python3
"""
Parameter Landscape Analysis Script.

Evaluates the loss function over a grid of parameter combinations
and creates visualizations of the optimization landscape.
"""

import sys
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motor_model import MotorModel
from control_policy import WallFollowerPolicy
from simulation import DifferentiableSimulation, RolloutConfig


def evaluate_parameter_landscape(
    center_params: jnp.ndarray = None,
    n_samples: int = 10,
    param_ranges: dict = None,
) -> dict:
    """
    Evaluate the parameter landscape around given parameters.
    
    Args:
        center_params: Center parameters for the landscape. If None, uses defaults.
        n_samples: Number of samples per parameter dimension.
        param_ranges: Dictionary with parameter ranges. If None, uses defaults.
    
    Returns:
        Dictionary with landscape data for plotting.
    """
    if center_params is None:
        center_params = jnp.array([0.05, 1.0, 1.0])  # Default center
    
    if param_ranges is None:
        param_ranges = {
            'wheel_radius': (0.01, 0.15),  # m
            'motor_kt': (0.1, 2.0),        # Nm/A
            'battery_rho': (0.1, 2.0),     # dimensionless
        }
    
    print(f"\n{'='*70}")
    print(f"Parameter Landscape Analysis")
    print(f"{'='*70}")
    print(f"Center parameters: wheel_radius={center_params[0]:.4f}, kt={center_params[1]:.4f}, rho={center_params[2]:.4f}")
    print(f"Sampling {n_samples}x{n_samples}x{n_samples} = {n_samples**3} points")
    print(f"Parameter ranges: {param_ranges}")
    
    # Create simulation for evaluation
    config = RolloutConfig(n_steps=500)
    motor_model = MotorModel()
    control_policy = WallFollowerPolicy()
    
    simulation = DifferentiableSimulation(
        motor_model=motor_model,
        control_policy=control_policy,
        config=config,
    )
    
    # Create parameter grids
    wheel_radii = jnp.linspace(param_ranges['wheel_radius'][0], param_ranges['wheel_radius'][1], n_samples)
    motor_kts = jnp.linspace(param_ranges['motor_kt'][0], param_ranges['motor_kt'][1], n_samples)
    battery_rhos = jnp.linspace(param_ranges['battery_rho'][0], param_ranges['battery_rho'][1], n_samples)
    
    # Evaluate landscape
    losses = []
    param_combinations = []
    
    print("Evaluating parameter combinations...")
    for i, wr in enumerate(wheel_radii):
        for j, kt in enumerate(motor_kts):
            for k, rho in enumerate(battery_rhos):
                params = jnp.array([wr, kt, rho])
                loss = simulation.objective(params)
                losses.append(float(loss))
                param_combinations.append(params)
                
                if (i * n_samples * n_samples + j * n_samples + k) % max(1, n_samples**3 // 10) == 0:
                    progress = (i * n_samples * n_samples + j * n_samples + k + 1) / n_samples**3 * 100
                    print(f"  Progress: {progress:5.1f}% ({i * n_samples * n_samples + j * n_samples + k + 1}/{n_samples**3})")
    
    losses = jnp.array(losses)
    param_combinations = jnp.array(param_combinations)
    
    # Find best in landscape
    best_idx = jnp.argmin(losses)
    landscape_best_params = param_combinations[best_idx]
    landscape_best_loss = losses[best_idx]
    
    print(f"Landscape best loss: {landscape_best_loss:.4f}")
    print(f"Landscape best params: wheel_radius={landscape_best_params[0]:.4f}, kt={landscape_best_params[1]:.4f}, rho={landscape_best_params[2]:.4f}")
    
    return {
        'wheel_radii': wheel_radii,
        'motor_kts': motor_kts,
        'battery_rhos': battery_rhos,
        'losses': losses,
        'param_combinations': param_combinations,
        'center_params': center_params,
        'landscape_best_params': landscape_best_params,
        'landscape_best_loss': landscape_best_loss,
    }


def plot_parameter_landscape(
    landscape_data: dict,
    save_path: Path = None,
) -> None:
    """
    Create plots of the parameter landscape.
    
    Args:
        landscape_data: Data from evaluate_parameter_landscape.
        save_path: Path to save the plot. If None, shows the plot.
    """
    wheel_radii = landscape_data['wheel_radii']
    motor_kts = landscape_data['motor_kts']
    battery_rhos = landscape_data['battery_rhos']
    losses = landscape_data['losses']
    center_params = landscape_data['center_params']
    landscape_best_params = landscape_data['landscape_best_params']
    
    n_samples = len(wheel_radii)
    
    # Reshape losses for 3D plotting
    losses_3d = losses.reshape((n_samples, n_samples, n_samples))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(
        landscape_data['param_combinations'][:, 0],
        landscape_data['param_combinations'][:, 1], 
        landscape_data['param_combinations'][:, 2],
        c=losses, cmap='viridis', alpha=0.6
    )
    ax1.scatter(center_params[0], center_params[1], center_params[2], 
               color='red', s=100, marker='*', label='Center Point')
    ax1.scatter(landscape_best_params[0], landscape_best_params[1], landscape_best_params[2],
               color='orange', s=100, marker='o', label='Landscape Best')
    ax1.set_xlabel('Wheel Radius (m)')
    ax1.set_ylabel('Motor Kt (Nm/A)')
    ax1.set_zlabel('Battery Rho')
    ax1.set_title('3D Parameter Landscape')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Loss')
    
    # Contour plots for each parameter pair
    # Fix battery_rho at center value
    rho_idx = jnp.argmin(jnp.abs(battery_rhos - center_params[2]))
    losses_wr_kt = losses_3d[:, :, rho_idx]
    
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.contourf(wheel_radii, motor_kts, losses_wr_kt.T, levels=20, cmap='viridis')
    ax2.scatter(center_params[0], center_params[1], color='red', s=100, marker='*', label='Center Point')
    ax2.scatter(landscape_best_params[0], landscape_best_params[1], color='orange', s=100, marker='o', label='Landscape Best')
    ax2.set_xlabel('Wheel Radius (m)')
    ax2.set_ylabel('Motor Kt (Nm/A)')
    ax2.set_title(f'Loss Landscape (Battery ρ = {battery_rhos[rho_idx]:.2f})')
    ax2.legend()
    plt.colorbar(contour, ax=ax2, label='Loss')
    
    # Fix motor_kt at center value
    kt_idx = jnp.argmin(jnp.abs(motor_kts - center_params[1]))
    losses_wr_rho = losses_3d[:, kt_idx, :]
    
    ax3 = fig.add_subplot(2, 2, 3)
    contour = ax3.contourf(wheel_radii, battery_rhos, losses_wr_rho.T, levels=20, cmap='viridis')
    ax3.scatter(center_params[0], center_params[2], color='red', s=100, marker='*', label='Center Point')
    ax3.scatter(landscape_best_params[0], landscape_best_params[2], color='orange', s=100, marker='o', label='Landscape Best')
    ax3.set_xlabel('Wheel Radius (m)')
    ax3.set_ylabel('Battery Rho')
    ax3.set_title(f'Loss Landscape (Motor Kt = {motor_kts[kt_idx]:.2f})')
    ax3.legend()
    plt.colorbar(contour, ax=ax3, label='Loss')
    
    # Fix wheel_radius at center value
    wr_idx = jnp.argmin(jnp.abs(wheel_radii - center_params[0]))
    losses_kt_rho = losses_3d[wr_idx, :, :]
    
    ax4 = fig.add_subplot(2, 2, 4)
    contour = ax4.contourf(motor_kts, battery_rhos, losses_kt_rho, levels=20, cmap='viridis')
    ax4.scatter(center_params[1], center_params[2], color='red', s=100, marker='*', label='Center Point')
    ax4.scatter(landscape_best_params[1], landscape_best_params[2], color='orange', s=100, marker='o', label='Landscape Best')
    ax4.set_xlabel('Motor Kt (Nm/A)')
    ax4.set_ylabel('Battery Rho')
    ax4.set_title(f'Loss Landscape (Wheel Radius = {wheel_radii[wr_idx]:.3f})')
    ax4.legend()
    plt.colorbar(contour, ax=ax4, label='Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter landscape plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter landscape for DMO"
    )
    parser.add_argument(
        "--center-params",
        type=float,
        nargs=3,
        default=[0.05, 1.0, 1.0],
        help="Center parameters [wheel_radius, motor_kt, battery_rho]",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per parameter dimension",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (default: show plot)",
    )
    parser.add_argument(
        "--param-ranges",
        type=float,
        nargs=6,
        default=[0.01, 0.15, 0.1, 2.0, 0.1, 2.0],
        help="Parameter ranges [wr_min, wr_max, kt_min, kt_max, rho_min, rho_max]",
    )
    
    args = parser.parse_args()
    
    # Parse parameter ranges
    param_ranges = {
        'wheel_radius': (args.param_ranges[0], args.param_ranges[1]),
        'motor_kt': (args.param_ranges[2], args.param_ranges[3]),
        'battery_rho': (args.param_ranges[4], args.param_ranges[5]),
    }
    
    center_params = jnp.array(args.center_params)
    
    # Evaluate landscape
    landscape_data = evaluate_parameter_landscape(
        center_params=center_params,
        n_samples=args.samples,
        param_ranges=param_ranges,
    )
    
    # Create plot
    plot_parameter_landscape(landscape_data, save_path=args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
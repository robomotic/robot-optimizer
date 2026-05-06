#!/usr/bin/env python3
"""
Evaluation script for trained DMO parameters.

Loads trained design parameters and runs a full trajectory
for analysis and visualization.
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mujoco_compat import mjx

from environment import load_robot_model
from motor_model import MotorModel
from control_policy import WallFollowerPolicy
from simulation import DifferentiableSimulation, RolloutConfig
from metrics_logging import TrajectoryLogger
from video_recorder import record_video


def evaluate_parameter_landscape(
    best_params: jnp.ndarray,
    n_samples: int = 10,
    param_ranges: dict = None,
) -> dict:
    """
    Evaluate the parameter landscape around the best parameters.
    
    Args:
        best_params: Best parameters found [wheel_radius, motor_kt, battery_rho].
        n_samples: Number of samples per parameter dimension.
        param_ranges: Dictionary with parameter ranges. If None, uses defaults.
    
    Returns:
        Dictionary with landscape data for plotting.
    """
    if param_ranges is None:
        param_ranges = {
            'wheel_radius': (0.01, 0.15),  # m
            'motor_kt': (0.1, 2.0),        # Nm/A
            'battery_rho': (0.1, 2.0),     # dimensionless
        }
    
    print(f"\n{'='*70}")
    print(f"Evaluating Parameter Landscape")
    print(f"{'='*70}")
    print(f"Best parameters: wheel_radius={best_params[0]:.4f}, kt={best_params[1]:.4f}, rho={best_params[2]:.4f}")
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
                
                if (i * n_samples * n_samples + j * n_samples + k) % 100 == 0:
                    print(f"  Evaluated {i * n_samples * n_samples + j * n_samples + k + 1}/{n_samples**3} combinations")
    
    losses = jnp.array(losses)
    param_combinations = jnp.array(param_combinations)
    
    # Find best in landscape (should be close to best_params)
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
        'best_params': best_params,
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
    best_params = landscape_data['best_params']
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
    ax1.scatter(best_params[0], best_params[1], best_params[2], 
               color='red', s=100, marker='*', label='Optimization Best')
    ax1.scatter(landscape_best_params[0], landscape_best_params[1], landscape_best_params[2],
               color='orange', s=100, marker='o', label='Landscape Best')
    ax1.set_xlabel('Wheel Radius (m)')
    ax1.set_ylabel('Motor Kt (Nm/A)')
    ax1.set_zlabel('Battery Rho')
    ax1.set_title('3D Parameter Landscape')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Loss')
    
    # Contour plots for each parameter pair
    # Fix battery_rho at best value
    rho_idx = jnp.argmin(jnp.abs(battery_rhos - best_params[2]))
    losses_wr_kt = losses_3d[:, :, rho_idx]
    
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.contourf(wheel_radii, motor_kts, losses_wr_kt.T, levels=20, cmap='viridis')
    ax2.scatter(best_params[0], best_params[1], color='red', s=100, marker='*', label='Optimization Best')
    ax2.scatter(landscape_best_params[0], landscape_best_params[1], color='orange', s=100, marker='o', label='Landscape Best')
    ax2.set_xlabel('Wheel Radius (m)')
    ax2.set_ylabel('Motor Kt (Nm/A)')
    ax2.set_title(f'Loss Landscape (Battery ρ = {battery_rhos[rho_idx]:.2f})')
    ax2.legend()
    plt.colorbar(contour, ax=ax2, label='Loss')
    
    # Fix motor_kt at best value
    kt_idx = jnp.argmin(jnp.abs(motor_kts - best_params[1]))
    losses_wr_rho = losses_3d[:, kt_idx, :]
    
    ax3 = fig.add_subplot(2, 2, 3)
    contour = ax3.contourf(wheel_radii, battery_rhos, losses_wr_rho.T, levels=20, cmap='viridis')
    ax3.scatter(best_params[0], best_params[2], color='red', s=100, marker='*', label='Optimization Best')
    ax3.scatter(landscape_best_params[0], landscape_best_params[2], color='orange', s=100, marker='o', label='Landscape Best')
    ax3.set_xlabel('Wheel Radius (m)')
    ax3.set_ylabel('Battery Rho')
    ax3.set_title(f'Loss Landscape (Motor Kt = {motor_kts[kt_idx]:.2f})')
    ax3.legend()
    plt.colorbar(contour, ax=ax3, label='Loss')
    
    # Fix wheel_radius at best value
    wr_idx = jnp.argmin(jnp.abs(wheel_radii - best_params[0]))
    losses_kt_rho = losses_3d[wr_idx, :, :]
    
    ax4 = fig.add_subplot(2, 2, 4)
    contour = ax4.contourf(motor_kts, battery_rhos, losses_kt_rho, levels=20, cmap='viridis')
    ax4.scatter(best_params[1], best_params[2], color='red', s=100, marker='*', label='Optimization Best')
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


def evaluate(
    checkpoint_path: Path,
    n_steps: int = 500,
    landscape_samples: int = 8,
    video_path: Path = None,
) -> dict:
    """
    Evaluate trained parameters on a full trajectory.
    
    Args:
        checkpoint_path: Path to saved parameters (*.npy).
        n_steps: Number of simulation steps.
    
    Returns:
        Dictionary with trajectory data and metrics.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating trained parameters")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # Load parameters
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None
    
    params = jnp.array(np.load(checkpoint_path))
    print(f"Loaded parameters: wheel_radius={params[0]:.4f}, kt={params[1]:.4f}, rho={params[2]:.4f}")
    
    # Load model
    try:
        base_model = load_robot_model()
        print(f"✓ Loaded robot model")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Convert to MJX
    try:
        mjx_model = mjx.Model.from_mujoco(base_model)
        print(f"✓ Converted to MJX\n")
    except Exception as e:
        print(f"✗ Failed to convert: {e}")
        return None
    
    # Create simulation
    config = RolloutConfig(n_steps=n_steps)
    motor_model = MotorModel()
    control_policy = WallFollowerPolicy()
    
    simulation = DifferentiableSimulation(
        motor_model=motor_model,
        control_policy=control_policy,
        config=config,
    )
    
    # Run trajectory
    print("Running trajectory...")
    try:
        trajectory = simulation.rollout_with_visualization(params)
    except Exception as e:
        print(f"✗ Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Extract metrics
    positions = trajectory["positions"]
    distances = trajectory["distances"]
    final_loss = trajectory["total_loss"]
    
    # Compute statistics
    final_position = positions[-1]
    final_distance = distances[-1]
    min_distance = jnp.min(distances)
    
    # Create trajectory visualization
    try:
        traj_logger = TrajectoryLogger(checkpoint_path.parent / "evaluation_trajectories")
        plot_path = checkpoint_path.parent / "evaluation_trajectory.png"
        traj_logger.create_trajectory_plot(
            trajectory,
            title=f"Evaluation Trajectory - {checkpoint_path.stem}",
            save_path=plot_path
        )
        print(f"✓ Saved trajectory plot to {plot_path}")
    except Exception as e:
        print(f"Warning: Failed to create trajectory plot: {e}")
    
    # Record video if requested
    if video_path is not None:
        try:
            record_video(
                runs=[{
                    "label": checkpoint_path.stem,
                    "params": params,
                    "trajectory": trajectory,
                }],
                output_path=video_path,
            )
        except Exception as e:
            print(f"Warning: Failed to record video: {e}")

    # Evaluate parameter landscape
    if landscape_samples > 0:
        try:
            landscape_data = evaluate_parameter_landscape(params, n_samples=landscape_samples)
            landscape_plot_path = checkpoint_path.parent / "parameter_landscape.png"
            plot_parameter_landscape(landscape_data, save_path=landscape_plot_path)
        except Exception as e:
            print(f"Warning: Failed to evaluate parameter landscape: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Parameter landscape evaluation disabled (--landscape-samples=0)")
    
    print(f"\n{'='*70}")
    print(f"Results")
    print(f"{'='*70}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final position: [{final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f}]")
    print(f"Final distance to goal: {final_distance:.4f} m")
    print(f"Minimum distance reached: {min_distance:.4f} m")
    
    # Simple trajectory visualization (text)
    print(f"\nTrajectory (every 50 steps):")
    for i in range(0, len(distances), 50):
        pos = positions[i]
        dist = distances[i]
        print(f"  Step {i:3d}: pos=[{pos[0]:6.3f}, {pos[1]:6.3f}], dist={dist:6.3f} m")
    
    print(f"{'='*70}\n")
    
    return {
        "params": params,
        "positions": positions,
        "distances": distances,
        "final_loss": final_loss,
        "final_distance": final_distance,
        "min_distance": min_distance,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DMO parameters"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint (*.npy file)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Simulation steps",
    )
    parser.add_argument(
        "--landscape-samples",
        type=int,
        default=8,
        help="Number of samples per parameter for landscape evaluation (0 to disable)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        metavar="PATH",
        help="Record a video of the evaluation trajectory to this file (.gif or .mp4)",
    )

    args = parser.parse_args()

    results = evaluate(
        checkpoint_path=args.checkpoint,
        n_steps=args.steps,
        landscape_samples=args.landscape_samples,
        video_path=args.video,
    )
    
    if results is None:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

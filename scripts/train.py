#!/usr/bin/env python3
"""
Training script for Differentiable Morphological Optimizer (DMO).

Runs end-to-end co-design optimization using gradient descent (Adam optimizer).
Minimizes a composite loss function (distance to goal, energy, collisions)
over design parameters (wheel radius, motor constant, battery factor).
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Disable JAX GPU memory preallocation so we can share the GPU with other processes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import optax
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motor_model import MotorModel
from control_policy import WallFollowerPolicy
from simulation import DifferentiableSimulation, RolloutConfig
from metrics_logging import MetricsLogger, TrajectoryLogger


def create_output_dir() -> Path:
    """Create output directory for checkpoints and logs."""
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def initialize_design_params(
    wheel_radius_init: float = 0.05,
    motor_kt_init: float = 1.0,
    battery_rho_init: float = 1.0,
) -> jnp.ndarray:
    """
    Initialize design parameters.
    
    Args:
        wheel_radius_init: Initial wheel radius (m).
        motor_kt_init: Initial motor torque constant (Nm/A).
        battery_rho_init: Initial battery mass factor (dimensionless).
    
    Returns:
        Design parameter array [wheel_radius, motor_kt, battery_rho].
    """
    return jnp.array([wheel_radius_init, motor_kt_init, battery_rho_init])


def train(
    n_iterations: int = 100,
    learning_rate: float = 1e-3,
    n_steps: int = 500,
    lambda_energy: float = 0.01,
    lambda_collision: float = 10.0,
    output_dir: Path = None,
) -> dict:
    """
    Run optimization loop.
    
    Args:
        base_model: Base MJX model.
        n_iterations: Number of optimization iterations.
        learning_rate: Adam learning rate.
        n_steps: Simulation steps per rollout.
        lambda_energy: Energy loss weight.
        lambda_collision: Collision loss weight.
        output_dir: Directory for checkpoints and logs.
    
    Returns:
        Dictionary with final params and training history.
    """
    if output_dir is None:
        output_dir = create_output_dir()
    
    print(f"\n{'='*70}")
    print(f"Differentiable Morphological Optimizer - Training")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {n_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Simulation steps: {n_steps}")
    print(f"Loss weights - Energy: {lambda_energy}, Collision: {lambda_collision}")
    print(f"{'='*70}\n")
    
    # Create simulation
    config = RolloutConfig(
        n_steps=n_steps,
        lambda_energy=lambda_energy,
        lambda_collision=lambda_collision,
    )
    
    motor_model = MotorModel()
    control_policy = WallFollowerPolicy()
    
    simulation = DifferentiableSimulation(
        motor_model=motor_model,
        control_policy=control_policy,
        config=config,
    )
    
    # Initialize parameters
    params = initialize_design_params()
    print(f"Initial parameters: wheel_radius={params[0]:.4f}, kt={params[1]:.4f}, rho={params[2]:.4f}")
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Metrics logger
    logger = MetricsLogger(output_dir / "training_log.csv")
    
    # Trajectory logger
    traj_logger = TrajectoryLogger(output_dir / "trajectories")
    
    # Optimization loop
    best_loss = jnp.inf
    best_params = params
    
    try:
        for iteration in range(n_iterations):
            # Compute gradients
            loss_fn = simulation.objective
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            # Optimizer update
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Clip each parameter to its own physical bounds
            params = params.at[0].set(jnp.clip(params[0], 0.01, 0.15))   # wheel_radius (m)
            params = params.at[1].set(jnp.clip(params[1], 0.1,  5.0))    # motor_kt
            params = params.at[2].set(jnp.clip(params[2], 0.1,  5.0))    # battery_rho
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            # Log metrics
            grad_norm = jnp.linalg.norm(grads)
            metrics = {
                "iteration": iteration + 1,
                "loss": float(loss),
                "grad_norm": float(grad_norm),
                "wheel_radius": float(params[0]),
                "motor_kt": float(params[1]),
                "battery_rho": float(params[2]),
                "best_loss": float(best_loss),
            }
            logger.log(metrics)
            
            # Log trajectory every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration == 0:
                try:
                    trajectory_data = simulation.rollout_with_visualization(params)
                    traj_logger.log_trajectory(trajectory_data, iteration + 1, np.array(params))
                    
                    # Create trajectory plot
                    plot_path = output_dir / "trajectories" / f"trajectory_iter_{iteration+1:04d}.png"
                    traj_logger.create_trajectory_plot(
                        trajectory_data,
                        title=f"Trajectory - Iteration {iteration+1}",
                        save_path=plot_path
                    )
                except Exception as e:
                    print(f"Warning: Failed to log trajectory for iteration {iteration+1}: {e}")
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(
                    f"Iter {iteration+1:4d} | "
                    f"Loss: {float(loss):10.6f} | "
                    f"GradNorm: {grad_norm:10.6f} | "
                    f"Params: [{params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f}]"
                )
    
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    
    # Save results
    print(f"\n{'='*70}")
    print(f"Training Complete")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best parameters: wheel_radius={best_params[0]:.4f}, kt={best_params[1]:.4f}, rho={best_params[2]:.4f}")
    
    # Save checkpoints
    checkpoint_path = output_dir / "best_params.npy"
    np.save(checkpoint_path, np.array(best_params))
    print(f"Saved best parameters to {checkpoint_path}")
    
    final_path = output_dir / "final_params.npy"
    np.save(final_path, np.array(params))
    print(f"Saved final parameters to {final_path}")
    
    # Save final trajectory
    try:
        final_trajectory = simulation.rollout_with_visualization(best_params)
        final_plot_path = output_dir / "final_trajectory.png"
        traj_logger.create_trajectory_plot(
            final_trajectory,
            title="Final Optimized Trajectory",
            save_path=final_plot_path
        )
        
        # Save all trajectories
        traj_logger.save_all_trajectories()
        print(f"Saved trajectory data to {output_dir / 'trajectories'}")
    except Exception as e:
        print(f"Warning: Failed to save final trajectory: {e}")
    
    # Close loggers
    logger.close()
    print(f"Saved training log to {output_dir / 'training_log.csv'}")
    print(f"{'='*70}\n")
    
    return {
        "best_params": best_params,
        "final_params": params,
        "best_loss": best_loss,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Differentiable Morphological Optimizer"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="MJX simulation steps per rollout",
    )
    parser.add_argument(
        "--lambda-energy",
        type=float,
        default=0.01,
        help="Weight for energy loss",
    )
    parser.add_argument(
        "--lambda-collision",
        type=float,
        default=10.0,
        help="Weight for collision loss",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (auto-created if not specified)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        args.output_dir = create_output_dir()
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        results = train(
            n_iterations=args.iterations,
            learning_rate=args.learning_rate,
            n_steps=args.steps,
            lambda_energy=args.lambda_energy,
            lambda_collision=args.lambda_collision,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

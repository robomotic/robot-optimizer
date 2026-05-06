"""
Logging utilities for training metrics.

Tracks optimization progress (loss, gradients, parameters) and saves to CSV.
Also provides trajectory logging and visualization for rollout analysis.
"""

import csv
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np


class MetricsLogger:
    """
    Simple CSV logger for training metrics.
    
    Writes metrics to a CSV file for later analysis and visualization.
    """
    
    def __init__(self, log_path: Path):
        """
        Initialize logger.
        
        Args:
            log_path: Path to CSV file to write metrics to.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = None
        self.csv_writer = None
        self.fieldnames = None
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a metrics dictionary to CSV.
        
        Args:
            metrics: Dictionary of metric name -> value pairs.
        """
        # Initialize CSV writer on first call
        if self.csv_writer is None:
            self.fieldnames = sorted(metrics.keys())
            self.csv_file = open(self.log_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
        
        # Write row
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
    
    def close(self) -> None:
        """Close the CSV file."""
        if self.csv_file is not None:
            self.csv_file.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class TrajectoryLogger:
    """
    Logger for robot trajectory data during rollouts.
    
    Saves position, distance, and other trajectory data for visualization.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize trajectory logger.
        
        Args:
            output_dir: Directory to save trajectory data.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories = []
    
    def log_trajectory(
        self,
        trajectory_data: Dict[str, Any],
        iteration: int,
        design_params: Optional[np.ndarray] = None,
    ) -> None:
        """
        Log trajectory data from a rollout.
        
        Args:
            trajectory_data: Dictionary with trajectory data (positions, distances, etc.)
            iteration: Training iteration number
            design_params: Current design parameters [wheel_radius, motor_kt, battery_rho]
        """
        trajectory = {
            "iteration": iteration,
            "positions": trajectory_data.get("positions", []),
            "distances": trajectory_data.get("distances", []),
            "total_loss": trajectory_data.get("total_loss", 0.0),
            "design_params": design_params.tolist() if design_params is not None else None,
        }
        self.trajectories.append(trajectory)
        
        # Save individual trajectory
        traj_path = self.output_dir / f"trajectory_iter_{iteration:04d}.npy"
        np.save(traj_path, trajectory)
    
    def save_all_trajectories(self) -> None:
        """Save all logged trajectories to a single file."""
        all_traj_path = self.output_dir / "all_trajectories.npy"
        np.save(all_traj_path, self.trajectories)
    
    def create_trajectory_plot(
        self,
        trajectory_data: Dict[str, Any],
        title: str = "Robot Trajectory",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create a simple trajectory plot.
        
        Args:
            trajectory_data: Trajectory data dictionary
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for trajectory plotting")
            return
        
        positions = np.array(trajectory_data.get("positions", []))
        distances = np.array(trajectory_data.get("distances", []))
        
        if len(positions) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Trajectory plot
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)
        
        # Goal position (northeast corner)
        ax1.scatter(0.8, 0.8, c='orange', s=100, marker='*', label='Goal', zorder=5)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Robot Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Distance to goal over time
        ax2.plot(distances, 'g-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Distance to Goal (m)')
        ax2.set_title('Distance to Goal vs Time')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trajectory plot to {save_path}")
        
        plt.close(fig)


def create_trajectory_animation(
    trajectories: List[Dict[str, Any]],
    output_path: Path,
    fps: int = 10,
) -> None:
    """
    Create an animation showing trajectory evolution over training iterations.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save the animation
        fps: Frames per second for the animation
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Warning: matplotlib not available for trajectory animation")
        return
    
    if not trajectories:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def animate(frame):
        ax.clear()
        
        traj = trajectories[frame]
        positions = np.array(traj.get("positions", []))
        iteration = traj.get("iteration", frame)
        design_params = traj.get("design_params", [0.05, 1.0, 1.0])
        
        if len(positions) > 0:
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
            ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)
            
            # Goal position
            ax.scatter(0.8, 0.8, c='orange', s=100, marker='*', label='Goal', zorder=5)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        params_str = (
            f"r={design_params[0]:.3f}  kt={design_params[1]:.2f}  ρ={design_params[2]:.2f}"
            if design_params else ""
        )
        ax.set_title(f"Iteration {iteration}  {params_str}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(trajectories),
        interval=1000/fps, repeat=True
    )
    
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved trajectory animation to {output_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    
    plt.close(fig)
    """
    Simple CSV logger for training metrics.
    
    Writes metrics to a CSV file for later analysis and visualization.
    """
    
    def __init__(self, log_path: Path):
        """
        Initialize logger.
        
        Args:
            log_path: Path to CSV file to write metrics to.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = None
        self.csv_writer = None
        self.fieldnames = None
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a metrics dictionary to CSV.
        
        Args:
            metrics: Dictionary of metric name -> value pairs.
        """
        # Initialize CSV writer on first call
        if self.csv_writer is None:
            self.fieldnames = sorted(metrics.keys())
            self.csv_file = open(self.log_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
        
        # Write row
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
    
    def close(self) -> None:
        """Close the CSV file."""
        if self.csv_file is not None:
            self.csv_file.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

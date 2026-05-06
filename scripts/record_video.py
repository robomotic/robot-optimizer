#!/usr/bin/env python3
"""
Record a video of the robot moving through the maze.

Can compare multiple parameter sets side-by-side. Accepts parameters either
from saved checkpoints (*.npy) or as inline values on the command line.

Examples
--------
# Record a single checkpoint:
python scripts/record_video.py --checkpoint outputs/run_xxx/best_params.npy

# Compare two checkpoints:
python scripts/record_video.py \\
    --checkpoint outputs/run_A/best_params.npy \\
    --checkpoint outputs/run_B/best_params.npy

# Mix checkpoints with inline parameter sets:
python scripts/record_video.py \\
    --checkpoint outputs/run_xxx/best_params.npy \\
    --params "0.05 1.0 1.0" "0.10 2.0 0.5"

# Custom output path and format (.gif or .mp4):
python scripts/record_video.py \\
    --checkpoint outputs/run_xxx/best_params.npy \\
    --output comparison.gif --fps 15
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motor_model import MotorModel
from control_policy import WallFollowerPolicy
from simulation import DifferentiableSimulation, RolloutConfig
from video_recorder import record_video, record_video_3d


def _load_params(path: Path) -> np.ndarray:
    params = np.load(path)
    if params.shape != (3,):
        raise ValueError(f"{path}: expected shape (3,), got {params.shape}")
    return params


def _parse_inline_params(s: str) -> np.ndarray:
    """Parse '0.05 1.0 1.0' → np.array([0.05, 1.0, 1.0])."""
    values = [float(v) for v in s.split()]
    if len(values) != 3:
        raise ValueError(
            f"--params expects 3 space-separated floats "
            f"(wheel_radius motor_kt battery_rho), got: {s!r}"
        )
    return np.array(values)


def _run_simulation(
    params: np.ndarray,
    n_steps: int,
) -> dict:
    config = RolloutConfig(n_steps=n_steps)
    sim = DifferentiableSimulation(
        motor_model=MotorModel(),
        control_policy=WallFollowerPolicy(),
        config=config,
    )
    traj = sim.rollout_with_visualization(jnp.array(params))
    # Attach goal so the renderer always uses the simulation's goal, not a hardcoded default
    traj["goal_pos"] = tuple(float(v) for v in config.goal_pos[:2])
    return traj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a video of the robot trajectory for one or more parameter sets"
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        action="append",
        type=Path,
        default=[],
        help="Path to a saved *.npy parameter file. Repeatable.",
    )
    parser.add_argument(
        "--params",
        metavar='"r kt rho"',
        action="append",
        default=[],
        help='Inline parameters as quoted string, e.g. "0.05 1.0 1.0". Repeatable.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robot_video.gif"),
        help="Output video path. Use .gif (default) or .mp4.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Simulation steps per run (default: 500)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frames per second (default: 20)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Simulation steps per video frame — higher = faster video (default: 2)",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=40,
        help="Number of past positions shown as trail (default: 40)",
    )
    parser.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        default=False,
        help="Use MuJoCo 3D renderer instead of 2D matplotlib animation",
    )
    parser.add_argument(
        "--camera",
        dest="camera_preset",
        choices=["iso", "top"],
        default="iso",
        help="3D camera preset: iso (isometric, default) or top (vertical top-down)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of each panel in 3D mode (default: 640)",
    )
    parser.add_argument(
        "--height3d",
        type=int,
        default=400,
        help="Height of each panel in 3D mode (default: 400)",
    )

    args = parser.parse_args()

    if not args.checkpoint and not args.params:
        print("No parameters specified. Using defaults: wheel_radius=0.05, kt=1.0, rho=1.0")
        args.params = ["0.05 1.0 1.0"]

    # Build the list of runs
    runs = []

    for ckpt_path in args.checkpoint:
        if not ckpt_path.exists():
            print(f"Error: checkpoint not found: {ckpt_path}")
            return 1
        params = _load_params(ckpt_path)
        label = ckpt_path.stem  # e.g. "best_params"
        runs.append({"label": label, "params": params})

    for raw in args.params:
        try:
            params = _parse_inline_params(raw)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        label = f"r={params[0]:.3f} kₜ={params[1]:.2f} ρ={params[2]:.2f}"
        runs.append({"label": label, "params": params})

    # Run simulations
    print(f"\nRunning {len(runs)} simulation(s) with {args.steps} steps each…")
    for i, run in enumerate(runs):
        p = run["params"]
        print(
            f"  [{i+1}/{len(runs)}] "
            f"wheel_radius={p[0]:.4f}  motor_kt={p[1]:.4f}  battery_rho={p[2]:.4f}"
        )
        try:
            run["trajectory"] = _run_simulation(run["params"], args.steps)
        except Exception as e:
            print(f"  Error: simulation failed — {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Record video
    print(f"\nRendering {'3D' if args.use_3d else '2D'} video → {args.output}")
    try:
        if args.use_3d:
            record_video_3d(
                runs=runs,
                output_path=args.output,
                fps=args.fps,
                trail_length=args.trail,
                step_stride=args.stride,
                width=args.width,
                height=args.height3d,
                camera_preset=args.camera_preset,
            )
        else:
            # All runs share the same RolloutConfig goal — take it from the first run
            goal_pos = runs[0]["trajectory"].get("goal_pos", (0.8, 0.2))
            record_video(
                runs=runs,
                output_path=args.output,
                fps=args.fps,
                trail_length=args.trail,
                step_stride=args.stride,
                goal_pos=goal_pos,
            )
    except Exception as e:
        print(f"Error: video rendering failed — {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

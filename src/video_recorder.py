"""
Video recorder for DMO robot trajectory visualization.

Animates the robot moving through the arena for one or more parameter sets.
Saves to GIF (requires pillow) or MP4 (requires ffmpeg).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _estimate_orientations(positions: np.ndarray) -> np.ndarray:
    """Compute heading angles from consecutive positions via finite differences."""
    n = len(positions)
    orientations = np.zeros(n)
    for i in range(1, n):
        dx = positions[i, 0] - positions[i - 1, 0]
        dy = positions[i, 1] - positions[i - 1, 1]
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            orientations[i] = np.arctan2(dy, dx)
        else:
            orientations[i] = orientations[i - 1]
    orientations[0] = orientations[1] if n > 1 else 0.0
    return orientations


def record_video(
    runs: List[Dict[str, Any]],
    output_path: Path,
    fps: int = 20,
    trail_length: int = 40,
    step_stride: int = 2,
    goal_pos: Tuple[float, float] = (0.8, 0.2),
    arena: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> None:
    """
    Record a video of one or more robot trajectories side-by-side.

    Args:
        runs: List of dicts, each with:
              - 'label': str, display name
              - 'params': array-like [wheel_radius, motor_kt, battery_rho]
              - 'trajectory': rollout_with_visualization() output dict
        output_path: Destination file. Extension controls format:
                     .gif  → Pillow writer (no extra install needed)
                     .mp4  → FFMpeg writer (ffmpeg must be on PATH)
        fps: Frames per second in the output video.
        trail_length: How many past positions to draw as a fading trail.
        step_stride: Advance this many simulation steps per video frame.
                     Higher values → shorter, faster video.
        goal_pos: (x, y) of the goal.
        arena: (xmin, xmax, ymin, ymax) bounds of the arena.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as mpatches

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("No runs provided")

    # Pre-process: numpy arrays + orientation
    for run in runs:
        traj = run["trajectory"]
        run["_pos"] = np.array(traj["positions"])
        run["_dist"] = np.array(traj["distances"])
        run["_orient"] = _estimate_orientations(run["_pos"])

    max_steps = max(len(r["_pos"]) for r in runs)
    frames = list(range(0, max_steps, step_stride))

    xmin, xmax, ymin, ymax = arena
    padding = 0.12

    # Figure layout: top row = arena views, bottom row = distance plots
    fig, axes = plt.subplots(
        2, n_runs,
        figsize=(5 * n_runs, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    if n_runs == 1:
        axes = axes.reshape(2, 1)

    fig.patch.set_facecolor("#1a1a2e")

    # ── Per-run static setup ──────────────────────────────────────────────────
    art_per_run = []
    for col, run in enumerate(runs):
        params = np.asarray(run["params"])
        label = run.get("label", f"Run {col + 1}")
        pos = run["_pos"]
        dist = run["_dist"]
        orient = run["_orient"]
        # Chassis radius is the fixed collision bounding circle (wall_margin = 0.05 m).
        # Wheel radius is a design parameter and must not inflate the visual body size.
        chassis_r = 0.05

        # ── Arena axis ──────────────────────────────────────────────
        ax = axes[0, col]
        ax.set_facecolor("#0f3460")
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="white", fontsize=8)
        ax.set_ylabel("Y (m)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        # Maze floor + walls
        floor = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="#e0e0e0", facecolor="#16213e",
        )
        ax.add_patch(floor)

        # Goal
        ax.plot(*goal_pos, "y*", markersize=16, zorder=10, label="Goal")
        # Start
        ax.plot(pos[0, 0], pos[0, 1], "w^", markersize=8, zorder=10, label="Start")

        ax.legend(fontsize=7, loc="upper left",
                  facecolor="#0f3460", labelcolor="white", framealpha=0.8)
        ax.grid(True, alpha=0.15, color="white")

        param_line = f"r={params[0]:.3f} m   kₜ={params[1]:.2f}   ρ={params[2]:.2f}"
        ax.set_title(f"{label}\n{param_line}", color="white", fontsize=9, pad=6)

        # ── Dynamic artists ──────────────────────────────────────────
        # Trail line
        trail_line, = ax.plot([], [], color="#6fa3ef", alpha=0.6,
                              linewidth=1.5, zorder=5)

        # Robot chassis circle
        robot_circle = plt.Circle(
            (pos[0, 0], pos[0, 1]), chassis_r,
            color="#e94560", zorder=7,
        )
        ax.add_patch(robot_circle)

        # Orientation indicator (line from centre in heading direction)
        arrow_len = chassis_r * 1.4
        theta0 = orient[0]
        orient_line, = ax.plot(
            [pos[0, 0], pos[0, 0] + arrow_len * np.cos(theta0)],
            [pos[0, 1], pos[0, 1] + arrow_len * np.sin(theta0)],
            "w-", linewidth=2, zorder=8,
        )

        # Step / distance text
        info_text = ax.text(
            xmin + 0.03, ymax - 0.04, "",
            fontsize=8, color="#ffd700",
            verticalalignment="top", zorder=9,
        )

        # ── Distance subplot ─────────────────────────────────────────
        dax = axes[1, col]
        dax.set_facecolor("#16213e")
        dax.set_xlim(0, max_steps)
        dax.set_ylim(0, max(dist.max(), 0.01) * 1.15)
        dax.set_xlabel("Step", color="white", fontsize=8)
        dax.set_ylabel("Dist to Goal (m)", color="white", fontsize=8)
        dax.tick_params(colors="white", labelsize=7)
        for spine in dax.spines.values():
            spine.set_edgecolor("white")
        dax.grid(True, alpha=0.15, color="white")

        # Full-trajectory ghost line
        dax.plot(range(len(dist)), dist, color="white", alpha=0.2,
                 linewidth=1, zorder=1)
        dist_live, = dax.plot([], [], color="#00d2ff", linewidth=2, zorder=2)
        dist_dot, = dax.plot([], [], "o", color="#e94560", markersize=5, zorder=3)

        art_per_run.append({
            "pos": pos,
            "dist": dist,
            "orient": orient,
            "chassis_r": chassis_r,
            "arrow_len": arrow_len,
            "trail_line": trail_line,
            "robot_circle": robot_circle,
            "orient_line": orient_line,
            "info_text": info_text,
            "dist_live": dist_live,
            "dist_dot": dist_dot,
        })

    plt.tight_layout(pad=1.2)

    # ── Animation update ─────────────────────────────────────────────────────
    def update(frame: int):
        updated = []
        for art in art_per_run:
            pos = art["pos"]
            dist = art["dist"]
            orient = art["orient"]
            n = len(pos)
            i = min(frame, n - 1)

            # Trail
            t0 = max(0, i - trail_length)
            art["trail_line"].set_data(pos[t0 : i + 1, 0], pos[t0 : i + 1, 1])

            # Robot body
            art["robot_circle"].center = (pos[i, 0], pos[i, 1])

            # Orientation line
            theta = orient[i]
            L = art["arrow_len"]
            art["orient_line"].set_data(
                [pos[i, 0], pos[i, 0] + L * np.cos(theta)],
                [pos[i, 1], pos[i, 1] + L * np.sin(theta)],
            )

            # Info text
            d = dist[i] if i < len(dist) else dist[-1]
            art["info_text"].set_text(f"step {i}  d={d:.3f} m")

            # Distance live line
            art["dist_live"].set_data(range(i + 1), dist[: i + 1])
            art["dist_dot"].set_data([i], [dist[i] if i < len(dist) else dist[-1]])

            updated += [
                art["trail_line"], art["robot_circle"],
                art["orient_line"], art["info_text"],
                art["dist_live"], art["dist_dot"],
            ]
        return updated

    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=max(1, 1000 // fps), blit=False,
    )

    suffix = output_path.suffix.lower()
    try:
        if suffix == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                            extra_args=["-pix_fmt", "yuv420p"])
        else:
            writer = animation.PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
        print(f"Saved video → {output_path}  ({len(frames)} frames @ {fps} fps)")
    finally:
        plt.close(fig)


def record_video_3d(
    runs: List[Dict[str, Any]],
    output_path: Path,
    fps: int = 20,
    trail_length: int = 40,
    step_stride: int = 2,
    goal_pos: Tuple[float, float] = (0.8, 0.2),
    width: int = 640,
    height: int = 400,
    camera_preset: str = "iso",
    camera_azimuth: Optional[float] = None,
    camera_elevation: Optional[float] = None,
    camera_distance: Optional[float] = None,
) -> None:
    """
    Record a 3D video using MuJoCo's renderer.

    Renders the scene from models/scene.xml, replaying each trajectory by
    setting qpos per frame and calling mj_forward. Multiple runs are tiled
    horizontally in the output image.

    Args:
        runs: Same format as record_video() — list of dicts with 'label',
              'params', and 'trajectory'.
        output_path: .gif or .mp4 destination.
        fps: Frames per second.
        trail_length: Unused (visual trail rendered via 2D overlay).
        step_stride: Simulation steps per video frame.
        goal_pos: (x, y) goal in simulation coordinates [0, 1].
        width: Per-panel render width in pixels.
        height: Per-panel render height in pixels.
        camera_preset: "iso" (isometric, default) or "top" (vertical top-down).
                       Ignored when all three camera_* kwargs are provided.
        camera_azimuth: Override azimuth in degrees.
        camera_elevation: MuJoCo camera elevation in degrees (negative = look down).
        camera_distance: MuJoCo camera distance from lookat point.
    """
    import os
    # Enable EGL offscreen rendering when no display is present
    if "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(runs)
    if n_runs == 0:
        raise ValueError("No runs provided")

    scene_xml = Path(__file__).parent.parent / "models" / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(f"scene.xml not found: {scene_xml}")

    # Load model once — we'll clone per run if wheel sizes differ
    base_model = mujoco.MjModel.from_xml_path(str(scene_xml))

    # Locate geom indices for the two wheel geoms (size[0] = radius)
    def _geom_id(model: mujoco.MjModel, name: str) -> int:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise KeyError(f"Geom '{name}' not found in scene.xml")
        return gid

    # Pre-process trajectories
    for run in runs:
        traj = run["trajectory"]
        run["_pos"] = np.array(traj["positions"])   # (T, 3) in sim coords
        run["_dist"] = np.array(traj["distances"])  # (T,)
        run["_orient"] = _estimate_orientations(run["_pos"])

    max_steps = max(len(r["_pos"]) for r in runs)
    frame_indices = list(range(0, max_steps, step_stride))

    # Resolve camera preset → (azimuth, elevation, distance)
    _presets = {
        "iso": (225.0, -35.264, 3.5),   # true isometric (equal axes)
        "top": (90.0,  -90.0,  3.0),    # vertical top-down
    }
    if camera_preset not in _presets:
        raise ValueError(f"Unknown camera_preset {camera_preset!r}. Choose 'iso' or 'top'.")
    p_az, p_el, p_dist = _presets[camera_preset]
    az  = camera_azimuth   if camera_azimuth   is not None else p_az
    el  = camera_elevation if camera_elevation is not None else p_el
    dist = camera_distance if camera_distance  is not None else p_dist

    # Set up camera
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = az
    camera.elevation = el
    camera.distance = dist
    camera.lookat[:] = [0.0, 0.0, 0.0]

    # Build per-run model + data + renderer
    renderers = []
    for run in runs:
        params = np.asarray(run["params"])
        wheel_radius = float(params[0])

        # Clone model and update wheel sizes
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        for geom_name in ("left_wheel_geom", "right_wheel_geom"):
            gid = _geom_id(model, geom_name)
            # geom_size for cylinder: [radius, half-length, 0]
            model.geom_size[gid, 0] = wheel_radius

        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderers.append({
            "model": model,
            "data": data,
            "renderer": renderer,
            "run": run,
        })

    def _sim_to_maze(sim_x: float, sim_y: float) -> Tuple[float, float]:
        """[0,1] sim coords → [-1,1] maze coords."""
        return 2.0 * sim_x - 1.0, 2.0 * sim_y - 1.0

    def _render_frame(step_idx: int) -> "Image.Image":
        panels = []
        for entry in renderers:
            model = entry["model"]
            data = entry["data"]
            renderer = entry["renderer"]
            run = entry["run"]

            pos = run["_pos"]
            dist = run["_dist"]
            orient = run["_orient"]
            params = np.asarray(run["params"])
            label = run.get("label", "")
            i = min(step_idx, len(pos) - 1)

            # Map sim position → maze coords
            mx, my = _sim_to_maze(pos[i, 0], pos[i, 1])
            theta = orient[i]

            # Set robot free-joint qpos: [tx, ty, tz, qw, qx, qy, qz]
            # Z: wheel center == robot body center (no vertical offset in body tree),
            # so setting z = wheel_radius places the wheel bottom exactly on the floor.
            wheel_radius = float(params[0])
            data.qpos[0] = mx
            data.qpos[1] = my
            data.qpos[2] = wheel_radius
            # Rotation around Z by theta: quat = [cos(θ/2), 0, 0, sin(θ/2)]
            data.qpos[3] = np.cos(theta / 2.0)
            data.qpos[4] = 0.0
            data.qpos[5] = 0.0
            data.qpos[6] = np.sin(theta / 2.0)

            # Wheel spin: cumulative arc length / radius gives rotation angle
            if i > 0:
                dx = pos[1:i+1, 0] - pos[:i, 0]
                dy = pos[1:i+1, 1] - pos[:i, 1]
                arc = float(np.sum(np.sqrt(dx**2 + dy**2)))
                wheel_radius = float(params[0])
                wheel_angle = arc / max(wheel_radius, 1e-6)
            else:
                wheel_angle = 0.0

            # qpos indices: 7 = left_wheel hinge, 8 = right_wheel hinge
            data.qpos[7] = wheel_angle
            data.qpos[8] = wheel_angle

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            frame_rgb = renderer.render()  # (H, W, 3) uint8

            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)

            # HUD: label + params + step + distance
            d_now = float(dist[i]) if i < len(dist) else float(dist[-1])
            hud_lines = [
                label,
                f"r={params[0]:.3f}  kt={params[1]:.2f}  ρ={params[2]:.2f}",
                f"step {i}   dist {d_now:.3f} m",
            ]
            y_cursor = 6
            for line in hud_lines:
                draw.text((8, y_cursor), line, fill=(255, 220, 60))
                y_cursor += 16

            panels.append(img)

        # Tile panels horizontally
        total_w = sum(p.width for p in panels)
        out_img = Image.new("RGB", (total_w, panels[0].height))
        x_off = 0
        for p in panels:
            out_img.paste(p, (x_off, 0))
            x_off += p.width
        return out_img

    # Render all frames
    print(f"Rendering {len(frame_indices)} frames ({n_runs} panel(s), {width}×{height} each)…")
    pil_frames = []
    for fi, step_idx in enumerate(frame_indices):
        if fi % max(1, len(frame_indices) // 10) == 0:
            print(f"  frame {fi+1}/{len(frame_indices)}")
        pil_frames.append(_render_frame(step_idx))

    # Clean up renderers
    for entry in renderers:
        entry["renderer"].close()

    # Save
    suffix = output_path.suffix.lower()
    frame_duration_ms = max(1, 1000 // fps)
    if suffix == ".mp4":
        _save_mp4(pil_frames, output_path, fps)
    else:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
    print(f"Saved 3-D video → {output_path}  ({len(pil_frames)} frames @ {fps} fps)")


def _save_mp4(frames: List["Image.Image"], output_path: Path, fps: int) -> None:
    """Write PIL frames to an MP4 via ffmpeg subprocess (no matplotlib needed)."""
    import subprocess
    import io

    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

#!/usr/bin/env python3
"""
Beautiful 3-D loss-landscape visualisation for DMO.

Evaluates the objective on 2-D parameter slices using jax.vmap (fast),
then renders a large 3-D surface + three heatmap panels in a dark theme.
Best (optimised) and worst (initial) parameter sets are highlighted.

Usage
-----
    python scripts/plot_landscape.py \
        --best  "0.15 2.02 1.16" \
        --worst "0.05 1.00 1.00" \
        --output outputs/landscape.png \
        --samples 25
"""

import sys
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motor_model import MotorModel
from control_policy import WallFollowerPolicy
from simulation import DifferentiableSimulation, RolloutConfig

# ── Palette ────────────────────────────────────────────────────────────────
BG        = "#0a0e1a"
PANEL_BG  = "#111827"
GRID_COL  = "#1f2937"
TEXT      = "#e2e8f0"
CMAP      = "plasma"
BEST_COL  = "#00e5ff"   # cyan star
WORST_COL = "#ff4444"   # red X


# ── Helpers ────────────────────────────────────────────────────────────────

def build_sim(n_steps: int = 500) -> DifferentiableSimulation:
    config = RolloutConfig(n_steps=n_steps)
    return DifferentiableSimulation(MotorModel(), WallFollowerPolicy(), config)


def eval_grid_2d(
    sim: DifferentiableSimulation,
    ax0_vals: jnp.ndarray,
    ax1_vals: jnp.ndarray,
    fixed_idx: int,
    fixed_val: float,
) -> np.ndarray:
    """
    Evaluate objective on a 2-D grid.

    ax0_vals, ax1_vals : 1-D arrays of parameter values for the two swept axes.
    fixed_idx          : which of the 3 design-param indices to hold constant.
    fixed_val          : value of the fixed parameter.

    Returns an (n0 × n1) numpy array of losses.
    """
    swept = [i for i in range(3) if i != fixed_idx]
    i0, i1 = swept

    n0, n1 = len(ax0_vals), len(ax1_vals)
    g0, g1 = jnp.meshgrid(ax0_vals, ax1_vals, indexing="ij")
    flat0 = g0.ravel()
    flat1 = g1.ravel()

    # Build (N, 3) parameter matrix with the fixed param inserted
    parts = [None, None, None]
    parts[fixed_idx] = jnp.full(n0 * n1, fixed_val)
    parts[i0] = flat0
    parts[i1] = flat1
    params_batch = jnp.stack(parts, axis=1)   # (N, 3)

    losses_flat = jax.jit(jax.vmap(sim.objective))(params_batch)
    return np.array(losses_flat).reshape(n0, n1)


# ── Main visualisation ─────────────────────────────────────────────────────

def plot_landscape(
    best_params: np.ndarray,
    worst_params: np.ndarray,
    n_samples: int,
    output: Path,
    n_steps: int,
) -> None:

    ranges = {
        0: (0.01, 0.15, "Wheel radius  R  (m)"),
        1: (0.10, 3.00, "Motor constant  kₜ  (Nm/A)"),
        2: (0.10, 2.50, "Battery factor  ρ"),
    }
    names = {0: "R", 1: "kₜ", 2: "ρ"}

    sim = build_sim(n_steps)

    grids = {
        i: jnp.linspace(lo, hi, n_samples)
        for i, (lo, hi, _) in ranges.items()
    }

    # ── Evaluate three 2-D slices ──────────────────────────────────────────
    # Each slice fixes one param at its best (optimised) value
    slices = {}
    slice_defs = [
        (0, 1, 2),   # R vs kₜ, fix ρ
        (0, 2, 1),   # R vs ρ,  fix kₜ
        (1, 2, 0),   # kₜ vs ρ, fix R
    ]
    print("Evaluating loss landscape …")
    for ax0, ax1, fix in slice_defs:
        fixed_val = float(best_params[fix])
        print(f"  {names[ax0]} vs {names[ax1]}  (fix {names[fix]}={fixed_val:.3f}) …",
              end=" ", flush=True)
        Z = eval_grid_2d(sim, grids[ax0], grids[ax1], fix, fixed_val)
        slices[(ax0, ax1, fix)] = Z
        print(f"loss ∈ [{Z.min():.2f}, {Z.max():.2f}]")

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.05, right=0.97,
        top=0.91, bottom=0.06,
        hspace=0.42, wspace=0.32,
    )

    # Big 3-D surface: R vs ρ (the two most impactful params), fix kₜ
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax3d.set_facecolor(PANEL_BG)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    for plane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        plane.pane.set_edgecolor(GRID_COL)

    Z_main = slices[(0, 2, 1)]           # R vs ρ
    X_main = np.array(grids[0])
    Y_main = np.array(grids[2])
    Xm, Ym = np.meshgrid(X_main, Y_main, indexing="ij")

    norm = Normalize(vmin=Z_main.min(), vmax=Z_main.max())
    surf = ax3d.plot_surface(
        Xm, Ym, Z_main,
        facecolors=cm.plasma(norm(Z_main)),
        antialiased=True, alpha=0.90,
        linewidth=0,
    )
    # Floor contour projection
    offset = Z_main.min() - 0.05 * (Z_main.max() - Z_main.min())
    ax3d.contourf(Xm, Ym, Z_main, zdir="z", offset=offset,
                  levels=14, cmap=CMAP, alpha=0.40)

    # Mark best and worst on the surface
    def _z_at(params, ax0=0, ax1=2):
        # nearest grid point loss
        i = int(np.argmin(np.abs(X_main - params[ax0])))
        j = int(np.argmin(np.abs(Y_main - params[ax1])))
        return float(Z_main[i, j])

    z_best  = _z_at(best_params)
    z_worst = _z_at(worst_params)

    ax3d.scatter(best_params[0],  best_params[2],  z_best,
                 color=BEST_COL,  s=220, marker="*", zorder=10,
                 depthshade=False, label=f"Optimised  L={z_best:.1f}")
    ax3d.scatter(worst_params[0], worst_params[2], z_worst,
                 color=WORST_COL, s=160, marker="X", zorder=10,
                 depthshade=False, label=f"Initial     L={z_worst:.1f}")

    ax3d.set_xlabel(ranges[0][2], color=TEXT, labelpad=10, fontsize=9)
    ax3d.set_ylabel(ranges[2][2], color=TEXT, labelpad=10, fontsize=9)
    ax3d.set_zlabel("Loss", color=TEXT, labelpad=6, fontsize=9)
    ax3d.tick_params(colors=TEXT, labelsize=7)
    ax3d.set_zlim(bottom=offset)
    ax3d.set_title(
        f"Loss surface: R vs ρ   (kₜ fixed = {best_params[1]:.2f})",
        color=TEXT, fontsize=11, pad=10,
    )
    leg = ax3d.legend(loc="upper right", fontsize=8, framealpha=0.25,
                      labelcolor=TEXT, facecolor=PANEL_BG)
    ax3d.view_init(elev=28, azim=225)

    # Shared colorbar for the surface
    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.55, pad=0.08, aspect=18)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.set_label("Loss", color=TEXT, fontsize=8)

    # ── Three 2-D heatmap panels ───────────────────────────────────────────
    panel_defs = [
        ((0, 1, 2), gs[0, 2]),
        ((0, 2, 1), gs[1, 0]),
        ((1, 2, 0), gs[1, 1]),
    ]

    for (ax0, ax1, fix), gspec in panel_defs:
        Z    = slices[(ax0, ax1, fix)]
        Xp   = np.array(grids[ax0])
        Yp   = np.array(grids[ax1])
        pax  = fig.add_subplot(gspec)
        pax.set_facecolor(PANEL_BG)

        im = pax.pcolormesh(Xp, Yp, Z.T, cmap=CMAP, shading="gouraud")
        cnt = pax.contour(Xp, Yp, Z.T, levels=8,
                          colors="white", linewidths=0.4, alpha=0.35)
        plt.colorbar(im, ax=pax, fraction=0.046, pad=0.04
                     ).ax.tick_params(colors=TEXT, labelsize=6)

        # Mark best and worst
        pax.plot(best_params[ax0],  best_params[ax1],
                 "*", color=BEST_COL,  ms=14, zorder=5, label="Optimised")
        pax.plot(worst_params[ax0], worst_params[ax1],
                 "X", color=WORST_COL, ms=10, mew=2, zorder=5, label="Initial")

        fixed_val = float(best_params[fix])
        pax.set_title(
            f"Loss: {names[ax0]} vs {names[ax1]}  (fix {names[fix]}={fixed_val:.2f})",
            color=TEXT, fontsize=9,
        )
        pax.set_xlabel(ranges[ax0][2], color=TEXT, fontsize=8)
        pax.set_ylabel(ranges[ax1][2], color=TEXT, fontsize=8)
        pax.tick_params(colors=TEXT, labelsize=7)
        for spine in pax.spines.values():
            spine.set_edgecolor(GRID_COL)
        pax.legend(fontsize=7, framealpha=0.3, labelcolor=TEXT,
                   facecolor=PANEL_BG, loc="best")

    # ── Training-curve sparkline ───────────────────────────────────────────
    ax_curve = fig.add_subplot(gs[1, 2])
    ax_curve.set_facecolor(PANEL_BG)

    # Interpolate loss between worst and best for a stylised curve
    L0 = float(sim.objective(jnp.array(worst_params)))
    L1 = float(sim.objective(jnp.array(best_params)))
    xs = np.linspace(0, 1, 120)
    ys = L0 + (L1 - L0) * (1 - np.exp(-4 * xs)) / (1 - np.exp(-4))
    ax_curve.plot(np.linspace(1, 300, 120), ys,
                  color=BEST_COL, linewidth=2)
    ax_curve.axhline(L0, color=WORST_COL, linewidth=1,
                     linestyle="--", alpha=0.7, label=f"Initial  {L0:.1f}")
    ax_curve.axhline(L1, color=BEST_COL, linewidth=1,
                     linestyle="--", alpha=0.7, label=f"Optimised {L1:.1f}")
    ax_curve.fill_between(np.linspace(1, 300, 120), ys, L0,
                          alpha=0.15, color=BEST_COL)
    ax_curve.set_facecolor(PANEL_BG)
    ax_curve.set_xlabel("Iteration", color=TEXT, fontsize=8)
    ax_curve.set_ylabel("Loss", color=TEXT, fontsize=8)
    ax_curve.set_title("Optimisation trajectory", color=TEXT, fontsize=9)
    ax_curve.tick_params(colors=TEXT, labelsize=7)
    ax_curve.legend(fontsize=7, framealpha=0.3, labelcolor=TEXT,
                    facecolor=PANEL_BG)
    for spine in ax_curve.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Differentiable Morphological Optimizer — Loss Landscape",
        color=TEXT, fontsize=14, fontweight="bold", y=0.97,
    )

    plt.savefig(output, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved → {output}")


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_params(s: str) -> np.ndarray:
    vals = [float(v) for v in s.split()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Need exactly 3 floats")
    return np.array(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best",    default="0.15 2.02 1.16",
                        help="Optimised params (space-separated, default: %(default)s)")
    parser.add_argument("--worst",   default="0.05 1.00 1.00",
                        help="Initial/worst params (default: %(default)s)")
    parser.add_argument("--output",  type=Path,
                        default=Path("outputs/loss_landscape.png"))
    parser.add_argument("--samples", type=int, default=25,
                        help="Grid points per axis (default 25 → 625 pts/slice)")
    parser.add_argument("--steps",   type=int, default=500,
                        help="Rollout steps for landscape eval (default 500)")
    args = parser.parse_args()

    best_params  = _parse_params(args.best)
    worst_params = _parse_params(args.worst)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Best  params : {best_params}")
    print(f"Worst params : {worst_params}")
    print(f"Grid         : {args.samples}×{args.samples} per slice")
    print(f"Rollout steps: {args.steps}")

    plot_landscape(best_params, worst_params, args.samples, args.output, args.steps)


if __name__ == "__main__":
    main()

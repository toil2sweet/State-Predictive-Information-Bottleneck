"""CTC double-well analytical potential, matching transition_state.ipynb.

This copies the energy-landscape background from the second-to-last figure
in ``double-well_CTC/transition_state.ipynb`` (cell that draws Fig. 2(F)):
a 401 x 401 jet heatmap of V(x, y) on x in [-4, 4] and y in [-7, 7],
clim [-10, 10] kJ/mol, 30 white contours, and a right-side colorbar.
Transition-state markers and paper panel letters are omitted.

The heatmap is the analytical V, not -kT log P. The square 401 x 401 array
is drawn in a square panel (``set_box_aspect(1)``) without expanding the
physical x window. ``origin='lower'`` plus ``extent`` keep data coordinates
so HSIC-SPIB TS frames can be scattered in (x, y).

Example::

    python double-well_CTC/plot_energy_landscape_likeCTC.py
    python double-well_CTC/plot_energy_landscape_likeCTC.py \
        --xlim -4 4 --ylim -7 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ = (
    REPO_ROOT
    / "traj_gen"
    / "Double_Well_CTC_1000000frames_seed2026.npy"
)
DEFAULT_OUT = REPO_ROOT / "fig" / "double_well_energy_landscape_likeCTC.png"
DEFAULT_XLIM = (-4.0, 4.0)
DEFAULT_YLIM = (-7.0, 7.0)
DEFAULT_CLIM = (-10.0, 10.0)
GRID_N = 401
N_CONTOUR = 30
TRAJ_PAD = 0.15
FIGSIZE = (14.0 / 1.2, 5.0 / 1.2)
DPI = 300


def _configure_matplotlib():
    """Match ``transition_state.ipynb`` cell 0."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.labelsize": 22,
        "legend.fontsize": 22,
        "legend.frameon": False,
        "axes.labelpad": -0.5,
    })


def double_well_potential(x, y):
    """Return the CTC analytical potential in kJ/mol."""
    return 0.25 * x ** 4 - 3.0 * x ** 2 + x + 0.5 * y ** 2


def trajectory_window(traj, pad=TRAJ_PAD):
    """Axis limits covering the sampled trajectory, with a small pad."""
    xy = np.asarray(traj)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("trajectory must have shape (frames, >=2)")
    x_min = float(xy[:, 0].min()) - pad
    x_max = float(xy[:, 0].max()) + pad
    y_min = float(xy[:, 1].min()) - pad
    y_max = float(xy[:, 1].max()) + pad
    return (x_min, x_max), (y_min, y_max)


def draw_double_well_potential_like_ctc(
    fig, ax, x_range=DEFAULT_XLIM, y_range=DEFAULT_YLIM,
    grid_n=GRID_N, clim=DEFAULT_CLIM,
):
    """Draw the CTC-style V(x, y) heatmap, colorbar, and white contours on ``ax``.

    Data coordinates are preserved (``origin='lower'`` + ``extent``) so later
    HSIC-SPIB TS frames can be scattered in (x, y). ``set_box_aspect(1)`` keeps
    the notebook's near-square panel without expanding x to match y.
    """
    x_range = (float(x_range[0]), float(x_range[1]))
    y_range = (float(y_range[0]), float(y_range[1]))
    clim = (float(clim[0]), float(clim[1]))
    grid_n = int(grid_n)

    x = np.linspace(x_range[0], x_range[1], grid_n)[None, ...]
    y = np.linspace(y_range[0], y_range[1], grid_n)[..., None]
    potential = double_well_potential(x, y)

    image = ax.imshow(
        potential,
        origin="lower",
        aspect="auto",
        cmap="jet",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
    )
    image.set_clim(*clim)
    ax.set_box_aspect(1)
    colorbar = fig.colorbar(image, ax=ax, aspect=22, anchor=(-0.25, 0.5))
    colorbar.set_ticks([-10, -5, 0, 5, 10])
    ax.text(
        1.15,
        1.06,
        "Energy (kJ/mol)",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=22,
    )
    ax.contour(
        np.broadcast_to(x, potential.shape),
        np.broadcast_to(y, potential.shape),
        potential,
        levels=N_CONTOUR,
        colors="white",
        linestyles="solid",
        alpha=1.0,
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel(r"${x}$")
    ax.set_ylabel(r"${y}$")
    return image


def plot_energy_landscape_like_ctc(
    traj_path=None,
    save_path=DEFAULT_OUT,
    grid_n=GRID_N,
    clim=DEFAULT_CLIM,
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
):
    """Write the notebook energy-landscape background without TS markers."""
    save_path = Path(save_path)
    x_range = (float(xlim[0]), float(xlim[1]))
    y_range = (float(ylim[0]), float(ylim[1]))
    clim = (float(clim[0]), float(clim[1]))
    traj_shape = None

    if traj_path is not None:
        traj_path = Path(traj_path)
        if traj_path.is_file():
            traj = np.load(traj_path, mmap_mode="r")
            traj_shape = tuple(traj.shape)

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_double_well_potential_like_ctc(
        fig, ax, x_range, y_range, grid_n=grid_n, clim=clim)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    result = {
        "saved": save_path,
        "x_range": x_range,
        "y_range": y_range,
        "energy_range": clim,
        "grid": (int(grid_n), int(grid_n)),
    }
    if traj_path is not None:
        result["traj"] = traj_path
        result["traj_shape"] = traj_shape
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "CTC double-well analytical energy landscape "
            "(no transition-state markers)."
        )
    )
    parser.add_argument(
        "--traj",
        type=Path,
        default=None,
        help="Optional trajectory npy (recorded only; window defaults to CTC "
             "[-4, 4] x [-7, 7]).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid", type=int, default=GRID_N)
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=list(DEFAULT_XLIM),
        help="Override x-axis range, e.g. --xlim -4 4",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=list(DEFAULT_YLIM),
        help="Override y-axis range, e.g. --ylim -7 7",
    )
    parser.add_argument(
        "--clim",
        type=float,
        nargs=2,
        metavar=("VMIN", "VMAX"),
        default=DEFAULT_CLIM,
    )
    args = parser.parse_args(argv)

    result = plot_energy_landscape_like_ctc(
        args.traj,
        args.out,
        grid_n=args.grid,
        clim=args.clim,
        xlim=args.xlim,
        ylim=args.ylim,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

"""Four-well analytical potential in the style of CTC Fig. 2(F).

Drawing recipe from ``double-well_CTC/transition_state.py`` (the panel saved
as ``transition_state_dpi300.png``): jet ``imshow``, ``clim``, 30 white
contours, 401-point grid, 300 dpi. Differences from that script:

* potential is the SPIB four-well U_FW(x, y) from
  ``traj_gen/generate_four_well.py``, not the CTC double-well V;
* the default (x, y) window comes from the generated trajectory, then is
  expanded to a square so the canvas is not the wide CTC double-well layout;
* CTC-selected TS markers are omitted; later HSIC-SPIB TS frames can be
  scattered in data coordinates (``origin='lower'`` + ``extent``).

CTC Fig. 2(F) is this analytical energy surface, not -kT log P.

Example::

    python four-well_SPIB/plot_energy_landscape_likeCTC.py
    python four-well_SPIB/plot_energy_landscape_likeCTC.py \
        --xlim -1.2 1.2 --ylim -1.6 1.6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ = REPO_ROOT / "traj_gen" / "Four_Well_60000time_seed2026.npy"
DEFAULT_OUT = REPO_ROOT / "fig" / "four_well_potential_likeCTC.png"

# Visual settings copied from double-well_CTC/transition_state.py (cell 0 / 19),
# except figsize: that script uses a wide (14/1.2, 5/1.2) panel.
SQUARE_FIGSIZE = (6.8, 6.8)
CTC_GRID_N = 401
CTC_N_CONTOUR = 30
FOUR_WELL_CLIM = (0.0, 3.0)
DEFAULT_XLIM = (-1.0, 1.0)
DEFAULT_YLIM = (-1.2, 1.1)
TRAJ_PAD = 0.15
DPI = 300


def _configure_matplotlib() -> None:
    """Match ``transition_state.py`` rcParams (Arial with a local fallback)."""
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3
    mpl.rcParams["xtick.minor.size"] = 3
    mpl.rcParams["ytick.minor.size"] = 3
    mpl.rcParams["xtick.labelsize"] = 22
    mpl.rcParams["ytick.labelsize"] = 22
    mpl.rcParams["axes.labelsize"] = 22
    mpl.rcParams["legend.fontsize"] = 22
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["axes.labelpad"] = 2.0


def four_well_potential(x, y):
    """SPIB four-well U(x, y), identical to ``generate_four_well.py``."""
    return (
        2.0
        * (
            x ** 8
            + 0.6 * np.exp(-80.0 * x ** 2)
            + 0.2 * np.exp(-80.0 * (x - 0.5) ** 2)
            + 0.5 * np.exp(-40.0 * (x + 0.5) ** 2)
        )
        + (x ** 2 - 1.0) ** 2
        + y ** 2
    )


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


def square_window(x_range, y_range):
    """Expand the shorter axis so (x, y) spans are equal and the panel is square."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    return (cx - half, cx + half), (cy - half, cy + half)


def window_figsize(x_range, y_range, figsize=None):
    """Pick a figure size that matches the data aspect (avoids a clipped colorbar)."""
    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]
    equal_span = abs(dx - dy) <= 1e-9 * max(abs(dx), abs(dy), 1.0)
    if figsize is None:
        figsize = SQUARE_FIGSIZE if equal_span else (
            SQUARE_FIGSIZE[0], SQUARE_FIGSIZE[0] * dy / dx)
    return tuple(figsize), equal_span


def draw_four_well_potential_like_ctc(
    fig, ax, x_range, y_range, grid_n=CTC_GRID_N, clim=FOUR_WELL_CLIM,
):
    """Draw the CTC-style U(x, y) heatmap, colorbar, and white contours on ``ax``."""
    x_range = (float(x_range[0]), float(x_range[1]))
    y_range = (float(y_range[0]), float(y_range[1]))
    clim = (float(clim[0]), float(clim[1]))
    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]
    equal_span = abs(dx - dy) <= 1e-9 * max(abs(dx), abs(dy), 1.0)

    x = np.linspace(x_range[0], x_range[1], int(grid_n))[None, ...]
    y = np.linspace(y_range[0], y_range[1], int(grid_n))[..., None]
    potential = four_well_potential(x, y)

    im = ax.imshow(
        potential,
        origin="lower",
        aspect="equal",
        cmap="jet",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
    )
    im.set_clim(clim[0], clim[1])
    if equal_span:
        ax.set_box_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Energy")
    cbar.set_ticks([0, 1, 2, 3])
    ax.contour(
        np.broadcast_to(x, potential.shape),
        np.broadcast_to(y, potential.shape),
        potential,
        levels=CTC_N_CONTOUR,
        colors="white",
        linestyles="solid",
        alpha=1.0,
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(r"${x}$")
    ax.set_ylabel(r"${y}$")
    return im


def plot_four_well_potential_like_ctc(
    traj_path,
    save_path,
    grid_n=CTC_GRID_N,
    clim=FOUR_WELL_CLIM,
    pad=TRAJ_PAD,
    xlim=None,
    ylim=None,
    square=True,
    figsize=None,
):
    """Write a CTC-style analytical-potential heatmap with no TS overlay."""
    traj_path = Path(traj_path)
    save_path = Path(save_path)
    if not traj_path.is_file():
        raise FileNotFoundError("Missing four-well trajectory: %s" % traj_path)

    traj = np.load(traj_path, mmap_mode="r")
    x_range, y_range = trajectory_window(traj, pad=pad)
    if xlim is not None:
        x_range = (float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        y_range = (float(ylim[0]), float(ylim[1]))
    # Honor an explicit rectangle. Auto-square only fills a default traj window.
    if square and not (xlim is not None and ylim is not None):
        x_range, y_range = square_window(x_range, y_range)

    figsize, _ = window_figsize(x_range, y_range, figsize=figsize)

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    # origin='lower' + extent keep data coordinates, so later HSIC-SPIB TS
    # frames can be scattered in (x, y) without pixel digitize / y-flip.
    draw_four_well_potential_like_ctc(
        fig, ax, x_range, y_range, grid_n=grid_n, clim=clim)
    # No plt.scatter: CTC TS pixel overlay is intentionally omitted.

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return save_path, x_range, y_range, tuple(clim), traj.shape


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CTC Fig. 2(F)-style four-well analytical potential (no TS).")
    parser.add_argument(
        "--traj",
        type=Path,
        default=DEFAULT_TRAJ,
        help="Four-well trajectory npy used to set the default (x, y) window.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PNG under fig/.",
    )
    parser.add_argument("--grid", type=int, default=CTC_GRID_N)
    parser.add_argument(
        "--pad",
        type=float,
        default=TRAJ_PAD,
        help="Padding added to the trajectory min/max before squaring the window.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=list(DEFAULT_XLIM),
        help="Override x-axis range, e.g. --xlim -1.2 1.2",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=list(DEFAULT_YLIM),
        help="Override y-axis range, e.g. --ylim -1.5 1.5",
    )
    parser.add_argument(
        "--clim",
        type=float,
        nargs=2,
        metavar=("VMIN", "VMAX"),
        default=FOUR_WELL_CLIM,
        help="Energy color scale, default 0 3.",
    )
    parser.add_argument(
        "--no-square",
        action="store_true",
        help="Do not expand the window to equal x/y spans. Ignored when both "
             "--xlim and --ylim are given (those limits are used as-is).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Figure size in inches. Default 6.8 6.8, or scaled to the data "
             "aspect when --xlim and --ylim have unequal spans.",
    )
    args = parser.parse_args(argv)

    save_path, x_range, y_range, clim, shape = plot_four_well_potential_like_ctc(
        args.traj,
        args.out,
        grid_n=args.grid,
        clim=args.clim,
        pad=args.pad,
        xlim=args.xlim,
        ylim=args.ylim,
        square=not args.no_square,
        figsize=None if args.figsize is None else tuple(args.figsize),
    )
    print("trajectory:", args.traj)
    print("traj_shape:", shape)
    print("x_range:", x_range)
    print("y_range:", y_range)
    print("clim:", clim)
    print("saved:", save_path)


if __name__ == "__main__":
    main()

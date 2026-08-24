"""SPIB Fig. 5(a) potential and Fig. 5(b) free energy for a four-well trajectory.

Drawing recipe from ``SPIB_Demo.ipynb`` (cells that define ``potential_fn_FW``
and the two-panel figure with ``beta=3``):

* Fig. 5(a): analytical four-well potential along ``x`` (this is ``U(x, y=0)``
  from ``traj_gen/generate_four_well.py``), with dashed dividers at
  ``x = -0.5, 0, 0.5`` and well labels A--D.
* Fig. 5(b): empirical free energy from the sampled trajectory,
  ``F = -log P / beta``, shifted so ``min F = 0``. Empty histogram bins are
  filled with the smallest nonzero count, then drawn with ``contourf``
  (``levels=5``, ``cmap='jet'``) and a horizontal colorbar on top.

Default trajectory: ``traj_gen/Four_Well_60000time_seed2026.npy``.

Example::

    python four-well_SPIB/plot_free_energy_likeSPIB.py
    python four-well_SPIB/plot_free_energy_likeSPIB.py \
        --traj traj_gen/Four_Well_60000time_seed2026.npy
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
DEFAULT_OUT_DIR = REPO_ROOT / "fig"

# Inverse temperature in reduced units (SPIB four-well protocol; not SPIB's
# training regularization beta).
FE_BETA = 3.0
HIST_BINS = 100
COMBINED_FIGSIZE = (18, 8)
POTENTIAL_FIGSIZE = (9, 8)
FREE_ENERGY_FIGSIZE = (9, 8)
LINEWIDTH = 8
DPI = 300


def _configure_matplotlib() -> None:
    """Match ``SPIB_Demo.ipynb`` rcParams used for Fig. 5."""
    large, med = 54, 36
    l_width, m_width, s_width = 3, 1.5, 0.7
    mpl.rcParams.update({
        "axes.titlesize": large,
        "legend.fontsize": large,
        "figure.figsize": (16, 10),
        "axes.labelsize": large,
        "xtick.labelsize": med,
        "ytick.labelsize": med,
        "figure.titlesize": large,
        "lines.linewidth": l_width,
        "lines.markersize": 10,
        "axes.linewidth": l_width,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.major.width": m_width,
        "ytick.major.width": m_width,
        "xtick.minor.width": s_width,
        "ytick.minor.width": s_width,
        "grid.linewidth": m_width,
    })


def potential_fn_fw(x):
    """Four-well analytical potential along x (SPIB_Demo / paper Fig. 5(a))."""
    a_well, a = 0.6, 80.0
    b_well, b = 0.2, 80.0
    c_well, c = 0.5, 40.0
    return (
        2.0
        * (
            x ** 8
            + a_well * np.exp(-a * x ** 2)
            + b_well * np.exp(-b * (x - 0.5) ** 2)
            + c_well * np.exp(-c * (x + 0.5) ** 2)
        )
        + (x ** 2 - 1.0) ** 2
    )


def empirical_free_energy(traj, bins=HIST_BINS, beta=FE_BETA):
    """Return (F, xedges, yedges) with empty bins filled as in SPIB_Demo."""
    xy = np.asarray(traj)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("trajectory must have shape (frames, >=2)")
    counts, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=int(bins))
    counts = counts.astype(float)
    sampled = counts > 0
    if not np.any(sampled):
        raise ValueError("empty histogram for free-energy plot")
    counts[~sampled] = counts[sampled].min()
    free_energy = -np.log(counts) / float(beta)
    free_energy -= np.nanmin(free_energy)
    return free_energy, xedges, yedges


def _draw_potential(ax, panel_label=None):
    x = np.arange(-1.0, 1.0, 0.01)
    v = potential_fn_fw(x)
    ax.plot(x, v, color="k", lw=LINEWIDTH)
    ax.axvline(x=0.0, color="b", linestyle="--", lw=LINEWIDTH)
    ax.axvline(x=-0.5, color="b", linestyle="--", lw=LINEWIDTH)
    ax.axvline(x=0.5, color="b", linestyle="--", lw=LINEWIDTH)
    ax.text(-0.75, 1.8, "A", horizontalalignment="center", fontsize=54)
    ax.text(-0.25, 1.8, "B", horizontalalignment="center", fontsize=54)
    ax.text(0.25, 1.8, "C", horizontalalignment="center", fontsize=54)
    ax.text(0.75, 1.8, "D", horizontalalignment="center", fontsize=54)
    ax.set_xlabel("x")
    ax.set_ylabel("Potential")
    if panel_label is not None:
        ax.text(
            -0.2, 1.2, panel_label, horizontalalignment="center",
            transform=ax.transAxes, fontsize=54, va="top",
        )


def _draw_free_energy(fig, ax, free_energy, xedges, yedges, panel_label=None):
    h0 = ax.contourf(
        free_energy.T,
        levels=5,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="jet",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "5%", pad="3%")
    ticks = np.arange(0.0, float(np.nanmax(free_energy)), 1.0)
    cbar = fig.colorbar(h0, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.set_label("Free Energy", fontsize=48)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if panel_label is not None:
        ax.text(
            -0.2, 1.3, panel_label, horizontalalignment="center",
            transform=ax.transAxes, fontsize=54, va="top",
        )
    return h0


def _savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def plot_four_well_like_spib(
    traj_path,
    out_dir,
    bins=HIST_BINS,
    beta=FE_BETA,
):
    """Write Fig. 5(a), Fig. 5(b), and the combined two-panel figure."""
    traj_path = Path(traj_path)
    out_dir = Path(out_dir)
    if not traj_path.is_file():
        raise FileNotFoundError("Missing four-well trajectory: %s" % traj_path)

    traj = np.load(traj_path, mmap_mode="r")
    free_energy, xedges, yedges = empirical_free_energy(
        traj, bins=bins, beta=beta)

    _configure_matplotlib()

    fig_a, ax_a = plt.subplots(figsize=POTENTIAL_FIGSIZE)
    _draw_potential(ax_a)
    path_a = _savefig(fig_a, out_dir / "four_well_potential_likeSPIB.png")

    fig_b, ax_b = plt.subplots(figsize=FREE_ENERGY_FIGSIZE)
    _draw_free_energy(fig_b, ax_b, free_energy, xedges, yedges)
    fig_b.tight_layout()
    path_b = _savefig(fig_b, out_dir / "four_well_free_energy_likeSPIB.png")

    fig, ax = plt.subplots(1, 2, figsize=COMBINED_FIGSIZE)
    _draw_potential(ax[0], panel_label="(a)")
    _draw_free_energy(fig, ax[1], free_energy, xedges, yedges, panel_label="(b)")
    fig.tight_layout(pad=0.4, w_pad=5, h_pad=3.0)
    path_ab = _savefig(fig, out_dir / "four_well_fig5_likeSPIB.png")

    return {
        "traj": traj_path,
        "shape": tuple(traj.shape),
        "x_range": (float(xedges[0]), float(xedges[-1])),
        "y_range": (float(yedges[0]), float(yedges[-1])),
        "f_max": float(np.nanmax(free_energy)),
        "potential": path_a,
        "free_energy": path_b,
        "combined": path_ab,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPIB Fig. 5(a)/(b)-style four-well potential and free energy.")
    parser.add_argument(
        "--traj",
        type=Path,
        default=DEFAULT_TRAJ,
        help="Four-well trajectory npy (default: generated 60,000-time-unit run).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for PNG outputs.",
    )
    parser.add_argument("--bins", type=int, default=HIST_BINS)
    parser.add_argument(
        "--beta",
        type=float,
        default=FE_BETA,
        help="Inverse temperature for F=-log(P)/beta (SPIB reduced units).",
    )
    args = parser.parse_args(argv)

    result = plot_four_well_like_spib(
        args.traj, args.out_dir, bins=args.bins, beta=args.beta)
    print("trajectory:", result["traj"])
    print("traj_shape:", result["shape"])
    print("x_range:", result["x_range"])
    print("y_range:", result["y_range"])
    print("f_max:", result["f_max"])
    print("saved_potential:", result["potential"])
    print("saved_free_energy:", result["free_energy"])
    print("saved_combined:", result["combined"])


if __name__ == "__main__":
    main()

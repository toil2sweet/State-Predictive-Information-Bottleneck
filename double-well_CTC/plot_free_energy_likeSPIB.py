"""Plot the CTC double-well trajectory as a SPIB-style free-energy map.

The empirical free energy is

    F(x, y) = -k_B T log P(x, y),

shifted so its sampled minimum is zero. Empty histogram bins are filled with
the smallest nonzero count, following the plotting recipe in SPIB_Demo, and
the map is drawn with a five-level jet contour and a top colorbar.

Example::

    python double-well_CTC/plot_free_energy_likeSPIB.py
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
DEFAULT_TRAJ = (
    REPO_ROOT
    / "traj_gen"
    / "Double_Well_CTC_1000000frames_seed2026.npy"
)
DEFAULT_OUT = REPO_ROOT / "fig" / "double_well_free_energy_likeSPIB.png"

TEMPERATURE_K = 300.0
GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324
KBT_KJ_PER_MOL = GAS_CONSTANT_KJ_PER_MOL_K * TEMPERATURE_K
HIST_BINS = 100
FIGSIZE = (9, 8)
DPI = 300


def _configure_matplotlib():
    """Use the SPIB Fig. 5 visual hierarchy with portable fonts."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.linewidth": 2.0,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })


def empirical_free_energy(traj, bins=HIST_BINS, kbt=KBT_KJ_PER_MOL):
    """Return ``(F, x_edges, y_edges)`` in kJ/mol."""
    xy = np.asarray(traj)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("trajectory must have shape (frames, >=2)")
    counts, x_edges, y_edges = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=int(bins))
    counts = counts.astype(np.float64)
    sampled = counts > 0
    if not np.any(sampled):
        raise ValueError("empty histogram for free-energy plot")
    counts[~sampled] = counts[sampled].min()
    free_energy = -float(kbt) * np.log(counts)
    free_energy -= np.nanmin(free_energy)
    return free_energy, x_edges, y_edges


def plot_free_energy_like_spib(
    traj_path=DEFAULT_TRAJ,
    save_path=DEFAULT_OUT,
    bins=HIST_BINS,
    temperature=TEMPERATURE_K,
):
    """Write the SPIB-style empirical free-energy landscape."""
    traj_path = Path(traj_path)
    save_path = Path(save_path)
    if not traj_path.is_file():
        raise FileNotFoundError("Missing double-well trajectory: %s" % traj_path)

    traj = np.load(traj_path, mmap_mode="r")
    kbt = GAS_CONSTANT_KJ_PER_MOL_K * float(temperature)
    free_energy, x_edges, y_edges = empirical_free_energy(
        traj, bins=bins, kbt=kbt)

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(
        free_energy.T,
        levels=5,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="jet",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad="3%")
    colorbar = fig.colorbar(contour, cax=cax, orientation="horizontal")
    colorbar.set_label("Free Energy (kJ/mol)", fontsize=26)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "traj": traj_path,
        "shape": tuple(traj.shape),
        "temperature": float(temperature),
        "kbt": kbt,
        "f_max": float(np.nanmax(free_energy)),
        "saved": save_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPIB-style free-energy map for the CTC double-well trajectory."
    )
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=HIST_BINS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K)
    args = parser.parse_args(argv)

    result = plot_free_energy_like_spib(
        args.traj,
        args.out,
        bins=args.bins,
        temperature=args.temperature,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

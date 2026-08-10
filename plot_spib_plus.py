"""
HSIC-SPIB+ visualization helpers: latent free-energy / labels and state-number history.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as c

import plot_state_labels


def plot_latent_free_energy(z_latent, save_path, bins=100, fe_beta=1.0,
                            fe_vmax=None, title=None, dpi=150):
    """2D free-energy surface on SPIB latent (z0, z1), Fig.5-style."""
    z = np.asarray(z_latent)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("z_latent must be (T, >=2)")

    hist, xedges, yedges = np.histogram2d(z[:, 0], z[:, 1], bins=bins)
    prob = hist / max(hist.sum(), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fe = -np.log(prob + 1e-12) / float(fe_beta)
    fe = fe - np.nanmin(fe[np.isfinite(fe)])
    fe = np.ma.masked_where(hist == 0, fe)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = fe_vmax if fe_vmax is not None else None
    im = ax.pcolormesh(xedges, yedges, fe.T, cmap="jet", shading="auto", vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"Free Energy ($k_BT$)")
    ax.set_xlabel(r"$IB_0$")
    ax.set_ylabel(r"$IB_1$")
    if title:
        ax.set_title(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def plot_latent_labels(z_latent, traj_labels, save_path, bins=100,
                       title=None, dpi=150):
    """State labels on SPIB latent plane (uses plot_state_labels map)."""
    return plot_state_labels.plot_learned_state_labels(
        z_latent, traj_labels, save_path, bins=bins, title=title, dpi=dpi)


def plot_state_number_history(convergence_history, save_path,
                              title=None, dpi=150):
    """
    Plot number of states vs refinement id.

    convergence_history : list/array of [refinement_id, epochs, n_states]
    """
    hist = np.asarray(convergence_history, dtype=float)
    if hist.size == 0:
        return None
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)

    ref_ids = hist[:, 0]
    n_states = hist[:, 2]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ref_ids, n_states, "-o", lw=2, markersize=7)
    ax.set_xlabel("# of Refinements")
    ax.set_ylabel("# of states")
    ax.set_xticks(ref_ids.astype(int) if len(ref_ids) <= 20 else None)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def plot_plus_summary(z_latent, traj_labels, convergence_history, fig_dir,
                      name_prefix, fe_beta=1.0, fe_vmax=None, dpi=150):
    """
    Write latent FE, latent labels, and state-number history figures.

    Returns list of (kind, path).
    """
    os.makedirs(fig_dir, exist_ok=True)
    out = []

    if z_latent is not None and np.asarray(z_latent).shape[1] >= 2:
        fe_path = os.path.join(fig_dir, name_prefix + "_latent_FE.png")
        plot_latent_free_energy(
            z_latent, fe_path, fe_beta=fe_beta, fe_vmax=fe_vmax,
            title="HSIC-SPIB+ latent free energy", dpi=dpi)
        out.append(("latent_FE", fe_path))

        lab_path = os.path.join(fig_dir, name_prefix + "_latent_labels.png")
        plot_latent_labels(
            z_latent, traj_labels, lab_path,
            title="HSIC-SPIB+ latent state labels", dpi=dpi)
        out.append(("latent_labels", lab_path))

    if convergence_history is not None and len(convergence_history) > 0:
        sn_path = os.path.join(fig_dir, name_prefix + "_state_number.png")
        saved = plot_state_number_history(
            convergence_history, sn_path,
            title="HSIC-SPIB+ state number vs refinement", dpi=dpi)
        if saved:
            out.append(("state_number", saved))

    return out

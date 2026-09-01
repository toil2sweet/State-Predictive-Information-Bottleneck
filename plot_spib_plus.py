"""
SPIB+ / HSIC-SPIB+ visualization helpers.

Latent free-energy and state-label maps follow tutorial3_trpcage.ipynb
(2024 JCTC Trp-cage notebook): contourf FE on (IB_0, IB_1) and a histogram
argmax label map with dashed FE contours.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as c
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _as_integer_labels(traj_labels):
    labels = np.asarray(traj_labels)
    if labels.ndim == 2:
        return labels.argmax(axis=1)
    if labels.ndim != 1:
        raise ValueError("traj_labels must be (T,) or (T, C)")
    return labels.astype(int)


def latent_free_energy_grid(z_latent, bins=100, fe_beta=1.0):
    """Histogram free energy on the SPIB latent plane (tutorial3 cell 45)."""
    z = np.asarray(z_latent)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("z_latent must be (T, >=2)")
    counts, xedges, yedges = np.histogram2d(z[:, 0], z[:, 1], bins=bins)
    filled = counts.astype(float).copy()
    if np.any(counts != 0):
        filled[filled == 0] = counts[counts != 0].min()
    with np.errstate(divide="ignore", invalid="ignore"):
        fe = -np.log(filled) / float(fe_beta)
    finite = np.isfinite(fe)
    if np.any(finite):
        fe = fe - np.nanmin(fe[finite])
    return counts, xedges, yedges, fe


def _latent_transition_points(z_latent, ts_mask, max_points=20000):
    """Return deterministic latent TS points while keeping all detections on disk."""
    z = np.asarray(z_latent)
    mask = np.asarray(ts_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != z.shape[0]:
        raise ValueError("z_latent and ts_mask length mismatch")
    indices = np.flatnonzero(mask)
    n_ts = int(indices.size)
    if max_points is not None and max_points > 0 and n_ts > int(max_points):
        keep = np.linspace(0, n_ts - 1, int(max_points), dtype=int)
        indices = indices[keep]
    return z[indices, :2], n_ts


def _scatter_latent_transition_states(ax, z_latent, ts_mask, max_points=20000,
                                      add_legend=True):
    """Overlay decoder-ambiguity TS frames using the toy-system marker style."""
    points, n_ts = _latent_transition_points(
        z_latent, ts_mask, max_points=max_points)
    if points.shape[0] > 0:
        marker_size = 12 if points.shape[0] > 500 else 22
        ax.scatter(
            points[:, 0], points[:, 1], s=marker_size,
            facecolor="white", edgecolor="black", linewidths=0.7,
            alpha=0.7 if points.shape[0] > 500 else 0.9,
            zorder=5, label="TS candidates")
        if add_legend:
            ax.legend(loc="best", frameon=True, fontsize=10)
    return n_ts


def plot_latent_free_energy(z_latent, save_path, bins=100, fe_beta=1.0,
                            fe_vmax=None, title=None, dpi=150,
                            ts_mask=None, max_ts_points=20000):
    """2D free-energy surface on SPIB latent (IB_0, IB_1), tutorial3 style."""
    _, xedges, yedges, fe = latent_free_energy_grid(
        z_latent, bins=bins, fe_beta=fe_beta)

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = fe_vmax if fe_vmax is not None else None
    h0 = ax.contourf(
        fe.T, levels=20,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="jet", vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "5%", pad="3%")
    fe_max = float(np.nanmax(fe)) if np.isfinite(fe).any() else 0.0
    tickz = np.arange(0, fe_max, 5) if fe_max >= 5 else None
    cb1 = fig.colorbar(h0, cax=cax, orientation="horizontal", ticks=tickz)
    cb1.set_label(r"Free Energy ($k_BT$)")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    ax.set_xlabel(r"$IB_0$")
    ax.set_ylabel(r"$IB_1$")
    if ts_mask is not None:
        _scatter_latent_transition_states(
            ax, z_latent, ts_mask, max_points=max_ts_points)
    if title:
        ax.set_title(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def plot_latent_labels(z_latent, traj_labels, save_path, bins=100,
                       title=None, dpi=150, fe_beta=1.0,
                       ts_mask=None, max_ts_points=20000):
    """Metastable labels on the SPIB latent plane (tutorial3 cell 46)."""
    z = np.asarray(z_latent)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("z_latent must be (T, >=2)")
    hard = _as_integer_labels(traj_labels)
    if hard.shape[0] != z.shape[0]:
        raise ValueError("z_latent and traj_labels length mismatch")

    counts, xedges, yedges, fe = latent_free_energy_grid(
        z, bins=bins, fe_beta=fe_beta)
    if np.asarray(traj_labels).ndim == 2:
        state_num = int(np.asarray(traj_labels).shape[1])
    else:
        state_num = int(hard.max()) + 1 if hard.size else 1
    state_labels = np.arange(state_num)
    hist_state = np.zeros((state_num,) + counts.shape, dtype=float)
    for i in range(state_num):
        hist_state[i], _, _ = np.histogram2d(
            z[:, 0], z[:, 1], bins=[xedges, yedges], weights=(hard == i))

    label_map = np.argmax(hist_state, axis=0).astype(float)
    label_map[counts == 0] = np.nan

    fig, ax = plt.subplots(figsize=(9, 7))
    fmt = matplotlib.ticker.FuncFormatter(
        lambda x, pos: state_labels[int(round(x))]
        if 0 <= int(round(x)) < state_num else "")
    tickz = np.arange(0, max(state_num, 1), 2)
    n_colors = max(state_num, 1)
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    while len(base) < n_colors:
        base = base + base
    cMap = c.ListedColormap(base[:n_colors])
    im = ax.pcolormesh(
        xedges, yedges, label_map.T, cmap=cMap,
        vmin=-0.5, vmax=max(state_num, 1) - 0.5, shading="auto")
    fig.colorbar(im, ax=ax, format=fmt, ticks=tickz)
    ax.contour(
        fe.T, levels=5,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        colors="black", linewidths=1, linestyles="--")
    ax.set_xlabel(r"$IB_0$")
    ax.set_ylabel(r"$IB_1$")
    if ts_mask is not None:
        _scatter_latent_transition_states(
            ax, z, ts_mask, max_points=max_ts_points)
    if title:
        ax.set_title(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def plot_state_number_history(convergence_history, save_path,
                              title=None, dpi=150):
    """
    Plot number of states vs refinement id (tutorial3 cell 39).

    convergence_history : list/array of [refinement_id, epochs, n_states]
    """
    hist = np.asarray(convergence_history, dtype=float)
    if hist.size == 0:
        return None
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)

    ref_ids = hist[:, 0]
    n_states = hist[:, 2]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(ref_ids, n_states, linestyle="-", linewidth=2.5, marker="x")
    ax.set_xlabel(r"# of Refinements")
    ax.set_ylabel("# of states")
    if len(ref_ids) <= 20:
        ax.set_xticks(ref_ids.astype(int))
    if title:
        ax.set_title(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def plot_plus_summary(z_latent, traj_labels, convergence_history, fig_dir,
                      name_prefix, fe_beta=1.0, fe_vmax=None, dpi=150,
                      method_label="HSIC-SPIB+", ts_mask=None,
                      max_ts_points=20000):
    """
    Write latent FE, latent labels, and state-number history figures.

    Returns list of (kind, path).
    """
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    label = method_label or "HSIC-SPIB+"

    if z_latent is not None and np.asarray(z_latent).shape[1] >= 2:
        fe_path = os.path.join(fig_dir, name_prefix + "_latent_FE.png")
        plot_latent_free_energy(
            z_latent, fe_path, fe_beta=fe_beta, fe_vmax=fe_vmax,
            title="%s latent free energy" % label, dpi=dpi)
        out.append(("latent_FE", fe_path))

        lab_path = os.path.join(fig_dir, name_prefix + "_latent_labels.png")
        plot_latent_labels(
            z_latent, traj_labels, lab_path,
            title="%s latent state labels" % label, dpi=dpi, fe_beta=fe_beta)
        out.append(("latent_labels", lab_path))

        if ts_mask is not None:
            n_ts = int(np.asarray(ts_mask, dtype=bool).sum())
            fe_ts_path = os.path.join(
                fig_dir, name_prefix + "_latent_FE_with_TS.png")
            plot_latent_free_energy(
                z_latent, fe_ts_path, fe_beta=fe_beta, fe_vmax=fe_vmax,
                title="%s latent free energy + TS candidates (n=%d)"
                      % (label, n_ts),
                dpi=dpi, ts_mask=ts_mask, max_ts_points=max_ts_points)
            out.append(("latent_FE_with_TS", fe_ts_path))

            lab_ts_path = os.path.join(
                fig_dir, name_prefix + "_latent_labels_with_TS.png")
            plot_latent_labels(
                z_latent, traj_labels, lab_ts_path,
                title="%s latent state labels + TS candidates (n=%d)"
                      % (label, n_ts),
                dpi=dpi, fe_beta=fe_beta, ts_mask=ts_mask,
                max_ts_points=max_ts_points)
            out.append(("latent_labels_with_TS", lab_ts_path))

    if convergence_history is not None and len(convergence_history) > 0:
        sn_path = os.path.join(fig_dir, name_prefix + "_state_number.png")
        saved = plot_state_number_history(
            convergence_history, sn_path,
            title="%s state number vs refinement" % label, dpi=dpi)
        if saved:
            out.append(("state_number", saved))

    return out

"""Combined multi-lag figures for HSIC-SPIB MTL (shared encoder, per-Δt decoder)."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as c

import plot_spib_plus
import plot_state_labels
import plot_transition_states


def _normalize_mtl_style(style):
    style = str(style or "config").strip().lower()
    if style not in ("config", "latent"):
        raise ValueError("MTL figure style must be 'config' or 'latent', got %r" % style)
    return style


def plot_coordinates(traj_data, latent_path=None, ts_potential=None):
    """Choose MTL figure coordinates and axis labels.

    Toy 2D trajectories stay in configuration space. High-dimensional protein
    features are plotted in a saved 2D SPIB latent (IB_0, IB_1).
    """
    traj_data = np.asarray(traj_data)
    protein = ts_potential in ("trpcage", "trp_cage", "protein")
    if traj_data.ndim != 2:
        raise ValueError(
            "MTL plot coordinates expect a 2D array, got shape %s" % (traj_data.shape,))
    if traj_data.shape[1] > 2:
        if not latent_path or not os.path.isfile(latent_path):
            raise ValueError(
                "High-dimensional MTL figures require saved latent %s" % latent_path)
        plot_traj = np.load(latent_path)
        if plot_traj.ndim != 2 or plot_traj.shape[1] < 2:
            raise ValueError(
                "SPIB latent for MTL plots must have at least 2 dimensions: %s"
                % (plot_traj.shape,))
        return plot_traj, r"$IB_0$", r"$IB_1$", True
    return traj_data, "x", "y", protein


def _hard_and_active(labels):
    labels = np.asarray(labels)
    hard = labels.argmax(axis=1)
    pop = np.bincount(hard, minlength=labels.shape[1]).astype(float) / max(len(hard), 1)
    active = np.where(pop > 0)[0]
    return hard, active, pop


def _draw_label_panel(ax, traj_data, traj_labels, ts_mask=None, title=None,
                      xlabel="x", ylabel="y"):
    traj_data = np.asarray(traj_data)
    traj_labels = np.asarray(traj_labels)
    _, xedges, yedges, label_map = plot_state_labels.build_label_map(
        traj_data, traj_labels, bins=100)
    state_num = traj_labels.shape[1]
    hard, active, _ = _hard_and_active(traj_labels)
    n_colors = max(state_num, 1)
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    cmap = c.ListedColormap(base[:n_colors])
    im = ax.pcolormesh(
        xedges, yedges, label_map.T, cmap=cmap,
        vmin=-0.5, vmax=state_num - 0.5, shading="auto")
    for s in active:
        mask = hard == s
        if not np.any(mask):
            continue
        ax.text(float(np.mean(traj_data[mask, 0])),
                float(np.mean(traj_data[mask, 1])),
                str(s), ha="center", va="center",
                fontsize=16, fontweight="bold", color="k", zorder=4)
    n_ts = 0
    if ts_mask is not None:
        x_ts, y_ts, n_ts = plot_transition_states._ts_xy(traj_data, ts_mask)
        plot_transition_states._scatter_ts(ax, x_ts, y_ts, s=14, zorder=5)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im, n_ts, len(active)


def _draw_latent_label_panel(ax, z_latent, traj_labels, ts_mask=None, title=None,
                             xlabel=r"$IB_0$", ylabel=r"$IB_1$", fe_beta=1.0,
                             add_ts_legend=False):
    """Tutorial3-style latent labels on the shared SPIB plane (IB_0, IB_1)."""
    z = np.asarray(z_latent)
    traj_labels = np.asarray(traj_labels)
    hard, active, _ = _hard_and_active(traj_labels)
    counts, xedges, yedges, fe = plot_spib_plus.latent_free_energy_grid(
        z, bins=100, fe_beta=fe_beta)
    state_num = traj_labels.shape[1]
    hist_state = np.zeros((state_num,) + counts.shape, dtype=float)
    for i in range(state_num):
        hist_state[i], _, _ = np.histogram2d(
            z[:, 0], z[:, 1], bins=[xedges, yedges], weights=(hard == i))
    label_map = np.argmax(hist_state, axis=0).astype(float)
    label_map[counts == 0] = np.nan
    n_colors = max(state_num, 1)
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    while len(base) < n_colors:
        base = base + base
    cmap = c.ListedColormap(base[:n_colors])
    im = ax.pcolormesh(
        xedges, yedges, label_map.T, cmap=cmap,
        vmin=-0.5, vmax=state_num - 0.5, shading="auto")
    ax.contour(
        fe.T, levels=5,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        colors="black", linewidths=1, linestyles="--")
    for s in active:
        mask = hard == s
        if not np.any(mask):
            continue
        ax.text(float(np.mean(z[mask, 0])),
                float(np.mean(z[mask, 1])),
                str(s), ha="center", va="center",
                fontsize=16, fontweight="bold", color="k", zorder=4)
    n_ts = 0
    if ts_mask is not None:
        n_ts = plot_spib_plus._scatter_latent_transition_states(
            ax, z, ts_mask, add_legend=add_ts_legend)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im, n_ts, len(active)


def _draw_free_energy_panel(ax, traj_data, ts_mask, fe_beta=3.0, vmax=3.0,
                            xlabel="x", ylabel="y"):
    traj_data = np.asarray(traj_data)
    counts, xedges, yedges = np.histogram2d(
        traj_data[:, 0], traj_data[:, 1], bins=100)
    counts = counts.astype(float).copy()
    sampled = counts > 0
    if not np.any(sampled):
        raise ValueError("empty histogram for free-energy plot")
    free_energy = np.full(counts.shape, float(vmax), dtype=float)
    free_energy[sampled] = -np.log(counts[sampled]) / float(fe_beta)
    free_energy[sampled] -= np.nanmin(free_energy[sampled])
    free_energy = np.clip(free_energy, 0.0, float(vmax))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    levels = np.linspace(0.0, float(vmax), 21)
    im = ax.contourf(
        free_energy.T, levels=levels, extent=extent, cmap="jet", extend="neither")
    x_ts, y_ts, n_ts = plot_transition_states._ts_xy(traj_data, ts_mask)
    plot_transition_states._scatter_ts(ax, x_ts, y_ts, s=14, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im, n_ts


def _draw_potential_panel(ax, traj_data, ts_mask, potential="four_well"):
    traj_data = np.asarray(traj_data)
    if potential in ("four_well", "fw"):
        pot_fn = plot_transition_states.potential_fn_four_well_2d
        x_range, y_range = (-1.0, 1.0), (-1.2, 1.1)
        clim = (0.0, 3.0)
    elif potential in ("double_well", "dw"):
        pot_fn = plot_transition_states.potential_fn_double_well_2d
        x_range = (float(np.min(traj_data[:, 0])), float(np.max(traj_data[:, 0])))
        y_range = (float(np.min(traj_data[:, 1])), float(np.max(traj_data[:, 1])))
        clim = (0.0, 3.0)
    else:
        raise ValueError("MTL combined potential plot supports four_well|double_well")
    grid_n = 201
    x_lin = np.linspace(x_range[0], x_range[1], grid_n)
    y_lin = np.linspace(y_range[0], y_range[1], grid_n)
    xx, yy = np.meshgrid(x_lin, y_lin)
    zz = pot_fn(xx, yy)
    im = ax.imshow(
        zz, origin="lower", aspect="auto", cmap="jet",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        vmin=clim[0], vmax=clim[1])
    ax.contour(xx, yy, zz, levels=20, colors="white", linewidths=0.5, alpha=0.8)
    x_ts, y_ts, n_ts = plot_transition_states._ts_xy(traj_data, ts_mask)
    plot_transition_states._scatter_ts(ax, x_ts, y_ts, s=14, zorder=5)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im, n_ts


def _lag_grid(n_dt, panel_width=5.2):
    fig, axes = plt.subplots(1, n_dt, figsize=(panel_width * n_dt, 4.8), squeeze=False)
    return fig, axes[0]


def plot_mtl_labels(traj_data, lag_results, save_path, dpi=150,
                    xlabel="x", ylabel="y", style="config", fe_beta=1.0):
    """One subplot per Δt: learned metastable labels on shared coordinates."""
    style = _normalize_mtl_style(style)
    n_dt = len(lag_results)
    fig, axes = _lag_grid(n_dt)
    for ax, item in zip(axes, lag_results):
        dt = int(item["dt"])
        _, active, _ = _hard_and_active(item["labels"])
        title = "dt=%d, C*=%d" % (dt, len(active))
        if style == "latent":
            im, _, n_states = _draw_latent_label_panel(
                ax, traj_data, item["labels"], title=title,
                xlabel=xlabel, ylabel=ylabel, fe_beta=fe_beta)
        else:
            im, _, n_states = _draw_label_panel(
                ax, traj_data, item["labels"], title=title,
                xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("HSIC-SPIB MTL learned labels", fontsize=14)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_mtl_labels_with_ts(traj_data, lag_results, save_path, dpi=150,
                            xlabel="x", ylabel="y", style="config", fe_beta=1.0):
    style = _normalize_mtl_style(style)
    n_dt = len(lag_results)
    fig, axes = _lag_grid(n_dt)
    for i, (ax, item) in enumerate(zip(axes, lag_results)):
        dt = int(item["dt"])
        _, active, _ = _hard_and_active(item["labels"])
        n_ts = int(np.sum(item.get("ts_mask", [])))
        title = "dt=%d, C*=%d, n_TS=%d" % (dt, len(active), n_ts)
        if style == "latent":
            im, _, n_states = _draw_latent_label_panel(
                ax, traj_data, item["labels"], ts_mask=item.get("ts_mask"),
                title=title, xlabel=xlabel, ylabel=ylabel, fe_beta=fe_beta,
                add_ts_legend=(i == n_dt - 1))
        else:
            im, _, n_states = _draw_label_panel(
                ax, traj_data, item["labels"], ts_mask=item.get("ts_mask"),
                title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("HSIC-SPIB MTL labels + TS", fontsize=14)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_mtl_free_energy_with_ts(traj_data, lag_results, save_path,
                                 fe_beta=3.0, fe_vmax=3.0, dpi=150,
                                 xlabel="x", ylabel="y", style="config"):
    style = _normalize_mtl_style(style)
    n_dt = len(lag_results)
    fig, axes = _lag_grid(n_dt)
    if style == "latent":
        # One shared encoder => one FE grid; panels differ only by TS overlay.
        _, xedges, yedges, fe = plot_spib_plus.latent_free_energy_grid(
            traj_data, bins=100, fe_beta=fe_beta)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = None
        for i, (ax, item) in enumerate(zip(axes, lag_results)):
            dt = int(item["dt"])
            im = ax.contourf(
                fe.T, levels=20, extent=extent, cmap="jet", vmax=fe_vmax)
            n_ts = 0
            if item.get("ts_mask") is not None:
                n_ts = plot_spib_plus._scatter_latent_transition_states(
                    ax, traj_data, item.get("ts_mask"),
                    add_legend=(i == n_dt - 1))
            ax.set_title("dt=%d, n_TS=%d" % (dt, n_ts))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        fig.colorbar(
            im, ax=list(axes), fraction=0.02, pad=0.04,
            label=r"Free Energy ($k_BT$)")
        fig.suptitle("HSIC-SPIB MTL free energy + TS", fontsize=14)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # Shared colorbar is incompatible with tight_layout.
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        for ax, item in zip(axes, lag_results):
            dt = int(item["dt"])
            n_ts = int(np.sum(item.get("ts_mask", [])))
            im, _ = _draw_free_energy_panel(
                ax, traj_data, item.get("ts_mask"), fe_beta=fe_beta, vmax=fe_vmax,
                xlabel=xlabel, ylabel=ylabel)
            ax.set_title("dt=%d, n_TS=%d" % (dt, n_ts))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("HSIC-SPIB MTL free energy + TS", fontsize=14)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path


def plot_mtl_potential_with_ts(traj_data, lag_results, save_path,
                               potential="four_well", dpi=150):
    n_dt = len(lag_results)
    fig, axes = _lag_grid(n_dt)
    for ax, item in zip(axes, lag_results):
        dt = int(item["dt"])
        n_ts = int(np.sum(item.get("ts_mask", [])))
        im, _ = _draw_potential_panel(
            ax, traj_data, item.get("ts_mask"), potential=potential)
        ax.set_title("dt=%d, n_TS=%d" % (dt, n_ts))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("HSIC-SPIB MTL potential + TS", fontsize=14)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_all_mtl_figures(traj_data, lag_results, fig_dir, name_prefix,
                         fe_beta=3.0, potential="four_well", fe_vmax=None, dpi=150,
                         xlabel="x", ylabel="y", style="config"):
    """Write combined-by-lag label / free-energy / potential figures.

    ``lag_results`` is a list of dicts with keys ``dt``, ``labels``, ``ts_mask``.
    ``style='config'`` is the Four-Well / Double-Well recipe (clip FE to 3 kT).
    ``style='latent'`` is the tutorial3 protein recipe: shared (IB_0, IB_1)
    free-energy grid with no default vmax clip, labels over dashed FE contours.
    """
    style = _normalize_mtl_style(style)
    os.makedirs(fig_dir, exist_ok=True)
    if style == "config" and fe_vmax is None:
        fe_vmax = 3.0
    results = []
    p_labels = os.path.join(fig_dir, name_prefix + "_mtl_labels.png")
    results.append(("mtl_labels", plot_mtl_labels(
        traj_data, lag_results, p_labels, dpi=dpi, xlabel=xlabel, ylabel=ylabel,
        style=style, fe_beta=fe_beta)))
    p_labels_ts = os.path.join(fig_dir, name_prefix + "_mtl_labels_with_TS.png")
    results.append(("mtl_labels_with_TS", plot_mtl_labels_with_ts(
        traj_data, lag_results, p_labels_ts, dpi=dpi, xlabel=xlabel, ylabel=ylabel,
        style=style, fe_beta=fe_beta)))
    p_fe = os.path.join(fig_dir, name_prefix + "_mtl_free_energy_with_TS.png")
    results.append(("mtl_free_energy_with_TS", plot_mtl_free_energy_with_ts(
        traj_data, lag_results, p_fe, fe_beta=fe_beta, fe_vmax=fe_vmax, dpi=dpi,
        xlabel=xlabel, ylabel=ylabel, style=style)))
    if potential is not None and str(potential).strip() != "":
        p_pot = os.path.join(fig_dir, name_prefix + "_mtl_potential_with_TS.png")
        results.append(("mtl_potential_with_TS", plot_mtl_potential_with_ts(
            traj_data, lag_results, p_pot, potential=potential, dpi=dpi)))
    return results

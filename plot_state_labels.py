"""
Plot learned SPIB / HSIC-SPIB state labels on the (x, y) plane.

Style follows SPIB_Demo.ipynb ("plot the learned state labels for four well
potential system"): 2D histogram weighted by one-hot labels, then argmax map.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as c


def build_label_map(traj_data, traj_labels, bins=100):
    """
    Build a spatial label map from trajectory coordinates and one-hot labels.

    Parameters
    ----------
    traj_data : (T, >=2) array
    traj_labels : (T, C) one-hot (or soft) labels
    bins : int or sequence
        Histogram bins (same convention as SPIB_Demo).

    Returns
    -------
    hist_counts, xedges, yedges, label_map
    """
    data = np.asarray(traj_data)
    labels = np.asarray(traj_labels)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("traj_data must have shape (T, >=2) for 2D label plots")
    if labels.shape[0] != data.shape[0]:
        raise ValueError("traj_data and traj_labels length mismatch")

    hist_counts, xedges, yedges = np.histogram2d(
        data[:, 0], data[:, 1], bins=bins)

    state_num = labels.shape[1]
    hist_state = np.zeros((state_num,) + hist_counts.shape, dtype=float)
    for i in range(state_num):
        hist_state[i], _, _ = np.histogram2d(
            data[:, 0], data[:, 1], bins=[xedges, yedges], weights=labels[:, i])

    label_map = np.argmax(hist_state, axis=0).astype(float)
    label_map[hist_counts == 0] = np.nan
    return hist_counts, xedges, yedges, label_map


def plot_learned_state_labels(traj_data, traj_labels, save_path,
                              bins=100, title=None, annotate_active=True,
                              dpi=150):
    """
    Plot and save learned state labels (SPIB_Demo style).

    Parameters
    ----------
    traj_data : (T, >=2)
    traj_labels : (T, C) one-hot labels from save_traj_results
    save_path : str
        Output image path (e.g. fig/HSIC_SPIB_labels_dt50.png)
    """
    _, xedges, yedges, label_map = build_label_map(traj_data, traj_labels, bins=bins)
    state_num = traj_labels.shape[1]
    state_labels = np.arange(state_num)

    # active states from hard labels (population > 0)
    hard = traj_labels.argmax(axis=1)
    pop = np.bincount(hard, minlength=state_num).astype(float) / max(len(hard), 1)
    active = np.where(pop > 0)[0]

    fig, ax = plt.subplots(figsize=(9, 6))
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: state_labels[int(round(x))]
                                          if 0 <= int(round(x)) < state_num else "")
    tickz = np.arange(0, state_num)

    n_colors = max(state_num, 1)
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    cMap = c.ListedColormap(base[:n_colors])

    im = ax.pcolormesh(
        xedges, yedges, label_map.T, cmap=cMap,
        vmin=-0.5, vmax=state_num - 0.5, shading="auto")
    cb1 = fig.colorbar(im, ax=ax, format=fmt, ticks=tickz)
    cb1.set_label("state index")

    if annotate_active and active.size > 0:
        # place text at the mass center of each active state (Demo-like annotation)
        for s in active:
            mask = hard == s
            if not np.any(mask):
                continue
            cx = float(np.mean(traj_data[mask, 0]))
            cy = float(np.mean(traj_data[mask, 1]))
            ax.text(cx, cy, str(s),
                    horizontalalignment="center", verticalalignment="center",
                    fontsize=28, fontweight="bold", color="k")

    if title is None:
        title = "Learned state labels (n=%d: %s)" % (
            len(active), ",".join(map(str, active.tolist())))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, active, pop


def save_label_plot_from_files(traj_data_path, labels_path, save_path, **kwargs):
    """Convenience: load npy files then plot."""
    traj_data = np.load(traj_data_path)
    traj_labels = np.load(labels_path)
    return plot_learned_state_labels(traj_data, traj_labels, save_path, **kwargs)

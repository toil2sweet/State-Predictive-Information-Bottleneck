"""
Visualize HSIC-SPIB / SPIB transition-state frames on (x, y).

Three complementary figures (after decoder-K_i TS detection):
  1. Learned labels + TS overlay
  2. Free-energy map + TS (SPIB_Demo panel (b) / paper Fig.5(b) style)
  3. Analytical potential + TS (CTC Fig.2(F) / transition_state.ipynb visual
     style on Four-Well V; all detected TS frames, not top-k)
"""
import os
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as c

import plot_state_labels


FOUR_WELL_CODE_DIR = Path(__file__).resolve().parent / "four-well_SPIB"
DOUBLE_WELL_CODE_DIR = Path(__file__).resolve().parent / "double-well_CTC"


@lru_cache(maxsize=None)
def _load_dir_module(code_dir, module_name):
    """Load a plotting utility from a non-package directory."""
    code_path = Path(code_dir)
    module_path = code_path / (module_name + ".py")
    spec = importlib.util.spec_from_file_location(
        "%s_%s" % (code_path.name, module_name), module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load plotting utility: %s" % module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_four_well_module(module_name):
    """Load a Four-Well utility from its non-package directory."""
    return _load_dir_module(str(FOUR_WELL_CODE_DIR), module_name)


def _load_double_well_module(module_name):
    """Load a double-well CTC utility from its non-package directory."""
    return _load_dir_module(str(DOUBLE_WELL_CODE_DIR), module_name)


def potential_fn_four_well_1d(x):
    """Four-well analytical potential along x (SPIB_Demo)."""
    A, a = 0.6, 80
    B, b = 0.2, 80
    C, c = 0.5, 40
    return (2 * (x ** 8 + A * np.exp(-a * x ** 2)
                 + B * np.exp(-b * (x - 0.5) ** 2)
                 + C * np.exp(-c * (x + 0.5) ** 2))
            + (x ** 2 - 1) ** 2)


def potential_fn_four_well_2d(x, y, k_y=1.0):
    """SPIB four-well U(x, y) = U_1D(x) + k_y y^2 (k_y=1 matches generate_four_well)."""
    return potential_fn_four_well_1d(x) + k_y * (y ** 2)


def potential_fn_double_well_2d(x, y):
    """Double-well potential from double-well_CTC/transition_state.py."""
    return 0.25 * x ** 4 - 3 * x ** 2 + x + 0.5 * y ** 2


def potential_fn_muller_2d(x, y):
    """
    Classic Müller-Brown potential (three metastable basins).

    Same parameters as TS-DAR / Müller & Brown (1979).
    """
    A = np.array([-10.0, -5.0, -17.0 / 2.0, 0.75])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    xbar = np.array([1.0, 0.0, -0.5, -1.0])
    ybar = np.array([0.0, 0.5, 1.5, 1.0])
    v = np.zeros_like(np.asarray(x, dtype=float), dtype=float)
    for i in range(4):
        v = v + A[i] * np.exp(
            a[i] * (x - xbar[i]) ** 2
            + b[i] * (x - xbar[i]) * (y - ybar[i])
            + c[i] * (y - ybar[i]) ** 2
        )
    return v


def _ts_xy(traj_data, ts_mask):
    data = np.asarray(traj_data)
    mask = np.asarray(ts_mask, dtype=bool)
    if mask.shape[0] != data.shape[0]:
        raise ValueError("traj_data and ts_mask length mismatch: %d vs %d" % (
            data.shape[0], mask.shape[0]))
    return data[mask, 0], data[mask, 1], int(mask.sum())


def _scatter_ts(ax, x_ts, y_ts, s=18, zorder=5, linewidths=0.8):
    """CTC-like white face / black edge markers for all TS frames."""
    if len(x_ts) == 0:
        return
    if len(x_ts) > 500:
        s = max(8, s // 2)
        alpha = 0.65
    else:
        alpha = 0.9
    ax.scatter(x_ts, y_ts, s=s, facecolor="white", edgecolor="k",
               linewidths=linewidths, alpha=alpha, zorder=zorder, label="TS")


def plot_labels_with_ts(traj_data, traj_labels, ts_mask, save_path,
                        bins=100, title=None, dpi=150):
    """Learned state-label map with transition-state frames overlaid."""
    traj_data = np.asarray(traj_data)
    traj_labels = np.asarray(traj_labels)
    _, xedges, yedges, label_map = plot_state_labels.build_label_map(
        traj_data, traj_labels, bins=bins)

    state_num = traj_labels.shape[1]
    hard = traj_labels.argmax(axis=1)
    pop = np.bincount(hard, minlength=state_num).astype(float) / max(len(hard), 1)
    active = np.where(pop > 0)[0]

    x_ts, y_ts, n_ts = _ts_xy(traj_data, ts_mask)

    fig, ax = plt.subplots(figsize=(9, 6))
    state_labels = np.arange(state_num)
    fmt = matplotlib.ticker.FuncFormatter(
        lambda x, pos: state_labels[int(round(x))]
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

    for s in active:
        mask = hard == s
        if not np.any(mask):
            continue
        cx = float(np.mean(traj_data[mask, 0]))
        cy = float(np.mean(traj_data[mask, 1]))
        ax.text(cx, cy, str(s), ha="center", va="center",
                fontsize=28, fontweight="bold", color="k", zorder=4)

    _scatter_ts(ax, x_ts, y_ts)

    if title is None:
        title = "Learned labels + TS (n_states=%d, n_TS=%d)" % (len(active), n_ts)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if n_ts > 0:
        ax.legend(loc="best", frameon=True, fontsize=12)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, n_ts


def plot_free_energy_with_ts(traj_data, ts_mask, save_path,
                             bins=100, fe_beta=3.0, title=None, dpi=150,
                             vmax=3.0, n_levels=20, recipe="clipped",
                             xlim=None, ylim=None):
    """
    Free-energy landscape + TS overlay.

    ``recipe="spib_demo"`` follows
    ``four-well_SPIB/plot_free_energy_likeSPIB.py`` / SPIB Fig. 5(b): empty
    bins filled with the smallest nonzero count, ``contourf`` with 5 levels,
    horizontal colorbar on top.

    ``recipe="clipped"`` (default, Müller-style): unsampled bins and F>vmax
    are clipped to vmax so contourf has no white holes.
    """
    if recipe in ("spib_demo", "spib"):
        return _plot_free_energy_with_ts_spib(
            traj_data, ts_mask, save_path, bins=bins, fe_beta=fe_beta,
            title=title, dpi=None, xlim=xlim, ylim=ylim)

    traj_data = np.asarray(traj_data)
    counts, xedges, yedges = np.histogram2d(
        traj_data[:, 0], traj_data[:, 1], bins=bins)
    counts = counts.astype(float).copy()
    sampled = counts > 0
    if not np.any(sampled):
        raise ValueError("empty histogram for free-energy plot")

    free_energy = np.full(counts.shape, float(vmax), dtype=float)
    free_energy[sampled] = -np.log(counts[sampled]) / float(fe_beta)
    free_energy[sampled] -= np.nanmin(free_energy[sampled])
    # Clip high-F sampled bins to vmax so contourf has no white holes
    # (values > max(levels) are otherwise left uncolored).
    free_energy = np.clip(free_energy, 0.0, float(vmax))

    x_ts, y_ts, n_ts = _ts_xy(traj_data, ts_mask)

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    levels = np.linspace(0.0, float(vmax), int(n_levels) + 1)
    h0 = ax.contourf(
        free_energy.T, levels=levels, extent=extent, cmap="jet",
        extend="neither")

    cb = fig.colorbar(h0, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Free Energy", fontsize=14)
    # Tick spacing: ~1 for small vmax, coarser for Müller-like vmax~8
    tick_step = 1.0 if float(vmax) <= 4.5 else max(1.0, round(float(vmax) / 4.0))
    cb.set_ticks(np.arange(0, float(vmax) + 0.1, tick_step))

    _scatter_ts(ax, x_ts, y_ts, s=22, zorder=5)

    if title is None:
        title = "Free energy + TS (n_TS=%d)" % n_ts
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if n_ts > 0:
        ax.legend(loc="best", frameon=True, fontsize=12)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, n_ts


def _plot_free_energy_with_ts_spib(traj_data, ts_mask, save_path,
                                   bins=100, fe_beta=3.0, title=None, dpi=None,
                                   xlim=None, ylim=None):
    """SPIB Fig. 5(b) free-energy map with HSIC-SPIB TS frames overlaid.

    The background is the same as
    ``four-well_SPIB/plot_free_energy_likeSPIB.py::_draw_free_energy``:
    histogram extent, 5-level jet ``contourf``, top colorbar. Axis limits
    stay on that extent unless the caller passes ``xlim`` / ``ylim``.
    """
    spib_fe = _load_four_well_module("plot_free_energy_likeSPIB")

    traj_data = np.asarray(traj_data)
    x_ts, y_ts, n_ts = _ts_xy(traj_data, ts_mask)
    free_energy, xedges, yedges = spib_fe.empirical_free_energy(
        traj_data, bins=bins, beta=fe_beta)
    if dpi is None:
        dpi = spib_fe.DPI

    with matplotlib.rc_context():
        spib_fe._configure_matplotlib()
        fig, ax = plt.subplots(figsize=spib_fe.FREE_ENERGY_FIGSIZE)
        spib_fe._draw_free_energy(fig, ax, free_energy, xedges, yedges)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        _scatter_ts(
            ax, x_ts, y_ts, s=50 if n_ts <= 80 else 22,
            zorder=3, linewidths=1.0)
        if title:
            ax.set_title(title)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return save_path, n_ts


def _plot_four_well_potential_with_ts(
        traj_data, ts_mask, save_path, x_ts, y_ts, n_ts,
        grid_n=None, title=None, dpi=None,
        clim=None, x_range=None, y_range=None):
    """CTC-style four-well U(x, y) with HSIC-SPIB TS frames overlaid.

    The background is
    ``four-well_SPIB/plot_energy_landscape_likeCTC.py`` with that script's
    default window, clim, grid, and colorbar.
    """
    ctc_land = _load_four_well_module("plot_energy_landscape_likeCTC")

    if x_range is None:
        x_range = ctc_land.DEFAULT_XLIM
    if y_range is None:
        y_range = ctc_land.DEFAULT_YLIM
    if clim is None:
        clim = ctc_land.FOUR_WELL_CLIM
    if grid_n is None:
        grid_n = ctc_land.CTC_GRID_N
    figsize, _ = ctc_land.window_figsize(x_range, y_range)
    dpi = ctc_land.DPI

    with matplotlib.rc_context():
        ctc_land._configure_matplotlib()
        fig, ax = plt.subplots(figsize=figsize)
        ctc_land.draw_four_well_potential_like_ctc(
            fig, ax, x_range, y_range, grid_n=grid_n, clim=clim)
        _scatter_ts(
            ax, x_ts, y_ts, s=50 if n_ts <= 80 else 22,
            zorder=3, linewidths=1.0)
        if title:
            ax.set_title(title)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return save_path, n_ts


def _plot_double_well_potential_with_ts(
        traj_data, ts_mask, save_path, x_ts, y_ts, n_ts,
        grid_n=None, title=None, dpi=None,
        clim=None, x_range=None, y_range=None):
    """CTC-style double-well V(x, y) with HSIC-SPIB TS frames overlaid.

    The background is
    ``double-well_CTC/plot_energy_landscape_likeCTC.py`` with that script's
    default window, clim, grid, and colorbar.
    """
    ctc_land = _load_double_well_module("plot_energy_landscape_likeCTC")

    if x_range is None:
        x_range = ctc_land.DEFAULT_XLIM
    if y_range is None:
        y_range = ctc_land.DEFAULT_YLIM
    if clim is None:
        clim = ctc_land.DEFAULT_CLIM
    if grid_n is None:
        grid_n = ctc_land.GRID_N
    dpi = ctc_land.DPI

    with matplotlib.rc_context():
        ctc_land._configure_matplotlib()
        fig, ax = plt.subplots(figsize=ctc_land.FIGSIZE)
        ctc_land.draw_double_well_potential_like_ctc(
            fig, ax, x_range, y_range, grid_n=grid_n, clim=clim)
        _scatter_ts(
            ax, x_ts, y_ts, s=50 if n_ts <= 80 else 22,
            zorder=3, linewidths=1.0)
        if title:
            ax.set_title(title)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    return save_path, n_ts


def plot_potential_with_ts(traj_data, ts_mask, save_path,
                           potential="four_well", grid_n=401,
                           title=None, dpi=150,
                           clim=None, x_range=None, y_range=None):
    """
    Analytical potential heatmap + all detected TS frames.

    Four-well uses ``four-well_SPIB/plot_energy_landscape_likeCTC.py``.
    Double-well uses ``double-well_CTC/plot_energy_landscape_likeCTC.py``.
    Müller keeps the previous imshow + white-contour recipe.
    """
    traj_data = np.asarray(traj_data)
    x_ts, y_ts, n_ts = _ts_xy(traj_data, ts_mask)

    if potential in ("four_well", "fw"):
        return _plot_four_well_potential_with_ts(
            traj_data, ts_mask, save_path, x_ts, y_ts, n_ts,
            grid_n=grid_n, title=title, dpi=dpi,
            clim=clim, x_range=x_range, y_range=y_range)

    if potential in ("double_well", "dw"):
        return _plot_double_well_potential_with_ts(
            traj_data, ts_mask, save_path, x_ts, y_ts, n_ts,
            grid_n=grid_n, title=title, dpi=dpi,
            clim=clim, x_range=x_range, y_range=y_range)

    if potential in ("muller", "muller_brown", "mb"):
        # TS-DAR / paper domain for Müller Brownian trajectory (T≈0.9)
        if x_range is None:
            x_range = (-1.5, 1.2)
        if y_range is None:
            y_range = (-0.3, 2.1)
        pot_fn = potential_fn_muller_2d
        pot_name = "Müller V(x,y)"
        n_contour = 20
        fig_size = [7.0, 6.0]
        aspect = "equal"
        # clim filled after evaluating V (shift to min 0, clip high barrier)
    else:
        raise ValueError(
            "Unknown potential=%r (use four_well|double_well|muller)" % potential)

    x_lin = np.linspace(x_range[0], x_range[1], grid_n)
    y_lin = np.linspace(y_range[0], y_range[1], grid_n)
    # CTC builds potential with shape (ny, nx) via y[...,None] and x[None,...]
    XX, YY = np.meshgrid(x_lin, y_lin)
    ZZ = pot_fn(XX, YY)

    if potential in ("muller", "muller_brown", "mb"):
        ZZ = ZZ - np.nanmin(ZZ)
        if clim is None:
            clim = (0.0, 10.0)
        ZZ = np.ma.masked_greater(ZZ, clim[1])

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        ZZ, origin="lower", aspect=aspect, cmap="jet",
        extent=[x_lin[0], x_lin[-1], y_lin[0], y_lin[-1]],
        vmin=clim[0], vmax=clim[1])
    cbar = fig.colorbar(im, ax=ax, aspect=22, pad=0.02)
    cbar.set_label("Energy")
    ax.contour(
        XX, YY, np.ma.filled(ZZ, np.nanmax(ZZ)), levels=n_contour, colors="white",
        linestyles="solid", alpha=1.0, linewidths=0.7, zorder=2)

    # CTC: white face / black edge, s=50; keep all TS (not top-k)
    _scatter_ts(ax, x_ts, y_ts, s=50 if n_ts <= 80 else 22,
                zorder=3, linewidths=1.0)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    if title is None:
        title = "%s + TS (n_TS=%d, CTC-style)" % (pot_name, n_ts)
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if n_ts > 0:
        ax.legend(loc="best", frameon=True, fontsize=12)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return save_path, n_ts


def plot_all_ts_figures(traj_data, traj_labels, ts_mask, fig_dir, name_prefix,
                        fe_beta=3.0, potential="four_well", dpi=150,
                        fe_vmax=None, title_prefix=None):
    """
    Write TS-annotated figures under fig_dir:
      labels+TS, free-energy+TS, and (if potential is set) analytical V+TS.

    ``potential`` should be ``four_well``, ``double_well``, or ``muller``.
    If None/empty, the potential+TS figure is skipped.

    ``*_free_energy_with_TS.png`` uses empirical F=-log(P)/fe_beta.
    Four-well follows ``four-well_SPIB/plot_free_energy_likeSPIB.py``
    (Fig. 5(b)).
    ``*_potential_with_TS.png`` uses the analytical V; four-well follows
    ``four-well_SPIB/plot_energy_landscape_likeCTC.py`` and double-well
    follows ``double-well_CTC/plot_energy_landscape_likeCTC.py``.

    Returns list of (kind, path, n_ts).
    """
    os.makedirs(fig_dir, exist_ok=True)
    results = []
    four_well = potential in ("four_well", "fw")
    double_well = potential in ("double_well", "dw")
    ctc_potential = four_well or double_well

    p1 = os.path.join(fig_dir, name_prefix + "_labels_with_TS.png")
    labels_title = None
    if title_prefix:
        labels_title = "%s: learned labels + TS (n_TS=%d)" % (
            title_prefix, int(np.sum(ts_mask)))
    path, n_ts = plot_labels_with_ts(
        traj_data, traj_labels, ts_mask, p1, dpi=dpi,
        title=labels_title or "Learned labels + TS (n_TS=%d)" % int(np.sum(ts_mask)))
    results.append(("labels_with_TS", path, n_ts))

    if fe_vmax is None:
        if potential in ("muller", "muller_brown", "mb"):
            # Müller BD uses T≈0.9; pair with fe_beta≈0.9 and a wider ceiling
            # so barriers remain visible (four/double-well demos use vmax=3).
            fe_vmax = 8.0
        else:
            fe_vmax = 3.0

    p2 = os.path.join(fig_dir, name_prefix + "_free_energy_with_TS.png")
    free_energy_title = None
    # Four-well SPIB Fig. 5(b) already labels the colorbar "Free Energy";
    # keep that panel free of the CTC-style title_prefix.
    if title_prefix and not four_well:
        free_energy_title = "%s: free energy + TS (n_TS=%d)" % (
            title_prefix, int(np.sum(ts_mask)))
    elif not four_well:
        free_energy_title = "Free energy + TS (n_TS=%d)" % int(np.sum(ts_mask))
    path, n_ts = plot_free_energy_with_ts(
        traj_data, ts_mask, p2, fe_beta=fe_beta, dpi=dpi, vmax=fe_vmax,
        recipe="spib_demo" if four_well else "clipped",
        title=free_energy_title)
    results.append(("free_energy_with_TS", path, n_ts))

    if potential is not None and str(potential).strip() != "":
        p3 = os.path.join(fig_dir, name_prefix + "_potential_with_TS.png")
        potential_title = None
        if title_prefix:
            potential_title = "%s: analytical potential + TS (n_TS=%d)" % (
                title_prefix, int(np.sum(ts_mask)))
        elif not ctc_potential:
            potential_title = "Potential + TS (n_TS=%d)" % int(np.sum(ts_mask))
        path, n_ts = plot_potential_with_ts(
            traj_data, ts_mask, p3, potential=potential, dpi=dpi,
            title=potential_title)
        results.append(("potential_with_TS", path, n_ts))

    return results

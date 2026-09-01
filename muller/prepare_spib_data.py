"""
Prepare HSIC-SPIB / HSIC-SPIB+ inputs from the TS-DAR Müller-potential trajectory.

Canonical Müller labels for training: xy_kmeans, K=20 (slanted three-basin geometry).
Other methods remain available for ablation:
  x_bins      — equal-width bins along x (legacy; poor for diagonal basins)
  tica_kmeans — TICA then MiniBatchKMeans (protein-style; little gain in 2D)

Source: ts-dar/data/muller/muller.npy  (Brownian dynamics, T ≈ 0.9)

Examples:
  python muller/prepare_spib_data.py
  python muller/prepare_spib_data.py --method xy_kmeans --n-clusters 20
"""
from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE = REPO_ROOT / "ts-dar" / "data" / "muller" / "muller.npy"

STATE_NUM = 6
EPS = 1e-3
DEFAULT_N_CLUSTERS = 20
DEFAULT_TICA_LAG = 10
DEFAULT_TICA_DIM = 2


def load_traj_data(source=DEFAULT_SOURCE):
    traj = np.load(source)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("Expected Müller traj shape [N, 2(+)], got %s" % (traj.shape,))
    return traj[:, :2].astype(np.float32)


def labels_to_one_hot(cluster_ids, n_states=None):
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    if n_states is None:
        n_states = int(cluster_ids.max()) + 1
    init_label = np.zeros((cluster_ids.shape[0], n_states), dtype=np.float32)
    init_label[np.arange(len(cluster_ids)), cluster_ids] = 1.0
    return init_label


def make_init_label_x_bins(traj_data, state_num=STATE_NUM, index=0, eps=EPS):
    """Equal-width bins along one coordinate (legacy SPIB_Demo recipe)."""
    x_max = float(traj_data[:, index].max()) + 0.01
    x_min = float(traj_data[:, index].min()) - 0.01
    x_det = (x_max - x_min + 2 * eps) / state_num
    x_list = np.array([(x_min - eps + n * x_det) for n in range(state_num + 1)])

    init_label = np.zeros((traj_data.shape[0], state_num), dtype=np.float32)
    for j in range(state_num):
        indices = (traj_data[:, index] > x_list[j]) & (
            traj_data[:, index] <= x_list[j + 1]
        )
        init_label[indices, j] = 1.0

    return init_label, {"edges": x_list.astype(np.float32)}


def make_init_label_xy_kmeans(traj_data, n_clusters=DEFAULT_N_CLUSTERS, seed=0):
    """Overcomplete clustering directly in (x, y) — suited to diagonal basins."""
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
        batch_size=min(4096, max(1024, len(traj_data) // 10)),
    )
    cluster_ids = km.fit_predict(traj_data)
    init_label = labels_to_one_hot(cluster_ids, n_states=n_clusters)
    meta = {
        "cluster_centers": km.cluster_centers_.astype(np.float32),
        "cluster_ids": cluster_ids.astype(np.int32),
    }
    return init_label, meta


def _tica_numpy(traj_data, lag=DEFAULT_TICA_LAG, dim=DEFAULT_TICA_DIM, eps=1e-8):
    """
    Minimal TICA via time-lagged covariances (no deeptime dependency).
    Returns projection of shape (N, dim).
    """
    X = traj_data.astype(np.float64)
    n, d = X.shape
    if lag <= 0 or lag >= n:
        raise ValueError("tica lag must satisfy 0 < lag < N (got lag=%d, N=%d)" % (lag, n))

    X0 = X[:-lag]
    Xt = X[lag:]
    mean0 = X0.mean(axis=0)
    meant = Xt.mean(axis=0)
    X0c = X0 - mean0
    Xtc = Xt - meant

    C0 = (X0c.T @ X0c) / max(len(X0c) - 1, 1)
    C_lag = (X0c.T @ Xtc) / max(len(X0c) - 1, 1)
    # Symmetrize lag cov for reversible dynamics
    C_lag = 0.5 * (C_lag + C_lag.T)

    # Whiten with C0^{-1/2}
    evals0, evecs0 = np.linalg.eigh(C0)
    evals0 = np.clip(evals0, eps, None)
    W = evecs0 * (1.0 / np.sqrt(evals0))[None, :]
    A = W.T @ C_lag @ W
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    n_comp = min(dim, d)
    V = W @ evecs[:, :n_comp]
    proj = ((X - mean0) @ V).astype(np.float32)
    return proj, {
        "tica_eigenvalues": evals[:n_comp].astype(np.float32),
        "tica_mean": mean0.astype(np.float32),
        "tica_components": V.astype(np.float32),
        "tica_lag": int(lag),
        "tica_dim": int(n_comp),
    }


def make_init_label_tica_kmeans(
    traj_data,
    n_clusters=DEFAULT_N_CLUSTERS,
    tica_lag=DEFAULT_TICA_LAG,
    tica_dim=DEFAULT_TICA_DIM,
    seed=0,
):
    """TICA on (x, y) then MiniBatchKMeans — mirrors 2024 protein init protocol."""
    try:
        from deeptime.decomposition import TICA

        tica = TICA(dim=min(tica_dim, traj_data.shape[1]), lagtime=tica_lag)
        tica.fit(traj_data)
        model = tica.fetch_model()
        proj = model.transform(traj_data).astype(np.float32)
        tica_meta = {
            "tica_backend": "deeptime",
            "tica_lag": int(tica_lag),
            "tica_dim": int(proj.shape[1]),
        }
    except ImportError:
        proj, tica_meta = _tica_numpy(traj_data, lag=tica_lag, dim=tica_dim)
        tica_meta["tica_backend"] = "numpy"

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
        batch_size=min(4096, max(1024, len(traj_data) // 10)),
    )
    cluster_ids = km.fit_predict(proj)
    init_label = labels_to_one_hot(cluster_ids, n_states=n_clusters)
    meta = dict(tica_meta)
    meta["tica_projection"] = proj
    meta["cluster_centers"] = km.cluster_centers_.astype(np.float32)
    meta["cluster_ids"] = cluster_ids.astype(np.int32)
    return init_label, meta


def main(
    method="xy_kmeans",
    state_num=STATE_NUM,
    n_clusters=DEFAULT_N_CLUSTERS,
    tica_lag=DEFAULT_TICA_LAG,
    tica_dim=DEFAULT_TICA_DIM,
    seed=0,
    source=DEFAULT_SOURCE,
):
    traj_data = load_traj_data(source)
    method = method.lower().strip()

    if method == "x_bins":
        init_label, meta = make_init_label_x_bins(traj_data, state_num=state_num)
        label_path = SCRIPT_DIR / ("init_label%d.npy" % state_num)
        edges_path = SCRIPT_DIR / ("init_label%d_x_edges.npy" % state_num)
        np.save(edges_path, meta["edges"])
        extra_msg = "Saved %s" % edges_path
    elif method == "xy_kmeans":
        init_label, meta = make_init_label_xy_kmeans(
            traj_data, n_clusters=n_clusters, seed=seed
        )
        label_path = SCRIPT_DIR / ("init_label_kmeans%d.npy" % n_clusters)
        centers_path = SCRIPT_DIR / ("init_label_kmeans%d_centers.npy" % n_clusters)
        np.save(centers_path, meta["cluster_centers"])
        extra_msg = "Saved %s" % centers_path
    elif method == "tica_kmeans":
        init_label, meta = make_init_label_tica_kmeans(
            traj_data,
            n_clusters=n_clusters,
            tica_lag=tica_lag,
            tica_dim=tica_dim,
            seed=seed,
        )
        label_path = SCRIPT_DIR / (
            "init_label_tica_kmeans%d_lag%d.npy" % (n_clusters, tica_lag)
        )
        proj_path = SCRIPT_DIR / (
            "tica_projection_lag%d.npy" % tica_lag
        )
        np.save(proj_path, meta["tica_projection"])
        centers_path = SCRIPT_DIR / (
            "init_label_tica_kmeans%d_lag%d_centers.npy" % (n_clusters, tica_lag)
        )
        np.save(centers_path, meta["cluster_centers"])
        extra_msg = "Saved %s and %s (backend=%s)" % (
            proj_path, centers_path, meta.get("tica_backend")
        )
    else:
        raise ValueError(
            "Unknown method %r; choose x_bins | xy_kmeans | tica_kmeans" % method
        )

    traj_path = SCRIPT_DIR / "traj_data.npy"
    np.save(traj_path, traj_data)
    np.save(label_path, init_label)

    meta_path = SCRIPT_DIR / "prepare_meta.json"
    meta_json = {
        "method": method,
        "source": str(source),
        "traj_shape": list(traj_data.shape),
        "label_shape": list(init_label.shape),
        "label_path": str(label_path.name),
        "n_clusters": int(n_clusters) if method != "x_bins" else int(state_num),
        "seed": int(seed),
    }
    if method == "tica_kmeans":
        meta_json["tica_lag"] = int(tica_lag)
        meta_json["tica_dim"] = int(tica_dim)
        meta_json["tica_backend"] = meta.get("tica_backend")
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)

    population = init_label.sum(axis=0)
    n_frames = len(traj_data)
    n_active = int((population > 0).sum())
    print("Source: %s" % source)
    print("Method: %s" % method)
    print("Saved %s  shape=%s" % (traj_path, traj_data.shape))
    print("Saved %s  shape=%s" % (label_path, init_label.shape))
    print(extra_msg)
    print("Saved %s" % meta_path)
    print("Initial state populations (%d states, %d non-empty):" % (
        init_label.shape[1], n_active))
    for j, pop in enumerate(population):
        if pop > 0:
            print(
                "  state %d: pop=%.0f (%.2f%%)"
                % (j, pop, 100.0 * pop / n_frames)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Müller SPIB/HSIC-SPIB+ initial labels"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="xy_kmeans",
        choices=["x_bins", "xy_kmeans", "tica_kmeans"],
        help="Initial label strategy (default: xy_kmeans)",
    )
    parser.add_argument(
        "--state-num",
        type=int,
        default=STATE_NUM,
        help="Number of x-bins when --method x_bins",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="K for xy_kmeans / tica_kmeans (default: 20)",
    )
    parser.add_argument("--tica-lag", type=int, default=DEFAULT_TICA_LAG)
    parser.add_argument("--tica-dim", type=int, default=DEFAULT_TICA_DIM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    main(
        method=args.method,
        state_num=args.state_num,
        n_clusters=args.n_clusters,
        tica_lag=args.tica_lag,
        tica_dim=args.tica_dim,
        seed=args.seed,
        source=args.source,
    )

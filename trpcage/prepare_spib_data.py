"""
Prepare HSIC-SPIB+ inputs for Trp-cage (DESRES pairwise-distance features).

Aligned with spib_msm/examples/tutorial3_trpcage.ipynb:
  1. Download cached trajectory (Google Drive via gdown)
  2. Feature mean/std for DataNormalize
  3. TICA + MiniBatchKMeans overcomplete initial labels

Source (tutorial3):
  gdown id 1X-Cf9MIGhWYPpCXcJ2ahKUOr2NcbAozs
  -> trpcage_153pairwise_distances_closeheavy_0.2ns_traj.npy  (~639 MB)

Examples:
  python trpcage/prepare_spib_data.py
  python trpcage/prepare_spib_data.py --n-clusters 200 --tica-lag 200 --tica-dim 3
  python trpcage/prepare_spib_data.py --skip-download   # if npy already present
"""
from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

GDRIVE_ID = "1X-Cf9MIGhWYPpCXcJ2ahKUOr2NcbAozs"
RAW_NAME = "trpcage_153pairwise_distances_closeheavy_0.2ns_traj.npy"

DEFAULT_TICA_LAG = 200
DEFAULT_TICA_DIM = 3
DEFAULT_N_CLUSTERS = 200


def download_trajectory(dest_dir=SCRIPT_DIR, force=False):
    dest = Path(dest_dir) / RAW_NAME
    if dest.is_file() and not force:
        print("Found existing trajectory: %s (%.1f MB)" % (
            dest, dest.stat().st_size / 1e6))
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("gdown is required to download Trp-cage data. Install with: pip install gdown")
        sys.exit(1)

    url = "https://drive.google.com/uc?id=%s" % GDRIVE_ID
    print("Downloading Trp-cage trajectory (~639 MB) ...")
    print("  url: %s" % url)
    print("  -> %s" % dest)
    gdown.download(url, str(dest), quiet=False)
    if not dest.is_file():
        raise RuntimeError("Download failed: %s not found" % dest)
    print("Download complete: %.1f MB" % (dest.stat().st_size / 1e6))
    return dest


def load_traj(path):
    traj = np.load(path)
    if traj.ndim != 2:
        raise ValueError("Expected 2D traj [N, F], got %s" % (traj.shape,))
    return traj.astype(np.float32, copy=False)


def compute_normalize_stats(traj):
    mean = traj.mean(axis=0).astype(np.float32)
    std = traj.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def _tica_numpy(traj, lag, dim, eps=1e-8):
    """Minimal reversible TICA (no deeptime dependency)."""
    X = traj.astype(np.float64, copy=False)
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
    C_lag = 0.5 * (C_lag + C_lag.T)

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
        "tica_backend": "numpy",
        "tica_eigenvalues": evals[:n_comp].astype(np.float32),
        "tica_mean": mean0.astype(np.float32),
        "tica_components": V.astype(np.float32),
    }


def run_tica(traj, lag, dim):
    try:
        from deeptime.decomposition import TICA

        tica = TICA(dim=min(dim, traj.shape[1]), lagtime=lag)
        tica.fit(traj)
        model = tica.fetch_model()
        proj = np.asarray(model.transform(traj), dtype=np.float32)
        return proj, {"tica_backend": "deeptime"}
    except ImportError:
        return _tica_numpy(traj, lag=lag, dim=dim)


def labels_to_one_hot(cluster_ids, n_states):
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    init_label = np.zeros((len(cluster_ids), n_states), dtype=np.float32)
    init_label[np.arange(len(cluster_ids)), cluster_ids] = 1.0
    return init_label


def make_tica_kmeans_labels(traj, n_clusters, tica_lag, tica_dim, seed=0):
    print("Running TICA (lag=%d, dim=%d) on shape %s ..." % (
        tica_lag, tica_dim, traj.shape))
    proj, tica_meta = run_tica(traj, lag=tica_lag, dim=tica_dim)
    print("TICA done: projection shape=%s backend=%s" % (
        proj.shape, tica_meta.get("tica_backend")))

    print("MiniBatchKMeans n_clusters=%d ..." % n_clusters)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=5,
        random_state=seed,
        batch_size=4096,
    )
    cluster_ids = km.fit_predict(proj)
    init_label = labels_to_one_hot(cluster_ids, n_states=n_clusters)
    meta = dict(tica_meta)
    meta.update({
        "tica_lag": int(tica_lag),
        "tica_dim": int(proj.shape[1]),
        "n_clusters": int(n_clusters),
        "cluster_centers": km.cluster_centers_.astype(np.float32),
        "cluster_ids": cluster_ids.astype(np.int32),
        "tica_projection": proj,
    })
    return init_label, meta


def main(
    skip_download=False,
    force_download=False,
    n_clusters=DEFAULT_N_CLUSTERS,
    tica_lag=DEFAULT_TICA_LAG,
    tica_dim=DEFAULT_TICA_DIM,
    seed=0,
):
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    if skip_download:
        raw_path = SCRIPT_DIR / RAW_NAME
        if not raw_path.is_file():
            # also accept already-copied traj_data.npy
            alt = SCRIPT_DIR / "traj_data.npy"
            if alt.is_file():
                raw_path = alt
            else:
                raise FileNotFoundError(
                    "No trajectory at %s; run without --skip-download" % raw_path)
    else:
        raw_path = download_trajectory(SCRIPT_DIR, force=force_download)

    traj = load_traj(raw_path)
    print("Loaded traj shape=%s dtype=%s" % (traj.shape, traj.dtype))

    # Canonical training path (copy/symlink-like save for config convenience)
    traj_path = SCRIPT_DIR / "traj_data.npy"
    if traj_path.resolve() != raw_path.resolve():
        np.save(traj_path, traj)
        print("Saved %s" % traj_path)
    else:
        print("Using %s as traj_data.npy" % traj_path)

    mean, std = compute_normalize_stats(traj)
    mean_path = SCRIPT_DIR / "data_mean.npy"
    std_path = SCRIPT_DIR / "data_std.npy"
    np.save(mean_path, mean)
    np.save(std_path, std)
    print("Saved %s / %s" % (mean_path, std_path))

    init_label, meta = make_tica_kmeans_labels(
        traj, n_clusters=n_clusters, tica_lag=tica_lag, tica_dim=tica_dim, seed=seed)

    label_path = SCRIPT_DIR / (
        "init_label_tica_kmeans%d_lag%d.npy" % (n_clusters, tica_lag))
    proj_path = SCRIPT_DIR / ("tica_projection_lag%d.npy" % tica_lag)
    centers_path = SCRIPT_DIR / (
        "init_label_tica_kmeans%d_lag%d_centers.npy" % (n_clusters, tica_lag))

    np.save(label_path, init_label)
    np.save(proj_path, meta["tica_projection"])
    np.save(centers_path, meta["cluster_centers"])

    meta_path = SCRIPT_DIR / "prepare_meta.json"
    meta_json = {
        "raw_name": RAW_NAME,
        "gdrive_id": GDRIVE_ID,
        "traj_shape": list(traj.shape),
        "label_shape": list(init_label.shape),
        "label_path": label_path.name,
        "data_mean": mean_path.name,
        "data_std": std_path.name,
        "tica_lag": int(tica_lag),
        "tica_dim": int(meta["tica_dim"]),
        "tica_backend": meta.get("tica_backend"),
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "frame_dt_ns": 0.2,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)

    pop = init_label.sum(axis=0)
    n_active = int((pop > 0).sum())
    print("Saved %s shape=%s" % (label_path, init_label.shape))
    print("Saved %s / %s" % (proj_path, centers_path))
    print("Saved %s" % meta_path)
    print("Initial labels: %d clusters, %d non-empty" % (init_label.shape[1], n_active))
    print("Top-10 populations (fraction): %s" % (
        (np.sort(pop)[::-1][:10] / len(traj)).round(4).tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Trp-cage HSIC-SPIB+ trajectory, normalize stats, TICA+kmeans labels"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Do not download; use existing npy under trpcage/")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download even if file exists")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--tica-lag", type=int, default=DEFAULT_TICA_LAG)
    parser.add_argument("--tica-dim", type=int, default=DEFAULT_TICA_DIM)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        skip_download=args.skip_download,
        force_download=args.force_download,
        n_clusters=args.n_clusters,
        tica_lag=args.tica_lag,
        tica_dim=args.tica_dim,
        seed=args.seed,
    )

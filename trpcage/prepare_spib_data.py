"""
Prepare 2024 SPIB / HSIC-SPIB+ inputs for Trp-cage from the local DESRES
Anton trajectory (Lindorff-Larsen et al., Science 2011).

Aligned with spib_msm/examples/tutorial3_trpcage.ipynb and the 2024 JCTC
MSM protocol:
  1. Protein DCD + Maestro topology (272 atoms)
  2. 153 minimal residue-residue closest-heavy distances (|i-j| >= 3)
  3. Feature mean/std for DataNormalize
  4. TICA(lag=200, dim=3) + MiniBatchKMeans(K=200) overcomplete labels

Default DESRES root (Science SI page 19, 290 K):
  ~/Desktop/DESRES/science2011 Anton Trajectories/Trp-cage at 290 K (page 19)

Frame spacing in that distribution is 200 ps = 0.2 ns/frame, matching
tutorial3. Expected length is 1,044,000 frames (208.8 us).

Examples:
  python trpcage/prepare_spib_data.py
  python trpcage/prepare_spib_data.py --max-frames 2000
  python trpcage/prepare_spib_data.py --features-npy path/to/precomputed.npy
"""
from __future__ import print_function

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DESRES_ROOT = Path(
    "/Users/lidepeng/Desktop/DESRES/science2011 Anton Trajectories/"
    "Trp-cage at 290 K (page 19)"
)
PROTEIN_REL = Path("DESRES-Trajectory_2JOF-0-protein") / "2JOF-0-protein"

DEFAULT_TICA_LAG = 200
DEFAULT_TICA_DIM = 3
DEFAULT_N_CLUSTERS = 200
EXPECTED_N_RESIDUES = 20
EXPECTED_N_CONTACTS = 153
FRAME_DT_NS = 0.2
MIN_SEQ_SEP = 3

def _mae_tokens(line):
    """Split a Maestro table row into tokens (quoted strings may contain spaces)."""
    tokens = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            j = line.find('"', i + 1)
            if j < 0:
                tokens.append(line[i + 1:].strip())
                break
            tokens.append(line[i + 1:j])
            i = j + 1
            continue
        j = i + 1
        while j < len(line) and not line[j].isspace():
            j += 1
        tokens.append(line[i:j])
        i = j
    return tokens


def _progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        if desc:
            print(desc)
        return iterable


def parse_mae_atoms(mae_path):
    """Parse PDB-like atom fields from a DESRES Maestro topology."""
    text = Path(mae_path).read_text(errors="replace")
    marker = "m_atom["
    start = text.find(marker)
    if start < 0:
        raise ValueError("No m_atom block in %s" % mae_path)
    block = text[start:]
    atoms = []
    in_rows = False
    for line in block.splitlines():
        stripped = line.strip()
        if not in_rows:
            if stripped == ":::":
                in_rows = True
            continue
        if stripped.startswith("}"):
            break
        if not stripped:
            continue
        tokens = _mae_tokens(stripped)
        # index, name, resname, chain, segment, resid, x, y, z, vx, vy, vz, Z, ...
        if len(tokens) < 13:
            continue
        atoms.append({
            "name": tokens[1].strip(),
            "resname": tokens[2].strip(),
            "resid": int(tokens[5]),
            "atomic_number": int(float(tokens[12])),
        })
    if not atoms:
        raise ValueError("Parsed 0 atoms from %s" % mae_path)
    return atoms


def residue_heavy_index_groups(atoms, min_seq_sep=MIN_SEQ_SEP):
    """Return heavy-atom index groups and residue pairs with |i-j| >= min_seq_sep."""
    resids = []
    seen = set()
    for atom in atoms:
        rid = atom["resid"]
        if rid not in seen:
            seen.add(rid)
            resids.append(rid)
    groups = []
    for rid in resids:
        heavy = [i for i, atom in enumerate(atoms)
                 if atom["resid"] == rid and atom["atomic_number"] > 1]
        if not heavy:
            heavy = [i for i, atom in enumerate(atoms) if atom["resid"] == rid]
        groups.append(np.asarray(heavy, dtype=np.int64))
    pairs = [(i, j)
             for i in range(len(resids))
             for j in range(i + min_seq_sep, len(resids))]
    return groups, pairs, resids


def n_minimal_contacts(n_residues, min_seq_sep=MIN_SEQ_SEP):
    n_pairs = n_residues * (n_residues - 1) // 2
    n_seq1 = max(n_residues - 1, 0)
    n_seq2 = max(n_residues - 2, 0) if min_seq_sep >= 3 else 0
    if min_seq_sep == 3:
        return n_pairs - n_seq1 - n_seq2
    raise ValueError("n_minimal_contacts only implemented for min_seq_sep=3")


def _read_fortran_record(handle, endian="<"):
    size_bytes = handle.read(4)
    if not size_bytes:
        return None
    if len(size_bytes) < 4:
        raise EOFError("truncated Fortran record marker")
    n_bytes = struct.unpack(endian + "i", size_bytes)[0]
    payload = handle.read(n_bytes)
    if len(payload) != n_bytes:
        raise EOFError("truncated Fortran record payload")
    end = handle.read(4)
    if len(end) != 4 or struct.unpack(endian + "i", end)[0] != n_bytes:
        raise ValueError("Fortran record size mismatch")
    return payload


def iter_dcd_xyz(path):
    """Yield (n_atom, 3) float32 coordinate frames from a CHARMM/NAMD DCD."""
    with open(path, "rb") as handle:
        rec1 = _read_fortran_record(handle)
        if rec1 is None or rec1[:4] != b"CORD":
            raise ValueError("%s is not a CORD DCD" % path)
        _read_fortran_record(handle)  # title
        rec3 = _read_fortran_record(handle)
        n_atom = struct.unpack("<i", rec3[:4])[0]
        coord_bytes = 4 * n_atom
        while True:
            rec = _read_fortran_record(handle)
            if rec is None:
                break
            if len(rec) == 48:
                x_rec = _read_fortran_record(handle)
                y_rec = _read_fortran_record(handle)
                z_rec = _read_fortran_record(handle)
            elif len(rec) == coord_bytes:
                x_rec, y_rec, z_rec = rec, _read_fortran_record(handle), _read_fortran_record(handle)
            else:
                raise ValueError(
                    "%s: unexpected DCD record length %d (n_atom=%d)"
                    % (path, len(rec), n_atom))
            if x_rec is None or y_rec is None or z_rec is None:
                break
            xyz = np.empty((n_atom, 3), dtype=np.float32)
            xyz[:, 0] = np.frombuffer(x_rec, dtype="<f4")
            xyz[:, 1] = np.frombuffer(y_rec, dtype="<f4")
            xyz[:, 2] = np.frombuffer(z_rec, dtype="<f4")
            yield xyz


def load_dcd_xyz(path):
    frames = list(iter_dcd_xyz(path))
    if not frames:
        raise ValueError("No frames in %s" % path)
    return np.stack(frames, axis=0)


def closest_heavy_contacts(xyz, groups, pairs):
    """xyz (T, N, 3) -> (T, n_pairs) minimum heavy-atom distances in Angstrom."""
    n_frames = xyz.shape[0]
    out = np.empty((n_frames, len(pairs)), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        left = xyz[:, groups[i], :]
        right = xyz[:, groups[j], :]
        delta = left[:, :, None, :] - right[:, None, :, :]
        dist2 = np.sum(delta * delta, axis=-1)
        out[:, k] = np.sqrt(np.min(dist2, axis=(1, 2)))
    return out


def find_protein_dir(desres_root):
    root = Path(desres_root).expanduser()
    protein = root / PROTEIN_REL
    if protein.is_dir():
        return protein
    matches = list(root.glob("**/2JOF-0-protein"))
    matches = [path for path in matches if path.is_dir() and any(path.glob("*.dcd"))]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        "Could not find 2JOF-0-protein DCD directory under %s" % root)


def list_protein_dcds(protein_dir):
    dcds = sorted(Path(protein_dir).glob("2JOF-0-protein-*.dcd"))
    if not dcds:
        dcds = sorted(Path(protein_dir).glob("*.dcd"))
    if not dcds:
        raise FileNotFoundError("No DCD files in %s" % protein_dir)
    return dcds


def find_mae(protein_dir):
    for name in ("2JOF-0-protein.mae", "system.mae"):
        path = Path(protein_dir) / name
        if path.is_file():
            return path
        parent = Path(protein_dir).parent / name
        if parent.is_file():
            return parent
    raise FileNotFoundError("No Maestro topology next to %s" % protein_dir)


def featurize_desres(desres_root, max_frames=None):
    protein_dir = find_protein_dir(desres_root)
    mae_path = find_mae(protein_dir)
    dcds = list_protein_dcds(protein_dir)
    atoms = parse_mae_atoms(mae_path)
    groups, pairs, resids = residue_heavy_index_groups(atoms)
    n_expected = n_minimal_contacts(len(resids))
    if len(pairs) != n_expected:
        raise ValueError(
            "Contact count %d != %d for %d residues"
            % (len(pairs), n_expected, len(resids)))
    print("Topology: %s (%d atoms, %d residues, %d contacts)" % (
        mae_path, len(atoms), len(resids), len(pairs)))
    print("DCD files: %d under %s" % (len(dcds), protein_dir))

    chunks = []
    n_keep = int(max_frames) if max_frames else None
    n_done = 0
    for dcd in _progress(dcds, total=len(dcds), desc="Featurizing DCDs"):
        xyz = load_dcd_xyz(dcd)
        if xyz.shape[1] != len(atoms):
            raise ValueError(
                "%s has %d atoms but topology has %d"
                % (dcd, xyz.shape[1], len(atoms)))
        if n_keep is not None:
            remain = n_keep - n_done
            if remain <= 0:
                break
            xyz = xyz[:remain]
        chunks.append(closest_heavy_contacts(xyz, groups, pairs))
        n_done += xyz.shape[0]
        if n_keep is not None and n_done >= n_keep:
            break
    if not chunks:
        raise RuntimeError("No frames featurized from %s" % protein_dir)
    traj = np.concatenate(chunks, axis=0)
    meta = {
        "desres_root": str(Path(desres_root).expanduser()),
        "protein_dir": str(protein_dir),
        "mae_path": str(mae_path),
        "n_atoms": int(len(atoms)),
        "n_residues": int(len(resids)),
        "n_contacts": int(len(pairs)),
        "resids": [int(r) for r in resids],
        "n_frames": int(traj.shape[0]),
        "n_dcd_files": int(len(dcds)),
        "frame_dt_ns": FRAME_DT_NS,
        "source": "desres_dcd",
    }
    return traj, meta


def compute_normalize_stats(traj):
    mean = traj.mean(axis=0).astype(np.float32)
    std = traj.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def labels_to_one_hot(cluster_ids, n_states=None):
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    if n_states is None:
        n_states = int(cluster_ids.max()) + 1
    init_label = np.zeros((cluster_ids.shape[0], n_states), dtype=np.float32)
    init_label[np.arange(len(cluster_ids)), cluster_ids] = 1.0
    return init_label


def _tica_numpy(traj, lag, dim, eps=1e-8):
    """Minimal reversible TICA (no deeptime dependency)."""
    X = traj.astype(np.float64, copy=False)
    n_frames, n_feat = X.shape
    if lag <= 0 or lag >= n_frames:
        raise ValueError("tica lag must satisfy 0 < lag < N (got lag=%d, N=%d)"
                         % (lag, n_frames))
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
    n_comp = min(dim, n_feat)
    V = W @ evecs[:, :n_comp]
    proj = ((X - mean0) @ V).astype(np.float32)
    return proj, {
        "tica_eigenvalues": evals[:n_comp].astype(np.float32),
        "tica_mean": mean0.astype(np.float32),
        "tica_components": V.astype(np.float32),
        "tica_lag": int(lag),
        "tica_dim": int(n_comp),
        "tica_backend": "numpy",
    }


def make_tica_kmeans_labels(
    traj,
    n_clusters=DEFAULT_N_CLUSTERS,
    tica_lag=DEFAULT_TICA_LAG,
    tica_dim=DEFAULT_TICA_DIM,
    seed=0,
):
    """TICA then MiniBatchKMeans — tutorial3 / 2024 protein init protocol."""
    try:
        from deeptime.decomposition import TICA
        tica = TICA(dim=min(tica_dim, traj.shape[1]), lagtime=tica_lag)
        tica.fit(traj)
        proj = tica.fetch_model().transform(traj).astype(np.float32)
        tica_meta = {
            "tica_backend": "deeptime",
            "tica_lag": int(tica_lag),
            "tica_dim": int(proj.shape[1]),
        }
    except ImportError:
        proj, tica_meta = _tica_numpy(traj, lag=tica_lag, dim=tica_dim)

    print("TICA done: projection shape=%s backend=%s" % (
        proj.shape, tica_meta.get("tica_backend")))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=5,
        random_state=seed,
        batch_size=4096,
    )
    cluster_ids = km.fit_predict(proj)
    init_label = labels_to_one_hot(cluster_ids, n_states=n_clusters)
    meta = dict(tica_meta)
    meta["tica_projection"] = proj
    meta["cluster_centers"] = km.cluster_centers_.astype(np.float32)
    meta["cluster_ids"] = cluster_ids.astype(np.int32)
    return init_label, meta


def load_features_npy(path):
    traj = np.load(path)
    if traj.ndim != 2:
        raise ValueError("Expected 2D traj [N, F], got %s" % (traj.shape,))
    return traj.astype(np.float32, copy=False)


def main(
    desres_root=DEFAULT_DESRES_ROOT,
    features_npy=None,
    max_frames=None,
    n_clusters=DEFAULT_N_CLUSTERS,
    tica_lag=DEFAULT_TICA_LAG,
    tica_dim=DEFAULT_TICA_DIM,
    seed=0,
    skip_labels=False,
):
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    if features_npy:
        raw_path = Path(features_npy).expanduser()
        traj = load_features_npy(raw_path)
        source_meta = {
            "source": "features_npy",
            "features_npy": str(raw_path),
            "n_frames": int(traj.shape[0]),
            "n_contacts": int(traj.shape[1]),
            "frame_dt_ns": FRAME_DT_NS,
        }
        print("Loaded precomputed features %s shape=%s" % (raw_path, traj.shape))
    else:
        traj, source_meta = featurize_desres(desres_root, max_frames=max_frames)
        print("Featurized traj shape=%s dtype=%s" % (traj.shape, traj.dtype))

    if traj.shape[1] != EXPECTED_N_CONTACTS:
        print("Warning: expected %d contacts, got %d" % (
            EXPECTED_N_CONTACTS, traj.shape[1]))

    traj_path = SCRIPT_DIR / "traj_data.npy"
    np.save(traj_path, traj)
    print("Saved %s (%.1f MB)" % (traj_path, traj_path.stat().st_size / 1e6))

    mean, std = compute_normalize_stats(traj)
    mean_path = SCRIPT_DIR / "data_mean.npy"
    std_path = SCRIPT_DIR / "data_std.npy"
    np.save(mean_path, mean)
    np.save(std_path, std)
    print("Saved %s / %s" % (mean_path, std_path))

    meta = dict(source_meta)
    meta.update({
        "traj_shape": list(traj.shape),
        "data_mean": mean_path.name,
        "data_std": std_path.name,
        "tica_lag": int(tica_lag),
        "tica_dim": int(tica_dim),
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "frame_dt_ns": FRAME_DT_NS,
        "max_frames": None if max_frames is None else int(max_frames),
    })

    if skip_labels:
        meta_path = SCRIPT_DIR / "prepare_meta.json"
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2)
        print("Skipped TICA+kmeans labels (--skip-labels)")
        print("Saved %s" % meta_path)
        return

    print("Running TICA (lag=%d, dim=%d) + MiniBatchKMeans(K=%d) ..." % (
        tica_lag, tica_dim, n_clusters))
    init_label, label_meta = make_tica_kmeans_labels(
        traj, n_clusters=n_clusters, tica_lag=tica_lag, tica_dim=tica_dim, seed=seed)

    label_path = SCRIPT_DIR / (
        "init_label_tica_kmeans%d_lag%d.npy" % (n_clusters, tica_lag))
    proj_path = SCRIPT_DIR / ("tica_projection_lag%d.npy" % tica_lag)
    centers_path = SCRIPT_DIR / (
        "init_label_tica_kmeans%d_lag%d_centers.npy" % (n_clusters, tica_lag))
    np.save(label_path, init_label)
    index_path = SCRIPT_DIR / (
        "init_label_tica_kmeans%d_lag%d_index.npy" % (n_clusters, tica_lag))
    np.save(index_path, init_label.argmax(axis=1).astype(np.int16))
    np.save(proj_path, label_meta["tica_projection"])
    np.save(centers_path, label_meta["cluster_centers"])

    meta["label_shape"] = list(init_label.shape)
    meta["label_path"] = label_path.name
    meta["tica_backend"] = label_meta.get("tica_backend")
    meta["tica_dim"] = int(label_meta.get("tica_dim", tica_dim))

    meta_path = SCRIPT_DIR / "prepare_meta.json"
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    pop = init_label.sum(axis=0)
    n_active = int((pop > 0).sum())
    print("Saved %s shape=%s" % (label_path, init_label.shape))
    print("Saved %s / %s" % (proj_path, centers_path))
    print("Saved %s" % meta_path)
    print("Initial labels: %d clusters, %d non-empty" % (
        init_label.shape[1], n_active))
    print("Top-10 populations (fraction): %s" % (
        (np.sort(pop)[::-1][:10] / len(traj)).round(4).tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Trp-cage 2024 SPIB features from local DESRES DCDs"
    )
    parser.add_argument(
        "--desres-root",
        type=str,
        default=str(DEFAULT_DESRES_ROOT),
        help="Science 2011 Trp-cage 290 K distribution directory",
    )
    parser.add_argument(
        "--features-npy",
        type=str,
        default=None,
        help="Skip DCD featurization and load a precomputed (N, 153) npy",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for a smoke test (full run omits this)",
    )
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--tica-lag", type=int, default=DEFAULT_TICA_LAG)
    parser.add_argument("--tica-dim", type=int, default=DEFAULT_TICA_DIM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Write traj_data and mean/std only (no TICA+kmeans)",
    )
    args = parser.parse_args()
    main(
        desres_root=args.desres_root,
        features_npy=args.features_npy,
        max_frames=args.max_frames,
        n_clusters=args.n_clusters,
        tica_lag=args.tica_lag,
        tica_dim=args.tica_dim,
        seed=args.seed,
        skip_labels=args.skip_labels,
    )

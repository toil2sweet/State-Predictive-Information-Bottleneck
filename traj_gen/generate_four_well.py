"""Generate the SPIB four-well trajectory with Langevin dynamics.

This implements the underdamped Bussi--Parrinello splitting integrator and
the reduced-unit protocol described for the analytical four-well potential in
Wang and Tiwary, J. Chem. Phys. 154, 134111 (2021).

The physical inverse temperature ``BETA`` below is unrelated to the ``beta``
regularization hyperparameter used by SPIB during model training.

Generated trajectories and labels are written beside this script in
``traj_gen/``.

Example::

    python traj_gen/generate_four_well.py
    python traj_gen/generate_four_well.py --seed 2026
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


MASS = 1.0
BETA = 3.0
GAMMA = 4.0
DT = 0.001
STRIDE = 10
PAPER_PRODUCTION_TIME = 60_000.0
PAPER_PRODUCTION_STEPS = int(PAPER_PRODUCTION_TIME / DT)
INITIAL_STATE_NUM = 10
LABEL_EPS = 0.001
DEFAULT_SEED = 2026


def potential(q):
    """Return the SPIB four-well potential U(x, y)."""
    x, y = q
    return (
        2.0
        * (
            x ** 8
            + 0.6 * np.exp(-80.0 * x ** 2)
            + 0.2 * np.exp(-80.0 * (x - 0.5) ** 2)
            + 0.5 * np.exp(-40.0 * (x + 0.5) ** 2)
        )
        + (x ** 2 - 1.0) ** 2
        + y ** 2
    )


def force(q):
    """Return the deterministic force F = -grad U."""
    x, y = q
    dUdx = (
        16.0 * x ** 7
        - 192.0 * x * np.exp(-80.0 * x ** 2)
        - 64.0 * (x - 0.5) * np.exp(-80.0 * (x - 0.5) ** 2)
        - 80.0 * (x + 0.5) * np.exp(-40.0 * (x + 0.5) ** 2)
        + 4.0 * x * (x ** 2 - 1.0)
    )
    return np.array([-dUdx, -2.0 * y], dtype=np.float64)


def simulate(
    seed=DEFAULT_SEED,
    burn_steps=1_000_000,
    production_steps=PAPER_PRODUCTION_STEPS,
    show_progress=True,
):
    """Simulate and return saved positions with shape ``(frames, 2)``.

    The default production segment is the 60,000 reduced-time-unit trajectory
    reported in the SPIB paper. Burn-in is additional and is not saved or
    counted toward that production length.
    """
    if burn_steps < 0:
        raise ValueError("burn_steps must be non-negative")
    if production_steps <= 0:
        raise ValueError("production_steps must be positive")
    if production_steps % STRIDE != 0:
        raise ValueError("production_steps must be divisible by STRIDE")

    rng = np.random.default_rng(seed)
    q = np.array([0.722817, 0.0], dtype=np.float64)
    p = rng.normal(0.0, np.sqrt(MASS / BETA), size=2)

    c1 = np.exp(-GAMMA * DT / 2.0)
    c2 = np.sqrt((1.0 - c1 ** 2) * MASS / BETA)
    trajectory = np.empty((production_steps // STRIDE, 2), dtype=np.float64)

    frame = 0
    total_steps = burn_steps + production_steps
    step_iter = range(total_steps)
    if show_progress:
        step_iter = tqdm(
            step_iter,
            total=total_steps,
            desc="Langevin burn-in",
            unit="step",
            miniters=max(total_steps // 1000, 1),
            mininterval=0.2,
            dynamic_ncols=True,
        )

    for step in step_iter:
        p = c1 * p + c2 * rng.normal(size=2)

        f0 = force(q)
        q_new = q + p / MASS * DT + 0.5 * f0 / MASS * DT ** 2
        f1 = force(q_new)
        p = p + 0.5 * (f0 + f1) * DT
        q = q_new

        p = c1 * p + c2 * rng.normal(size=2)

        if step >= burn_steps:
            production_step = step - burn_steps + 1
            if production_step % STRIDE == 0:
                trajectory[frame] = q
                frame += 1

        if show_progress and step + 1 == burn_steps:
            step_iter.set_description("Langevin production")

    return trajectory


def output_filenames(seed, production_time=PAPER_PRODUCTION_TIME):
    """Return trajectory and label filenames that include the RNG seed."""
    time_tag = (
        int(production_time)
        if float(production_time).is_integer()
        else production_time
    )
    stem = f"Four_Well_{time_tag}time_seed{int(seed)}"
    return f"{stem}.npy", f"{stem}_init_label10.npy"


def initialize_spib_labels(
    trajectory,
    state_num=INITIAL_STATE_NUM,
    eps=LABEL_EPS,
):
    """Create SPIB's overcomplete one-hot labels from equal-width x bins.

    This follows the initialization used in ``SPIB_Demo.ipynb``: extend the
    observed x range by 0.01, add a small epsilon at both ends, then divide the
    resulting interval into ``state_num`` equal-width states.
    """
    if trajectory.ndim != 2 or trajectory.shape[1] < 1:
        raise ValueError("trajectory must have shape (frames, dimensions)")
    if trajectory.shape[0] == 0:
        raise ValueError("trajectory must contain at least one frame")
    if state_num <= 0:
        raise ValueError("state_num must be positive")

    x = trajectory[:, 0]
    x_min = x.min() - 0.01
    x_max = x.max() + 0.01
    x_det = (x_max - x_min + 2.0 * eps) / state_num
    x_edges = x_min - eps + np.arange(state_num + 1) * x_det

    labels = np.zeros((trajectory.shape[0], state_num), dtype=np.float64)
    for state in range(state_num):
        indices = (x > x_edges[state]) & (x <= x_edges[state + 1])
        labels[indices, state] = 1.0

    if not np.all(labels.sum(axis=1) == 1.0):
        raise RuntimeError(
            "SPIB initialization did not assign exactly one label per frame"
        )
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a four-well Langevin trajectory and SPIB initial labels."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed used for the integrator and output filenames (default: 2026).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    args = parser.parse_args(argv)

    traj = simulate(seed=args.seed, show_progress=not args.no_progress)
    labels = initialize_spib_labels(traj)

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_name, label_name = output_filenames(args.seed)
    trajectory_output = output_dir / traj_name
    labels_output = output_dir / label_name
    np.save(trajectory_output, traj)
    np.save(labels_output, labels)

    print("seed:", args.seed)
    print("trajectory saved:", trajectory_output)
    print("labels saved:", labels_output)
    print("trajectory shape:", traj.shape)
    print("labels shape:", labels.shape)
    print("state counts:", labels.sum(axis=0).astype(np.int64))
    print("mean(x,y):", traj.mean(axis=0))
    print("std(x,y):", traj.std(axis=0))
    print("expected std(y):", np.sqrt(1.0 / (2.0 * BETA)))


if __name__ == "__main__":
    main()

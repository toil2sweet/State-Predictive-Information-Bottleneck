"""Generate the CTC double-well trajectory with Langevin dynamics.

The analytical potential and default physical protocol follow the CTC paper:

    V(x, y) = x**4 / 4 - 3*x**2 + x + y**2 / 2

    mass = 1 Da, temperature = 300 K, friction = 100 ps^-1,
    time step = 4 fs, integration steps = 1e8, simulation time = 400 ns.

Saving every 100 integration steps produces the 1,000,000 frames used by the
CTC double-well example. The paper specifies Langevin dynamics but not the
particular finite-step integrator; this implementation uses the symmetric
Langevin-middle (BAOAB) splitting scheme.

The generated five-state one-hot labels are overcomplete SPIB initialization
labels based on equal-width bins along x. They are not CTC clustering labels.

Examples::

    python traj_gen/generate_double_well.py
    python traj_gen/generate_double_well.py --seed 2026
    python traj_gen/generate_double_well.py --frames 10000 --stride 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        del args, kwargs

        def decorator(function):
            return function

        return decorator


MASS_DALTON = 1.0
TEMPERATURE_K = 300.0
FRICTION_PER_PS = 100.0
DT_PS = 0.004  # 4 fs
GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324
KBT_KJ_PER_MOL = GAS_CONSTANT_KJ_PER_MOL_K * TEMPERATURE_K

DEFAULT_FRAMES = 1_000_000
DEFAULT_STRIDE = 100
DEFAULT_INTEGRATION_STEPS = DEFAULT_FRAMES * DEFAULT_STRIDE
DEFAULT_SIMULATION_TIME_NS = DEFAULT_INTEGRATION_STEPS * DT_PS / 1000.0
DEFAULT_STATE_NUM = 5
DEFAULT_SEED = 2026
DEFAULT_CHUNK_FRAMES = 10_000
LABEL_EPS = 1e-3
INITIAL_POSITION_NM = np.array([-1.0, 0.0], dtype=np.float64)


def potential(q):
    """Return the CTC double-well potential in kJ/mol."""
    x, y = np.asarray(q)
    return 0.25 * x ** 4 - 3.0 * x ** 2 + x + 0.5 * y ** 2


def force(q):
    """Return ``-grad(V)`` in kJ mol^-1 nm^-1."""
    x, y = np.asarray(q)
    return np.array([-x ** 3 + 6.0 * x - 1.0, -y], dtype=np.float64)


@njit(cache=False)
def _integrate_chunk(q, velocity, normal_noise, stride, dt, friction, kbt, mass):
    """Advance one trajectory chunk with Langevin-middle (BAOAB)."""
    frame_count = normal_noise.shape[0] // stride
    trajectory = np.empty((frame_count, 2), dtype=np.float64)
    c = np.exp(-friction * dt)
    noise_scale = np.sqrt((1.0 - c * c) * kbt / mass)
    noise_index = 0

    for frame in range(frame_count):
        for _ in range(stride):
            x = q[0]
            y = q[1]
            fx = -x ** 3 + 6.0 * x - 1.0
            fy = -y

            velocity[0] += 0.5 * dt * fx / mass
            velocity[1] += 0.5 * dt * fy / mass
            q[0] += 0.5 * dt * velocity[0]
            q[1] += 0.5 * dt * velocity[1]

            velocity[0] = c * velocity[0] + noise_scale * normal_noise[noise_index, 0]
            velocity[1] = c * velocity[1] + noise_scale * normal_noise[noise_index, 1]
            noise_index += 1

            q[0] += 0.5 * dt * velocity[0]
            q[1] += 0.5 * dt * velocity[1]

            x = q[0]
            y = q[1]
            fx = -x ** 3 + 6.0 * x - 1.0
            fy = -y
            velocity[0] += 0.5 * dt * fx / mass
            velocity[1] += 0.5 * dt * fy / mass

        trajectory[frame, 0] = q[0]
        trajectory[frame, 1] = q[1]

    return trajectory, q, velocity


def simulate(
    frames=DEFAULT_FRAMES,
    stride=DEFAULT_STRIDE,
    seed=DEFAULT_SEED,
    chunk_frames=DEFAULT_CHUNK_FRAMES,
    show_progress=True,
):
    """Return saved ``(x, y)`` positions with shape ``(frames, 2)``.

    A saved frame is recorded after every ``stride`` integration steps. With
    the defaults, this is 1,000,000 frames from 100,000,000 steps (400 ns).
    """
    frames = int(frames)
    stride = int(stride)
    chunk_frames = int(chunk_frames)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    rng = np.random.default_rng(seed)
    q = INITIAL_POSITION_NM.copy()
    velocity = rng.normal(
        loc=0.0,
        scale=np.sqrt(KBT_KJ_PER_MOL / MASS_DALTON),
        size=2,
    )
    trajectory = np.empty((frames, 2), dtype=np.float32)

    progress = tqdm(
        total=frames,
        desc="CTC double-well Langevin",
        unit="frame",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    start = 0
    while start < frames:
        count = min(chunk_frames, frames - start)
        normal_noise = rng.normal(size=(count * stride, 2))
        chunk, q, velocity = _integrate_chunk(
            q,
            velocity,
            normal_noise,
            stride,
            DT_PS,
            FRICTION_PER_PS,
            KBT_KJ_PER_MOL,
            MASS_DALTON,
        )
        trajectory[start:start + count] = chunk
        start += count
        progress.update(count)
    progress.close()
    return trajectory


def initialize_spib_labels(
    trajectory,
    state_num=DEFAULT_STATE_NUM,
    eps=LABEL_EPS,
):
    """Create overcomplete one-hot labels from equal-width bins along x."""
    trajectory = np.asarray(trajectory)
    if trajectory.ndim != 2 or trajectory.shape[1] < 1:
        raise ValueError("trajectory must have shape (frames, dimensions)")
    if trajectory.shape[0] == 0:
        raise ValueError("trajectory must contain at least one frame")
    state_num = int(state_num)
    if state_num <= 0:
        raise ValueError("state_num must be positive")

    x = trajectory[:, 0]
    x_min = float(x.min()) - 0.01
    x_max = float(x.max()) + 0.01
    x_det = (x_max - x_min + 2.0 * eps) / state_num
    x_edges = x_min - eps + np.arange(state_num + 1) * x_det

    labels = np.zeros((trajectory.shape[0], state_num), dtype=np.float32)
    for state in range(state_num):
        indices = (x > x_edges[state]) & (x <= x_edges[state + 1])
        labels[indices, state] = 1.0

    if not np.all(labels.sum(axis=1) == 1.0):
        raise RuntimeError(
            "SPIB initialization did not assign exactly one label per frame"
        )
    return labels, x_edges


def output_filenames(frames, seed, state_num):
    """Return filenames containing the frame count, seed, and label count."""
    stem = f"Double_Well_CTC_{int(frames)}frames_seed{int(seed)}"
    return f"{stem}.npy", f"{stem}_init_label{int(state_num)}.npy"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate the CTC double-well Langevin trajectory and SPIB "
            "overcomplete initial labels."
        )
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Integration steps per saved frame (CTC default: 100).",
    )
    parser.add_argument("--state-num", type=int, default=DEFAULT_STATE_NUM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="Saved frames per compiled integration chunk.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory (default: traj_gen beside this script).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    args = parser.parse_args(argv)

    trajectory = simulate(
        frames=args.frames,
        stride=args.stride,
        seed=args.seed,
        chunk_frames=args.chunk_frames,
        show_progress=not args.no_progress,
    )
    labels, x_edges = initialize_spib_labels(
        trajectory, state_num=args.state_num)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_name, label_name = output_filenames(
        args.frames, args.seed, args.state_num)
    trajectory_path = args.output_dir / trajectory_name
    label_path = args.output_dir / label_name
    np.save(trajectory_path, trajectory)
    np.save(label_path, labels)

    integration_steps = int(args.frames) * int(args.stride)
    simulation_time_ns = integration_steps * DT_PS / 1000.0
    print("integrator: Langevin-middle (BAOAB)")
    print("numba acceleration:", NUMBA_AVAILABLE)
    print("seed:", args.seed)
    print("integration steps:", integration_steps)
    print("simulation time (ns):", simulation_time_ns)
    print("trajectory saved:", trajectory_path)
    print("labels saved:", label_path)
    print("trajectory shape:", trajectory.shape)
    print("labels shape:", labels.shape)
    print("x-bin edges:", x_edges)
    print("state counts:", labels.sum(axis=0).astype(np.int64))
    print("mean(x,y):", trajectory.mean(axis=0))
    print("std(x,y):", trajectory.std(axis=0))


if __name__ == "__main__":
    main()

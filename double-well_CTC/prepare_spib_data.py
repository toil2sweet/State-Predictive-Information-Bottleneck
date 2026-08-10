"""
Prepare SPIB inputs from CTC double-well trajectory.

Default initial labels use the same overcomplete x-axis discretization as the
four-well SPIB demo (SPIB_Demo.ipynb): divide x into state_num bins.
"""
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_NUM = 5
EPS = 1e-3


def load_traj_data() -> np.ndarray:
    traj_raw = np.load(SCRIPT_DIR / "traj.npy")
    return traj_raw[:, :2].astype(np.float32)


def make_init_label_overcomplete(
    traj_data: np.ndarray,
    state_num: int = STATE_NUM,
    index: int = 0,
    eps: float = EPS,
) -> np.ndarray:
    """Discretize along x into state_num states (same recipe as SPIB_Demo.ipynb)."""
    x_max = traj_data[:, index].max() + 0.01
    x_min = traj_data[:, index].min() - 0.01
    x_det = (x_max - x_min + 2 * eps) / state_num
    x_list = np.array([(x_min - eps + n * x_det) for n in range(state_num + 1)])

    init_label = np.zeros((traj_data.shape[0], state_num), dtype=np.float32)
    for j in range(state_num):
        indices = (traj_data[:, index] > x_list[j]) & (traj_data[:, index] <= x_list[j + 1])
        init_label[indices, j] = 1.0

    return init_label


def main(state_num: int = STATE_NUM) -> None:
    traj_data = load_traj_data()
    init_label = make_init_label_overcomplete(traj_data, state_num=state_num)

    traj_path = SCRIPT_DIR / "traj_data.npy"
    label_path = SCRIPT_DIR / f"init_label{state_num}.npy"
    np.save(traj_path, traj_data)
    np.save(label_path, init_label)

    population = init_label.sum(axis=0)
    n_frames = len(traj_data)
    print(f"Saved {traj_path}  shape={traj_data.shape}")
    print(f"Saved {label_path}  shape={init_label.shape}")
    print(f"Initial state populations ({state_num} bins along x):")
    for j, pop in enumerate(population):
        print(f"  state {j}: {pop:.0f} ({100 * pop / n_frames:.2f}%)")


if __name__ == "__main__":
    main()

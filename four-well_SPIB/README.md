# Four-Well SPIB utilities

This directory contains plotting code specific to the analytical Four-Well
system:

- `plot_free_energy_likeSPIB.py`: SPIB Fig. 5(a)/(b)-style potential and
  empirical free-energy plots.
- `plot_energy_landscape_likeCTC.py`: CTC Fig. 2(F)-style analytical
  potential plot adapted to the Four-Well system.

The trajectory generator, generated `.npy` trajectories, and labels are kept
together under `traj_gen/`. Generated figures remain in `fig/`. Run all scripts
from the repository root.

```bash
python traj_gen/generate_four_well.py --seed 2026
python four-well_SPIB/plot_free_energy_likeSPIB.py
python four-well_SPIB/plot_energy_landscape_likeCTC.py
```

Shared training entry points and configuration files intentionally remain at
`test_model_advanced.py` and `examples/*.ini`. The shared
`plot_transition_states.py` loads the two Four-Well plotting modules directly
from this directory.

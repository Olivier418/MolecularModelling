# CLAUDE.md

Classical molecular dynamics (MD) simulator for small-molecule mixtures (water, ethanol, O2, H2) in a periodic cubic box. Pure NumPy, with one Numba kernel. Forces: harmonic bonds, harmonic angles, Ryckaert-Bellemans dihedrals and Lennard-Jones (Lorentz-Berthelot mixing). Output: `.xyz` trajectories, plus energy, temperature, phase-space and RDF plots.

> Layout below is the **pre-refactor** state. Update this file whenever modules move.

## Running

- Run everything **from the repo root**. `molecule_data/...` paths are relative to the CWD.
- Simulation: `python main.py`. It prompts for water and ethanol weights via `input()`. All other parameters are module-level constants in `main.py`.
- Tests: `python -m pytest tests/` (`tests/helpers.py` holds shared fixtures). Known bugs are marked `xfail(strict=True)`.
- Deps: `requirements.txt` (unpinned). The local env runs Python 3.14 and NumPy 2.x.
- Output goes to `trajectories/boxsize=..._roff=..__<mix>/dt=..._T=..._.../` (gitignored).
- `VMD_vis=True` shells out to a hard-coded `C:\Program Files\VMD\vmd.exe`.

## Module map / data flow

```
main.py                      params → validate_parameters → random_config → integrate
initial_config_generation.py random_config(): reads molecule_data/*.itp/.xyz, tiles molecules on an
                             N×N×N grid, returns 18-tuple of flat per-atom/per-bond arrays.
                             random_molecule(): random test fixture (uses auxiliary.random_graph)
integrators.py               euler/verlet/velo_verlet/beeman = dicts of lambdas {'positions','velocities'}
                             integrator(): one step. integrate(): setup + optional LJ "soft start"
                             equilibration (r_threshold_scalar) + main loop + save closure + plotting
forces.py                    bond_force, ang_force, LJ_force, dihedral_force, net_force
energy.py                    same four terms as energies + kinetic_energy (duplicates force geometry)
auxiliary.py                 grab-bag: bond-index mapping, LJ pair list, cutoff calc, PBC distances
                             (@njit diff_and_dist_arr), thermostat (scale_velocity), molecule
                             rotation/placement, parameter validation, progress bar, test graph helpers
parse_functions.py           xyz/itp readers and writers
visualizations.py            energy/temperature plots, phase-space diagrams, RDF (init + GIFs), pie chart, VMD
config.py                    masses, molar volumes, k_B, Avogadro
```

Key representations:
- `bonds` (B×2) are stored **sorted** (`i<j`). `angles` (A×3) and `dihedrals` (D×4) hold atom indices.
- `find_bond_indices` maps each angle/dihedral to rows of `bonds`, plus a ±1 sign, so angle and dihedral terms reuse `diff_bonds`/`D_bonds` and never recompute vectors.
- LJ uses a precomputed **all-pairs** list `i_LJ, j_LJ` (`compute_LJ_indices`). It excludes 1-2/1-3/1-4 pairs via angles/dihedrals and atoms with zero ε. Per-pair `epsilon_LJ`, `sigma_LJ` and `r_cutoff_LJ` are aligned with this list. No neighbour list exists.
- `force_parameters` is an 18-element positional tuple splatted into `net_force`. Its order must match the signature.
- Positions are **not** wrapped during integration. `project_to_box` wraps whole molecules, and only at save time. Bonds therefore use `boxsize=0` (no minimum-image); LJ uses minimum-image.

## Units (internal)

nm · ps · amu · kJ/mol · K. `kboltz = 0.00831` kJ/(mol·K).

The input files use **other units**, converted in `itp_to_data` / `xyz_to_array`:
- `.xyz`: Å (×0.1 on read, ×10 on write).
- `.itp` bonds: `k_b` ×100 → kJ/mol/nm², `r0` Å ×0.1 → nm.
- `.itp` angles: radians, kJ/mol/rad².
- `.itp` atoms: ε kJ/mol, σ nm (already nm!), mass amu.
- Dihedral C1–C4: kJ/mol.

## .itp format (custom, not GROMACS)

```
bonds N        → N lines: i j k_b r0
angles N       → N lines: i j k k_theta theta0
dihedrals N    → N lines: i j k l C1 C2 C3 C4
atoms N        → N lines: epsilon sigma mass
```
Section headers are parsed by fixed character offsets (`[6:]`, `[7:]`, `[10:]`), so header spelling matters.

## Known issues (verified, not yet fixed)

- `visualizations.initialize_RDF` uses `np.in1d` (removed in NumPy 2.x), so the default `RDF_vis=True` crashes. It also uses deprecated `np.core.records`.
- O2: the bonded O–O pair is **not** excluded from LJ, because diatomics have no angles. The soft-start loop can then never terminate.
- `array_to_xyz` truncates `str(float)` to 10 chars, which corrupts values printed in scientific notation (1.23e-05 → 1.23).
- `integrate.save_visualization_data`: the RDF `for i in ...` loop shadows the timestep `i`, so the xyz comment line gets the wrong time.
- Temperature control has no effect for `verlet`, because Verlet positions never read `v`.
- `random_config` computes per-grid-site rotations (`theta`, `phi`) and then ignores them (`rotate_molecule(pos, 0, 0)`).
- Performance: an 8.4k-atom water/ethanol box has ~19.7M LJ pairs, and only ~1.2% are inside the cutoff. The minimum-image distance over all pairs takes ~2 s/step.

## Conventions

- Snake-case functions, but mixed-case arrays are common (`D_LJ`, `Xim1`, `KB`). `im1`/`im2` = step i−1/i−2.
- Comments are dense and block-style (`# ---- Section ----`). Lines are wrapped at ~80 chars.
- Vectorised NumPy everywhere. Scatter-adds use `np.add.at` or `np.bincount`.

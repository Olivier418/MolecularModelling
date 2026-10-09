# CLAUDE.md

Classical molecular dynamics (MD) simulator for small-molecule mixtures (water, ethanol, O2, H2) in a periodic cubic box. Python package `mdsim` in `src/` (NumPy, SciPy, Numba). Forces: harmonic bonds, harmonic angles, Ryckaert-Bellemans dihedrals and Lennard-Jones (Lorentz-Berthelot mixing). Output: `.xyz` trajectories, plus energy, temperature, phase-space and RDF plots.

## Running

- Install: `pip install -e .[dev]`. Tests also run without installing, because `pyproject.toml` adds `src` and `tests` to `pythonpath`.
- Simulation: `python -m mdsim -c configs/default.toml [-w Water=1 -w Ethanol=1] [--seed N] [--no-show]`.
  - Without `-c`, the built-in defaults in `config.Config` apply.
  - `configs/default.toml` documents every setting and must match those defaults; `tests/test_config.py` checks this.
- Tests: `python -m pytest`. `tests/helpers.py` has `make_topology`, `forcefield`, `random_molecule` and `water_ethanol_pair`.
- Lint: `ruff check src tests` (configured in pyproject, not installed by default).
- Output goes to `<output.directory>/boxsize=..._roff=..__<mix>/dt=..._T=..._.../` (`trajectories/` is gitignored).
- VMD (`plots.vmd = true`) uses `output.vmd_path`.

## Package map (`src/mdsim/`)

```
cli.py / __main__.py  argparse → config.load_config → simulation.run
config.py         Config dataclass (defaults + validate()); TOML_LAYOUT maps TOML sections/keys → fields
constants.py      KBOLTZ, AVOGADRO, MOLAR_VOLUMES (needed per molecule!), MOLECULE_DIR
data/molecules/   <Name>.xyz + <Name>.itp per molecule
io.py             read_xyz / write_xyz(open file, ...) / read_itp → dict / write_itp(topology)
system.py         Topology dataclass (+ replicate/concatenate), load_molecule, build_system (grid
                  placement), orient_molecule, initial_velocities, kinetic_energy, temperature_scale
geometry.py       pair_vectors (@njit, minimum image if boxsize≠0), map_to_bonds,
                  angle_geometry, dihedral_geometry (shared by forces and energies), wrap_molecules
forcefield.py     lj_pairs (exclusions), lj_cutoffs; bond/angle/dihedral/LJ_term → (atom indices,
                  force vectors, energy|None); ForceField(topology, boxsize, ...)
                  .compute(x, compute_energy=False, cap_scalar=None) → ForceResult(forces, diff/D
                  for bonds and LJ, energies {"Bond","Angular","LJ","Dihedral"} | None); .timings
integrators.py    Method(positions, velocities) records: euler, verlet, velocity_verlet, beeman;
                  METHODS by name; State(x, v, a, x_prev, a_prev, KE); initial_state; step
simulation.py     output_folder, soft_start, run(config): build → soft start → loop → plots
observables.py    PhaseSpace (bond length + exact dD/dt from velocities), RDF recorders
plotting.py       energy/temperature/phase-space/RDF gifs/computation pie chart (show flag)
vmd.py            launches VMD with a generated Tcl script
progress.py       ProgressBar (redraws only when the % changes)
```

Key representations:
- `bonds` (B×2) are stored **sorted** (`i<j`). `angles` (A×3) and `dihedrals` (D×4) hold atom indices. `C` is a (D×4) array.
- `map_to_bonds` maps each angle/dihedral to rows of `bonds`, plus a ±1 sign, so angle and dihedral terms reuse `diff_bonds`/`D_bonds` and never recompute vectors.
- LJ pairs are precomputed in `ForceField` (`i_LJ`, `j_LJ`, aligned per-pair `epsilon_LJ`, `sigma_LJ`, `r_cutoff_LJ`). Excluded: bonded/1-3/1-4 pairs and atoms with ε=0.
- Positions are **not** wrapped during integration. `wrap_molecules` wraps whole molecules, and only when writing xyz. Bonds use `boxsize=0` (no minimum image); LJ and RDF use minimum image.
- `step(..., temp=T)` is the thermostat: it rescales v every step. For Verlet it rescales `x_prev` instead, because Verlet has no explicit velocity.
- The first step is always Euler (`initial_state` sets `x_prev = x`).
- Energies are only computed on save steps: `simulation.run` passes a `compute_energy=True` force function to `step` there. `State.KE` is computed once per step, and the thermostat reuses it.
- All randomness flows from one `np.random.default_rng(config.seed)`.

## Units (internal)

nm · ps · amu · kJ/mol · K.

The input files use **other units**, converted in `io.py`:
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

## Known issues / caveats (not yet fixed)

- Performance: an 8.4k-atom water/ethanol box has ~19.7M LJ pairs, and only ~1.2% are inside the cutoff. The minimum-image distance over all pairs takes ~2 s/step.
- `MOLAR_VOLUMES` for Oxygen/Hydrogen (22.4) look like the ideal-gas value in **L/mol**, but they are used as cm³/mol, so these gases are packed at liquid-like density. This is left as is; ask the user before changing it.
- Modelling approximations (no electrostatics, truncated unshifted LJ, isokinetic thermostat, N(0,1) initial velocities, RDF normalised by shape only) are deliberate. Don't "fix" them without asking.

## Conventions

- Comments are dense and block-style (`# ---- Section ----`). Lines are wrapped at 80 chars.
- Snake_case, but physics arrays keep the original names (`D_LJ`, `diff_bonds`, `im1`/`im2` = step i−1/i−2).
- Vectorised NumPy everywhere.
- Refactors should keep results bit-identical unless a change is intended. Check this by comparing against the previous commit.

# jostle internals and reference

This document covers everything the [README](../README.md) leaves out: every
setting, the results and output files, the model, how the code is organised,
and the design decisions behind it.

- [1. A run, step by step](#1-a-run-step-by-step)
- [2. Configuration](#2-configuration)
- [3. Pre-run checks and warnings](#3-pre-run-checks-and-warnings)
- [4. Results and output files](#4-results-and-output-files)
- [5. Viewing in VMD](#5-viewing-in-vmd)
- [6. Model](#6-model)
- [7. Molecules and input files](#7-molecules-and-input-files)
- [8. Code organisation](#8-code-organisation)
- [9. Implementation details](#9-implementation-details)
- [10. Performance](#10-performance)
- [11. Development](#11-development)

## 1. A run, step by step

`run(config)` in [`simulation.py`](../src/jostle/simulation.py):

1. **Validate** the config (`config.validate()`).
2. **Check** it (`checks.check_config(config)`): print the expected size of
   `trajectory.xyz` and `results.npz`, and warn about settings that are valid
   but would waste the run ([section 3](#3-pre-run-checks-and-warnings)).
3. **Build** the box: molecules on random grid sites with random orientations,
   then **minimise** the energy to remove overlaps, then draw
   Maxwell–Boltzmann velocities.
4. **Integrate**, recording each measurement at its own interval, and writing
   trajectory frames as it goes.
5. Collect everything in a `Results`, **save** it (`results.npz` +
   `config.json`) and return it.


## 2. Configuration

A `Config` ([`config.py`](../src/jostle/config.py)) holds three groups of
settings, plus run control:

- **`config.system`: what is simulated.** Changing these gives a different
  physical system or ensemble.
- **`config.numerics`: how it is approximated.** Changing these gives a
  better or worse approximation of the same system.
- **`config.recording`: what is measured during the run.** Changing these
  measures the same trajectory differently.

Every setting is optional, and can be given by name whatever its group:

- as keyword arguments: `Config(box_length=3.0, dt=0.001, seed=1)` (flat
  keywords are routed to their group; code reads `config.numerics.dt`);
- as whole groups: `Config(system=SystemSettings(box_length=3.0))`.

`config.validate()` checks the settings (each group has its own `validate()`,
and `Config` adds cross-group checks such as the largest LJ cutoff ≤ L/2) and
returns the config. `config.replace(...)` returns a validated copy with some
settings changed. `to_dict`/`from_dict` convert to and from the `config.json`
record. `help(Config)` and `help(SystemSettings)` (etc.) show the documented
defaults.

Settings are Python only: there are no input config files (`config.json` is
only a record of a run), no presets and no CLI. Plotting is never configured
in `Config`.

| Field | Default | Meaning |
|---|---|---|
| **`system`** | | |
| `weights` | `{"Water": 1, "Ethanol": 1}` | Relative amount of each molecule (normalised). |
| `box_length` | `4.0` | Edge length of the cubic box (nm). |
| `temperature` | `298.15` | Initial (and target) temperature (K). Starting velocities are Maxwell–Boltzmann with zero centre-of-mass momentum (3N−3 degrees of freedom). |
| `thermostat` | `True` | Rescale the velocities to `temperature` every step. This samples at constant number of molecules, volume and temperature (NVT, the canonical ensemble, here only approximately; see [Integration](#integration)). `False` lets the system evolve freely at constant number, volume and total energy (NVE, the microcanonical ensemble). |
| `molecule_dir` | built-in | Folder with `<Name>.xyz` and `<Name>.itp` files. |
| **`numerics`** | | |
| `integrator` | `"velocity_verlet"` | `euler`, `verlet`, `velocity_verlet` or `beeman`. |
| `dt` | `0.00025` | Time step (ps). |
| `lj_cutoff` | `2.5` | LJ cutoff as a multiple of σ (shifted force). The largest cutoff must be at most half the box. |
| **`recording`** | | |
| `energy_interval` | `20` | Record the energies, kinetic energy and temperature every this many steps (5 fs at the default `dt`). |
| `trajectory_interval` | `100` | Write a trajectory frame every this many steps (25 fs). The default run writes 400 frames, about 170 MB. |
| `rdf_interval` | `100` | Record the RDF histograms every this many steps (25 fs: 40 frames per 1 ps block). Closer frames are nearly identical, so they cost time without improving g(r). |
| `record_potential_energies` | `True` | Record the potential energy per term, and the LJ correction (costs a little time on energy steps). |
| `phase_space_bonds` | water O–H, ethanol O–H, C–O, C–H | Bond types for the phase-space diagrams, as `"O-H (Water)"`. At most 100 bonds of each type are tracked, spread through the box. `[]` records none. |
| `phase_space_window` | `1.0` | The phase space is recorded every step (about 34 points per O–H vibration of 8.6 fs), but only over the last `phase_space_window` ps of the run: the orbits repeat, so a short window shows them all, and the data stays small (at most about 30 MB) whatever the run length. `None` records the whole run. |
| `rdf_pairs` | six O~H and O~O pairs | Pairs for the RDF, as `"O~H (Water~Ethanol)"`: an O atom in a water molecule with an H atom in an ethanol molecule. Only pairs in different molecules count. `[]` records none. |
| `rdf_pair_fraction` | `1.0` | Fraction of the pairs to sample (0 < fraction ≤ 1). |
| `rdf_bin_width` | `0.005` | Histogram bin width (nm). |
| **Run control** | | |
| `end_time` | `10.0` | Simulated time (ps). |
| `seed` | `None` (random) | Seed for all random choices (placement, orientations, velocities, RDF thinning). The RDF draws from its own stream, so the RDF settings do not change the trajectory. |
| `output_dir` | `"trajectories"` | Where run folders are created, relative to the working directory. The notebook uses `"../trajectories"`, the repository root. |

Temperature, computation times and the trajectory are always recorded.

The notebook's fast end-to-end check is
`Config(box_length=2.0, end_time=5.0, seed=1, rdf_bin_width=0.01)`.

### Internal constants

These numerical constants need no tuning, so they are deliberately not
settings. Each is documented next to the code that uses it:

| Constant | Value | Meaning |
|---|---|---|
| `neighbours.NEIGHBOUR_SKIN` | `0.1` nm | Extra neighbour-list distance. Affects speed only, not results. |
| `plotting.RDF_CONFIDENCE` | `0.9` | Confidence level of the shaded RDF intervals. |
| `plotting.RDF_BLOCK_TIME` | `1.0` ps | Only the default of the `block_time` keyword of `plot_rdf` and `animate_rdf` (see [RDF](#rdf)). |
| `vmd.CUE_MODE`, `vmd.CUE_DENSITY` | `"Exp2"`, `0.4` | Depth cueing in VMD. |
| `observables.PHASE_SPACE_MAX_BONDS` | `100` | Most bonds of each type tracked in the phase space. More only overlap in the diagram. |


## 3. Pre-run checks and warnings

Before building the box, `run` calls `checks.check_config(config)`
([`checks.py`](../src/jostle/checks.py)). It always prints the expected size of
`trajectory.xyz` and `results.npz`, and gives a `JostleWarning` for settings
that are valid but would waste the run or a measurement. `check_config(config)` can
also be called on its own. The default settings give none (this is
tested).

| Warning | When |
|---|---|
| `dt` too large | Fewer than 20 steps per vibration of the fastest bond in the mixture (O–H: 8.6 fs, so `dt` > 0.43 fs): the energy is poorly conserved and the phase-space orbits are jagged. Above period/π, velocity Verlet is unstable. |
| Euler | `integrator="euler"` does not conserve energy. |
| Too few frames | The energy, trajectory or RDF gets fewer than 10 frames. |
| RDF too frequent | RDF frames less than 10 fs apart. |
| RDF run too short | `end_time` shorter than two RDF blocks (2 ps), so no confidence interval. |
| Phase space too short | `phase_space_window` shorter than one vibration of the slowest tracked bond. |
| Large output | `trajectory.xyz` or `results.npz` above 1 GB. |
| Likely typo | A `phase_space_bonds` or `rdf_pairs` entry for a molecule without files in `molecule_dir`, or for atoms or a bond that its molecule does not have. Entries for known molecules that are not in the mixture (like `"O-O (Oxygen)"` in a water–ethanol run) are skipped silently. |

Bond periods come from `describe_molecule`: 2π/√(k_bond/μ) per bond type, with μ the
reduced mass. `JostleWarning` uses the `"always"` filter, so each warning shows
on every run. To silence them:
`warnings.filterwarnings("ignore", category=JostleWarning)`.

## 4. Results and output files

`run(config)` returns a `Results` object
([`results.py`](../src/jostle/results.py)). Each measurement has its own frames
(see the `*_interval` settings, plus the start). A measurement that was switched
off is `None`.

| Attribute / method | Contents |
|---|---|
| `energy_times` | Time of each energy frame (ps), for `potential_energies`, `kinetic_energies` and `temperature`. |
| `potential_energies` | Dict of the potential energy terms `"Bond"`, `"Angle"`, `"LJ"` and `"Dihedral"` (arrays, kJ/mol). `"LJ"` is the shifted-force energy that the dynamics conserve. |
| `lj_correction` | Added to `potential_energies["LJ"]`, estimates the energy of the full LJ potential (kJ/mol; see [LJ energy correction](#lj-energy-correction)). Not part of the total energy. |
| `temperature`, `kinetic_energies` | Temperature (K) and kinetic energy (kJ/mol). |
| `rdf` | `rdf.times` (ps), `rdf.histograms` (raw counts, frames × pairs × bins), `rdf.bin_edges`, `rdf.pair_labels`, `rdf.nr_pairs` (sampled pairs per entry) and `rdf.g_of_r()` (the counts as g(r)). |
| `phase_space` | `phase_space.times` (ps), `phase_space.lengths` and `.length_rates` (frames × bonds): bond lengths and their rates of change, per tracked bond type in `phase_space.bond_types`. |
| `topology` | The system (elements, bonds, masses, …). |
| `timings`, `wall_time` | Seconds spent in each part of the main loop, and in total (wall-clock time, not simulated time). |
| `folder`, `trajectory` | The run folder and the path of `trajectory.xyz`. |
| `plot_energy()`, `plot_temperature()`, `plot_timings()` | Figures of the above. |
| `plot_rdf(time=None, pairs=None, block_time=1.0)` | Running-mean g(r) up to `time` (default: the end) with 90% confidence bands (see [RDF](#rdf)). |
| `plot_phase_space(bond_type=None)` | Figure for one bond type, named as in `phase_space_bonds` (e.g. `"O-H (Water)"`); without `bond_type`, a list of figures for all of them. |
| `animate_rdf(pair=None, max_frames=200, block_time=1.0)` | Makes and displays an animation of the RDF over time: of one pair, or of the means of all pairs. |
| `save()`, `Results.load(folder)` | Write the recorded data to the run folder (`run` already does), and read it back. |
| `view_in_vmd(vmd_path=None, show_periodic_images=True, show_box=True, show_image_outlines=True)` | Opens the trajectory in VMD, in its own window. |
| `vmd_script` | The Tcl script `view_in_vmd()` wrote to the run folder. |

**Figures are made on request.** Each `plot_*` method returns a matplotlib `Figure`, which a
notebook shows inline (the builders in `plotting.py` close it in pyplot right
away, so it is shown once). It takes `save=False` (the default: don't save),
`save=True` (save in the run folder, under the name in the table below) or
`save="path/name.pdf"` (save anywhere, in any format matplotlib knows).
`animate_rdf()` takes `save` the same way (without it, the gif goes to a temporary
file), except that a path is a folder (for the gif and pngs of its first and
last frames), as it is for `plot_phase_space()` without a bond type.

**Loading a finished run.** `Results.load(folder)` reads `results.npz` and
`config.json` from a run folder. Every method then works as it did right
after the run. An `.npz` file is NumPy's archive of named arrays
(`np.load(path)` opens it as a dict); values are stored exactly. Run folders
from older versions (with the former `save_steps` setting) fail to load on
purpose: there is no compatibility shim.

### Files

`run` writes its files to
`<output_dir>/<mixture>_<box_length>nm_<end_time>ps_<hash>/`, e.g.
`1Water_1Ethanol_2nm_5ps_c3acea28/`. The hash is the first 8 hex digits of
the SHA-256 of all settings except `output_dir` (`config_hash`; `config.json`
holds the settings themselves). So running the same settings again
overwrites the previous results, and any other run, one with another seed or
recording setting included, gets its own folder. Ints count as floats
(`box_length=2` is `box_length=2.0`), and the built-in `molecule_dir` is left out
of the hash, so it does not depend on where the package is installed. Any
`trajectories/` folder is gitignored.

| File | Written by | Contents |
|---|---|---|
| `trajectory.xyz` | `run` | Trajectory: frame 0 is the starting configuration (after the energy minimisation), then one frame every `trajectory_interval` steps, with whole molecules wrapped into the box. |
| `results.npz`, `config.json` | `run` | The recorded data and the settings, read by `Results.load`. The phase-space data takes most of the space. |
| `energy.png` | `plot_energy(save=True)` | Bond, angle, LJ, dihedral, kinetic and total energy over time. |
| `temperature.png` | `plot_temperature(save=True)` | Temperature over time. |
| `phase_space/phase_space_<bond>.png` | `plot_phase_space(save=True)` | Bond length against its rate of change, for each tracked bond. |
| `RDF/RDF_t=<time>.png` | `plot_rdf(save=True)` | The running-mean RDF up to that time. |
| `RDF/RDF.gif`, `RDF/<pair>/RDF_<pair>.gif` | `animate_rdf(save=True)` | Evolution of the RDFs (running mean, with confidence interval per pair), plus first/last frame PNGs. |
| `timings.png` | `plot_timings(save=True)` | Where the computation time went. |
| `view_in_vmd.tcl` | `view_in_vmd()` | VMD script that shows the trajectory with the box and its periodic images. |

With the defaults, the notebook's run (2 nm box, 5 ps) writes an 11 MB
trajectory, and the default run (4 nm, 10 ps) about 170 MB. `results.npz` is
mostly phase-space data, at most about 30 MB at the default
`phase_space_window`.

## 5. Viewing in VMD

[VMD](https://www.ks.uiuc.edu/Research/vmd/) is optional.
`results.view_in_vmd()` ([`vmd.py`](../src/jostle/vmd.py)) writes a Tcl script
to the run folder and launches VMD with it (`Popen`, so VMD runs in its own
window and the notebook stays usable). The script shows the trajectory on a
white background, in licorice style coloured by element, with the bonds of the
topology. Depth cueing fades atoms further from the camera. The 26 periodic
images of the box show how the system repeats (`show_periodic_images=False` leaves them
out, which is lighter for large boxes). The images are outlined in dashed grey
(`show_image_outlines=False` leaves that out) and the box itself in red
(`show_box=False`).

jostle looks for VMD in this order (`jostle.find_vmd()` shows the result):

1. `view_in_vmd(vmd_path=...)`;
2. the `JOSTLE_VMD` environment variable, which sets it once per machine;
3. a `vmd` command on the PATH (the usual Linux and macOS install);
4. the standard install folders: `C:\Program Files\University of Illinois\VMD*`,
   `C:\Program Files\VMD*` and `%LOCALAPPDATA%\Programs\VMD*` on Windows
   (newest first), `/Applications/VMD*.app` on macOS, and `/opt/vmd*` on
   Linux.

A path given in 1 or 2 is used as is: if it is wrong, nothing else is tried.
To reopen a finished run without Python, run `vmd -e view_in_vmd.tcl` in its
folder, or use *File > Load Visualization State* in VMD. Tested with VMD
2.0.0a6 on Windows 11.

## 6. Model

Internal units: **nm, ps, amu, kJ/mol and K**.

### Force field

The potential energy is the sum of four terms
([`forcefield.py`](../src/jostle/forcefield.py)):

- **Bonds** (harmonic): $E = \tfrac12 k_b (r - r_0)^2$
- **Angles** (harmonic): $E = \tfrac12 k_\theta (\theta - \theta_0)^2$
- **Dihedrals** (Ryckaert–Bellemans):
  $E = \tfrac12\left[C_1(1+\cos\phi) + C_2(1-\cos 2\phi) + C_3(1+\cos 3\phi) + C_4(1-\cos 4\phi)\right]$
- **Lennard-Jones**: $E = 4\varepsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right]$ for $r$ below the cutoff $r_c$ = `lj_cutoff` × σ.
  - Shifted force: the force $F(r)$ becomes $F(r) - F(r_c)$ and the energy
    $E(r) - E(r_c) + (r - r_c)F(r_c)$, so both reach zero at the cutoff. This
    removes the jumps when pairs cross it, at the price of slightly weaker
    attraction at every distance.
  - Lorentz–Berthelot mixing: $\sigma_{ij} = (\sigma_i+\sigma_j)/2$,
    $\varepsilon_{ij} = \sqrt{\varepsilon_i\varepsilon_j}$.
  - No LJ between atoms that share a bond, angle or dihedral (1-2, 1-3 and
    1-4 pairs), for atoms with ε = 0, or for pairs where both σ = 0.
  - Distances use the minimum image convention.

### LJ energy correction

The dynamics use the shifted-force potential $E_\text{SF}$, but each energy
frame also records `lj_correction`, so that `LJ + lj_correction` estimates
the energy of the full LJ potential ([`forcefield.py`](../src/jostle/forcefield.py)).
It has two parts:

- **Shift** (exact, summed in `lj_kernel` over the pairs inside the cutoff):
  $E(r) - E_\text{SF}(r) = E(r_c) - (r - r_c)F(r_c)$. Both terms are
  negative: the attraction the shift took away.
- **Tail** (analytic, `ForceField.lj_tail_energy`): the LJ energy beyond the
  cutoffs, assuming a uniform fluid there ($g(r) = 1$). Summed over ordered
  pairs of LJ atom types $a, b$ with counts $N_a$, $N_b$:

  $$E_\text{tail} = \frac{2\pi}{V}\sum_{a,b} N_a N_b \int_{r_c}^{\infty} r^2 E_{ab}(r)\,\mathrm{d}r = \frac{8\pi}{3V}\sum_{a,b} N_a N_b\,\varepsilon_{ab}\sigma_{ab}^3\left[\tfrac13\left(\tfrac{\sigma_{ab}}{r_c}\right)^9 - \left(\tfrac{\sigma_{ab}}{r_c}\right)^3\right]$$

  It is a constant, as $N$ and $V$ are, and 0 for an infinite cutoff or
  without a box.

In the notebook's 2 nm box the correction is about −900 kJ/mol (tail −274,
shift about −630), a few kJ/mol per molecule. It changes only the reported
energy: the forces stay those of the shifted-force potential (a uniform
medium beyond the cutoff exerts no net force anyway), so the trajectory, RDFs
and phase space are the same with or without it. Pressure is not computed, so
there is no pressure tail correction.

### Integration

Euler, Verlet, velocity Verlet or Beeman
([`integrators.py`](../src/jostle/integrators.py)). Verlet and Beeman need the
previous step, so the first step is a velocity Verlet step, which needs only
the starting positions, velocities and accelerations (Euler runs stay Euler).
This adds no energy offset.

The time step is limited by the fastest motion, the X–H bond vibrations
(about 9 fs for O–H). At 1 fs their integration error dominates the
total-energy fluctuations, so the shifted-force cutoff barely changes them.
At the default 0.25 fs it does: std 0.23 vs 0.39 kJ/mol in the 2 nm box.

With `thermostat`, all velocities are rescaled to the target temperature
after every step (an isokinetic thermostat). Verlet has no explicit velocity,
so it rescales its last displacement instead. The temperature uses
$N_f = 3N - 3$ degrees of freedom.

### Setup

- **Placement.** The number of molecules of each type comes from the box
  volume, the weights and the pure-component molar volumes in
  [`constants.py`](../src/jostle/constants.py). Molecules start on random
  sites of a cubic grid, with random orientations.
- **Energy minimisation** ([`minimise.py`](../src/jostle/minimise.py)).
  On the grid, LJ atoms of neighbouring molecules can nearly coincide.
  Steepest descent on the potential energy of the run's force field removes
  these overlaps: every atom moves along its force, with an adaptive step
  (a move is kept only if the energy drops). The bonded forces carry whole
  molecules along, so they translate and rotate, deforming only slightly.
- **Stop criterion.** The descent stops once the largest force on any atom
  is below the thermal force of the stiffest bond,
  $F_T = \sqrt{k_B T\,k_\text{bond}}$: about 1100 kJ/mol/nm at 298 K
  (water's O–H). For a harmonic bond $E = \tfrac12 k x^2$, Boltzmann
  statistics give $\langle x^2\rangle = k_B T/k$, so the force $F = -kx$
  has $\langle F^2\rangle = k_B T\,k$: $F_T$ is its root mean square (rms).
  This is the bond part of the exact identity
  $\langle|\nabla U|^2\rangle = k_B T\,\langle\nabla^2 U\rangle$, which
  defines the configurational temperature (Rugh 1997; Butler et al. 1998);
  the bonds, the stiffest terms, dominate it. In the 298 K liquid the rms
  atom force is indeed about 1100 kJ/mol/nm (largest about 3900). Stopping
  on the largest force rather than the rms keeps any badly overlapping
  atoms from surviving, at the cost of a slightly cold start. The criterion
  sets the
  scale, not the exact starting temperature. Without a thermostat (3 nm
  mixture, 2 ps), stopping at $F_T$ dips briefly to about 180 K and settles
  at about 295 K. Stopping at $10F_T$ leaves overlaps that heat the box to
  about 950 K, and at $F_T/10$ the box stays near 190 K: with almost no
  potential energy left, the kinetic energy is shared with the bonds.
- **Starting velocities.** Maxwell–Boltzmann, with the centre-of-mass
  momentum removed, scaled to the initial temperature.

### Phase space

For each tracked bond type, the bond length $D$ and its exact rate of change
$\dot D = \hat{\mathbf d} \cdot (\mathbf v_j - \mathbf v_i)$ (from the
velocities, not finite differences) are recorded every step over the last
`phase_space_window` ps, for at most 100 bonds per type spread through the box.

### RDF

For each pair of interest, the distances between atoms in different molecules
are counted in bins of `rdf_bin_width`, up to half the box edge. The counts are
normalised to the radial distribution function

$$g(r) = \frac{n(r)}{N_\text{pairs}\, V_\text{shell}(r) / V_\text{box}},$$

the ratio to the counts of the same number of pairs spread uniformly over the
box. So g = 1 means no structure, and g(r) tends to 1 at large r.

The plots show its running mean from the start up to a time, with a 90%
confidence interval from block averaging. Successive frames are correlated,
so they are grouped into consecutive blocks of `block_time` (default 1 ps),
and the spread of the block means gives the standard error (with Student's t,
as there may be few blocks). The interval only appears once two blocks are
complete, and is recomputed for each plot. A `block_time` shorter than the
correlation time gives intervals that are too narrow.

### Deliberate approximations

These are modelling choices, not bugs:

- **No electrostatics.** There are no partial charges, so hydrogen bonding
  is essentially absent. Water's H atoms have no LJ either.
- **LJ cutoff.** The forces are those of the shifted-force potential: zero
  beyond the cutoff, and slightly less attractive inside it. Only the
  reported energy is corrected for this ([LJ energy correction](#lj-energy-correction)).
- **Thermostat.** Rescaling every step gives an isokinetic ensemble: the
  temperature does not fluctuate, so it is not canonical.
- **RDF range.** g(r) only reaches half the box edge (the minimum image),
  and in a small box its long-range part is noisy.

## 7. Molecules and input files

Each molecule needs three things:

- a `<Name>.xyz` geometry;
- a `<Name>.itp` force field, in the format below;
- an entry in `MOLAR_VOLUMES` in [`constants.py`](../src/jostle/constants.py),
  in cm³/mol. Water and ethanol are at 298.15 K.

jostle is built for water and ethanol. Oxygen and hydrogen are also shipped,
as test molecules: O₂ is a molecule with only a bond (no angles), H₂ has very
different LJ parameters. Their molar volumes, O₂ 28.0 and H₂ 28.5, are those
of the liquids at their boiling points, so they are packed at liquid density.

Built-in molecules live in
[`src/jostle/data/molecules/`](../src/jostle/data/molecules). To add your own,
put the two files in a folder and set `molecule_dir` in the `Config`.

When a molecule is loaded (`system.load_molecule` → `check_molecule`), its two
files are checked against each other. They must have the same number of
atoms, and every index in the `.itp` must be a valid atom. Each xyz bond
length must be within 20% of its `r0` (`BOND_TOLERANCE`), and each angle
within 20° of its `theta0` (`ANGLE_TOLERANCE`). The geometry doesn't have to
be at equilibrium (the built-in files are up to 7% and 3.4° off): the check
catches atom-order and unit mistakes. Dihedrals aren't checked.

### .itp format

The format is specific to this project (it is not GROMACS). Atom indices
start at 0:

```
bonds N        then N lines:  i j k_b r0             (k_b in kJ/(mol·Å²), r0 in Å)
angles N       then N lines:  i j k k_theta theta0   (kJ/(mol·rad²), rad; j is the middle atom)
dihedrals N    then N lines:  i j k l C1 C2 C3 C4    (kJ/mol)
atoms N        then N lines:  epsilon sigma mass     (kJ/mol, nm, amu)
```

### Unit conversions on input

The input files use other units than the internal ones; [`molecule_files.py`](../src/jostle/molecule_files.py)
converts them:

- `.xyz`: Å (×0.1 on read, ×10 on write). `read_xyz` returns float32 on
  purpose, to keep results bit-identical with earlier versions.
- `.itp` bonds: `k_b` ×100 → kJ/(mol·nm²), `r0` Å ×0.1 → nm (stored as `Topology.k_bond` and `bond_r0`).
- `.itp` angles: radians and kJ/(mol·rad²), unchanged.
- `.itp` atoms: ε in kJ/mol, σ **already in nm**, mass in amu.
- Dihedral C1–C4: kJ/mol, unchanged.

## 8. Code organisation

```
notebooks/
  jostle.ipynb       the example notebook (start here)
src/jostle/
  config.py         SystemSettings / NumericsSettings / RecordingSettings (each
                    validate()) and Config (groups + run control; flat-keyword
                    routing, validate() incl. largest LJ cutoff ≤ L/2, replace(),
                    to_dict/from_dict)
  constants.py      BOLTZMANN_CONSTANT, AVOGADRO, MOLAR_VOLUMES (needed per molecule!),
                    BUILTIN_MOLECULE_DIR
  data/molecules/   <Name>.xyz + <Name>.itp per molecule
  molecule_files.py read_xyz (float32) / write_xyz(open file, atoms,
                    positions, time) / read_itp → dict
  system.py         Topology dataclass (+ concatenate), load_molecule
                    (+ check_molecule), build_initial_box (random grid sites + random
                    orientations), initial_velocities,
                    molecule_counts (molecules per type, shared with checks)
  thermo.py         kinetic_energy, degrees_of_freedom (3N-3, also used by
                    Results.temperature), velocity_scale_factor
  checks.py         JostleWarning, describe_molecule (bond periods per type),
                    check_config(config): size estimate + warnings
  minimise.py       thermal_force (stop criterion), minimise_energy
                    (steepest descent on the potential energy, before the run)
  geometry.py       pair_vectors (@njit, minimum image if box_length ≠ 0),
                    bond_index_map, angle_geometry, dihedral_geometry (one caller
                    each in forcefield), wrap_molecules, periodic_pairs
                    (cKDTree, used by the neighbour list)
  forcefield.py     POTENTIAL_TERMS; bond/angle/dihedral_term → (atom indices,
                    force vectors, energy | None); lj_kernel (@njit, loops over
                    neighbour-list pairs); ForceField(topology, box_length,
                    lj_cutoff, skin=NEIGHBOUR_SKIN): LJ parameters, cutoffs
                    and shifted-force cutoff_energy/cutoff_force per atom-type pair (tables),
                    lj_tail_energy;
                    .compute_forces(positions, compute_energy=False) →
                    ForceEvaluation(forces, bond_vectors, bond_lengths,
                    potential_energies {POTENTIAL_TERMS} | None,
                    lj_correction | None)
  neighbours.py     excluded_pair_keys, filter_lj_pairs, lj_pairs (all pairs),
                    NeighbourList (periodic cKDTree; refresh(positions))
  integrators.py    Integrator(update_positions, update_velocities) records: euler, verlet,
                    velocity_verlet, beeman; INTEGRATORS by name; startup_integrator
                    (self-starting first step); State(x, v, a, x_prev,
                    a_prev, kinetic_energy); initial_state; integration_step
  simulation.py     config_hash, run_folder,
                    first_phase_space_step, run(config)
  observables.py    PhaseSpace (bond length + exact dD/dt; bond_types = the
                    config entries as typed, e.g. 'O-H (Water)'), RDF
                    (accumulate_pair_histogram
                    @njit, same binning as np.histogram; intermolecular pairs
                    only; nr_pairs → g_of_r()); both have their own `times` and
                    from_arrays (for loaded results, which cannot record)
  results.py        Results dataclass with plot_*(save=) methods, animate_rdf,
                    save/load, view_in_vmd
  plotting.py       *_figure builders return Figures (closed in pyplot right
                    away); rdf_statistics (running mean of g(r), CI from block
                    averaging); save_animate_rdf
  vmd.py            find_vmd, write_vmd_script, launch_vmd
tests/              pytest suite (helpers.py has shared fixtures)
docs/images/        README figures, made from the notebook's run (2 nm, 5 ps)
```

## 9. Implementation details

**Data representations.**

- `bonds` (B×2) are stored **sorted** (`i < j`). `angles` (A×3) and
  `dihedrals` (D×4) hold atom indices. The dihedral constants `rb_coefficients` are a
  (D×4) array.
- `bond_index_map` maps each angle and dihedral to rows of `bonds`, plus a ±1
  sign, so the angle and dihedral terms reuse the bond vectors `bond_vectors`
  and lengths `bond_lengths` and never recompute vectors.
- Integrator formulas use short names: `x`/`v`/`a`, with the suffixes
  `_curr`, `_prev` (current and previous step) and `_new`.

**Lennard-Jones and the neighbour list.**

- LJ pairs come from a Verlet neighbour list
  ([`neighbours.py`](../src/jostle/neighbours.py)): a periodic `cKDTree`,
  rebuilt when an atom has moved more than half the skin. If cutoff + skin
  ≥ L/2 or `box_length` = 0 it falls back to all pairs.
- The pairs are `ForceField.pair_i`, `pair_j`, with aligned `pair_epsilon`,
  `pair_sigma`, `pair_cutoff`, `pair_cutoff_energy` and `pair_cutoff_force`, refreshed on every
  rebuild. Exclusions (1-2/1-3/1-4, ε = 0, both σ = 0) are applied there.
  Results equal all-pairs LJ up to rounding (tested).
- LJ forces and RDF histograms run in compiled Numba kernels. They use the
  minimum image `d -= L * floor(d / L + 0.5)`, about 9× faster than `%` in
  Numba.

**Periodic boundaries.** Positions are **not** wrapped during integration.
`wrap_molecules` wraps whole molecules, and only when writing the xyz. Bonds
use `box_length=0` (no minimum image); LJ and the RDF use the minimum image.

**Integration loop.**

- `integration_step(..., target_temperature=T)` is the thermostat: it rescales v every step (for
  Verlet, `x_prev` instead).
- `initial_state` sets `x_prev = x` and `a_prev = a`, so the first step must
  be self-starting: `startup_integrator(integrator)` gives velocity Verlet (Euler for
  Euler runs).
- Energies are only computed on energy steps: `simulation.run` passes a
  `compute_energy=True` force function to `integration_step` there. `State.kinetic_energy` is
  computed once per step, and the thermostat reuses it.

**Randomness.** All randomness flows from one
`np.random.default_rng(config.seed)`. The RDF thinning draws from its own
stream, so RDF settings don't change the trajectory.

## 10. Performance

Measured with the default settings (a 4 nm box of water and ethanol, 8,376
atoms) on a desktop PC: AMD Ryzen 5 2600 (6 cores, 3.4 GHz; the code runs
on a single core), 16 GB RAM, Windows 11, Python 3.14, NumPy 2.5, Numba 0.68.

- **One time step:** about 62 ms with the default 2.5σ shifted-force cutoff
  (neighbour list ~22 ms, LJ ~26 ms, bonded ~11 ms), so about 40 minutes for
  the default 10 ps (40,000 steps). At 2σ: about 41 ms. The original
  all-pairs version took about 2.4 s per step.
- **Energy minimisation:** about 20 s (470 iterations) before the run
  starts (4 nm box).

Only about 3% of all atom pairs are visited each step.

## 11. Development

```bash
pip install -e .[dev]       # pytest and friends
python -m pytest            # ~10 s
ruff check src tests        # linting (pip install ruff)
```

Tests also run without installing: `pyproject.toml` adds `src` and `tests` to
the `pythonpath`. To use the package without installing, put `src` on
`PYTHONPATH`.

**Tests.**

- They check that every force term equals minus the numerical gradient of its
  energy, that the LJ correction gives the full potential (with the tail
  checked against numerical integration), that velocity Verlet conserves
  energy, that the energy minimisation brings the forces down to thermal
  size, that the thermostat reaches its target, that the neighbour list
  matches all-pairs LJ, and that results are reproducible with a seed.
- `tests/helpers.py` has `make_topology`, `forcefield`, `forces_and_energy`
  and `water_ethanol_pair`.
- `test_simulation.py` shares two module-scoped runs (`full_run`, `bare_run`);
  new end-to-end checks go on those rather than adding `run()` calls.
- pytest ignores `JostleWarning` globally (the test runs are tiny);
  `test_checks.py` only checks that the default settings give no warnings.
- The suite is kept to the essentials: physics, the run pipeline, saving and
  loading, and config validation. The notebook and VMD are not tested.

**The notebook** is committed with its outputs (about 2 MB, mostly the RDF
animation), so it can be read on GitHub without running it. Git keeps every
committed version, so commit a re-executed notebook only when its results
change. To re-execute it in place:
`jupyter nbconvert --to notebook --execute --inplace notebooks/jostle.ipynb`.

**Code conventions.**

- Comments are dense and block-style (`# ---- Section ----`). Lines are
  wrapped at 80 characters.
- snake_case, except physics arrays, which keep their original names.
- Vectorised NumPy everywhere; hot loops in Numba.
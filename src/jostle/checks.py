"""Checks of a Config before a run: settings that are valid, but make a
measurement useless or expensive, give an JostleWarning. The expected size of
the output files is always printed. run() calls check_config() before
building the box, so a run that would be wasted can be stopped at once.

The checks only need the config and the molecule files: the number of
molecules of each type follows from the box size and the weights, and the
vibration period of each bond type from its force constant and masses."""
import warnings
from dataclasses import dataclass

import numpy as np

from .constants import TRAJECTORY_FILE
from .observables import PHASE_SPACE_MAX_BONDS, _unique_species_pairs
from .plotting import RDF_BLOCK_TIME
from .system import load_molecule, molecule_counts


class JostleWarning(UserWarning):
    """Settings that are valid, but probably not what was intended."""


# Show these warnings on every run, not only the first time (the default
# filter shows a warning once per code location, so a notebook would show it
# once per session). warnings.filterwarnings can still silence them.
warnings.simplefilter("always", JostleWarning)

# ---- Thresholds ----
MIN_STEPS_PER_PERIOD = 20   # dt: energy drifts with fewer per bond vibration
MIN_RDF_INTERVAL = 0.01     # ps: closer RDF frames are nearly identical
MIN_FRAMES_PER_STREAM = 10  # fewer frames make a meaningless plot
LARGE_FILE_BYTES = 1e9      # bytes
# '<atom:2><x:16><y:16><z:16>\n' (molecule_files.write_xyz)
XYZ_BYTES_PER_ATOM = 51
XYZ_HEADER_BYTES = 40       # atom count and comment line


@dataclass
class MoleculeInfo:
    """What the checks need to know about one molecule type."""
    elements: set            # element symbols
    bond_periods: dict       # 'H-O' (sorted atoms) -> shortest period (ps)
    nr_atoms: int
    bond_counts: dict        # 'H-O' -> bonds of that type per molecule
    bond_labels: dict        # 'H-O' -> 'O-H', in the order of the .itp


def describe_molecule(name, molecule_dir):
    """Elements and bond vibration periods of a molecule, from its files. The
    period of a harmonic bond is 2 pi / sqrt(k_bond / mu), with mu the reduced
    mass of its atoms (the angles and other bonds shift it a little)."""
    _, top = load_molecule(name, molecule_dir)
    periods, counts, labels = {}, {}, {}
    for (i, j), k_bond in zip(top.bonds, top.k_bond, strict=True):
        bond = '-'.join(sorted((top.elements[i], top.elements[j])))
        labels.setdefault(bond, f"{top.elements[i]}-{top.elements[j]}")
        mu = top.masses[i] * top.masses[j] / (top.masses[i] + top.masses[j])
        period = 2 * np.pi / np.sqrt(k_bond / mu)
        periods[bond] = min(period, periods.get(bond, np.inf))
        counts[bond] = counts.get(bond, 0) + 1
    return MoleculeInfo(set(top.elements), periods, top.nr_atoms, counts,
                        labels)


def _format_size(nr_bytes):
    """Bytes in kB, MB or GB, to two significant digits."""
    unit, size = next(((unit, size) for unit, size in
                       (("GB", 1e9), ("MB", 1e6)) if nr_bytes >= size),
                      ("kB", 1e3))
    return f"{float(f'{nr_bytes / size:.2g}'):g} {unit}"


def check_config(config):
    """Prints the expected size of the output files, and warns about
    settings that would waste the run or a measurement. Returns the warning
    messages."""
    from .simulation import first_phase_space_step

    system, numerics = config.system, config.numerics
    recording = config.recording
    counts = molecule_counts(system.box_length, system.weights)
    molecule_infos = {m: describe_molecule(m, system.molecule_dir)
                      for m in counts}
    dt, nr_steps = numerics.dt, config.nr_steps
    messages = []

    def nr_frames(interval):
        return 1 + nr_steps // interval

    # ---- Requested bond types and pairs: present, absent or a typo ----
    def has_molecule_files(molecule):
        return (system.molecule_dir / f"{molecule}.itp").exists()

    tracked = {}      # phase space entry -> (molecule, bond)
    for entry, species in _unique_species_pairs(recording.phase_space_bonds,
                                                '-'):
        (atom1, molecule), (atom2, _) = species
        bond = f"{atom1}-{atom2}"
        if molecule in molecule_infos:
            if bond in molecule_infos[molecule].bond_periods:
                tracked[entry] = (molecule, bond)
            else:
                messages.append(
                    f"phase_space_bonds: {molecule} has no {bond} bond, so "
                    f"{entry!r} records nothing (a typo?).")
        elif not has_molecule_files(molecule):
            messages.append(f"phase_space_bonds: there are no files for a "
                            f"molecule {molecule!r} in molecule_dir, so "
                            f"{entry!r} records nothing (a typo?).")
    rdf_entries = []
    for entry, species in _unique_species_pairs(recording.rdf_pairs, '~'):
        unknown = [m for _, m in species
                   if m not in molecule_infos and not has_molecule_files(m)]
        missing = [f"{atom} in {m}" for atom, m in species
                   if m in molecule_infos
                   and atom not in molecule_infos[m].elements]
        if unknown:
            messages.append(f"rdf_pairs: there are no files for a molecule "
                            f"{' or '.join(map(repr, unknown))} in "
                            f"molecule_dir, so {entry!r} records nothing "
                            f"(a typo?).")
        elif missing:
            messages.append(
                f"rdf_pairs: there is no {' and no '.join(missing)}, so "
                f"{entry!r} records nothing (a typo?).")
        elif all(m in molecule_infos for _, m in species):
            rdf_entries.append(entry)

    # ---- Expected output size ----
    nr_atoms = sum(counts[m] * molecule_infos[m].nr_atoms for m in counts)
    xyz_frames = nr_frames(recording.trajectory_interval)
    xyz_bytes = xyz_frames * (nr_atoms * XYZ_BYTES_PER_ATOM
                              + XYZ_HEADER_BYTES)
    phase_space_frames = nr_steps + 1 - first_phase_space_step(config)
    phase_space_bonds = sum(
        min(counts[m] * molecule_infos[m].bond_counts[b],
            PHASE_SPACE_MAX_BONDS) for m, b in tracked.values())
    nr_bins = int(system.box_length / 2 / recording.rdf_bin_width)
    # float64 arrays: phase space (lengths and rates), RDF histograms, the
    # energy terms, kinetic energy and times. Stored compressed, so this is
    # an upper bound.
    npz_bytes = 8 * (2 * phase_space_bonds * phase_space_frames
                     + (nr_frames(recording.rdf_interval) * len(rdf_entries)
                        * nr_bins)
                     + 7 * nr_frames(recording.energy_interval))
    print(f"Expected output: {TRAJECTORY_FILE} {_format_size(xyz_bytes)} "
          f"({xyz_frames} frames, {nr_atoms} atoms), results.npz up to "
          f"{_format_size(npz_bytes)}.")
    for name, size in ((TRAJECTORY_FILE, xyz_bytes),
                       ("results.npz", npz_bytes)):
        if size > LARGE_FILE_BYTES:
            setting = ("trajectory_interval" if name == TRAJECTORY_FILE
                       else "rdf_interval or phase_space_window")
            messages.append(f"{name} will take about {_format_size(size)}; a "
                            f"larger {setting} makes it smaller.")

    # ---- Time step and integrator ----
    periods = {f"{molecule_infos[m].bond_labels[b]} ({m})": period
               for m in molecule_infos
               for b, period in molecule_infos[m].bond_periods.items()}
    fastest = min(periods, key=periods.get)
    period = periods[fastest]
    if dt > period / np.pi:
        messages.append(
            f"dt = {dt * 1000:g} fs is unstable: the {fastest} bond vibrates "
            f"every {period * 1000:.1f} fs, and velocity Verlet needs "
            f"dt < period / pi ({period / np.pi * 1000:.1f} fs). Use dt <= "
            f"{period / MIN_STEPS_PER_PERIOD * 1000:.2g} fs.")
    elif dt > period / MIN_STEPS_PER_PERIOD:
        messages.append(
            f"dt = {dt * 1000:g} fs gives only {period / dt:.0f} steps per "
            f"vibration of the {fastest} bond ({period * 1000:.1f} fs): the "
            f"energy is poorly conserved and the phase space orbits are "
            f"jagged. Use dt <= {period / MIN_STEPS_PER_PERIOD * 1000:.2g} "
            f"fs.")
    if numerics.integrator == "euler":
        messages.append("integrator='euler' does not conserve energy (it "
                        "drifts every step); use it only for comparison.")

    # ---- Measurements ----
    frame_streams = [
        ("energy_interval", "energy", recording.energy_interval),
        ("trajectory_interval", "trajectory", recording.trajectory_interval)]
    if rdf_entries:
        frame_streams.append(("rdf_interval", "RDF", recording.rdf_interval))
    for setting, label, interval in frame_streams:
        if nr_frames(interval) < MIN_FRAMES_PER_STREAM:
            messages.append(
                f"Only {nr_frames(interval)} {label} frames are recorded; a "
                f"smaller {setting} (now {interval}) or a longer end_time "
                f"gives more.")
    if rdf_entries:
        if recording.rdf_interval * dt < MIN_RDF_INTERVAL:
            messages.append(
                f"RDF frames {recording.rdf_interval * dt * 1000:g} fs apart "
                f"are nearly identical: they cost time without improving "
                f"g(r). rdf_interval >= "
                f"{np.ceil(MIN_RDF_INTERVAL / dt):.0f} is enough.")
        if config.end_time < 2 * RDF_BLOCK_TIME:
            messages.append(
                f"The RDF confidence intervals need two blocks of "
                f"{RDF_BLOCK_TIME:g} ps; end_time = {config.end_time:g} ps "
                f"gives none (a smaller block_time in plot_rdf/animate_rdf does).")
    if tracked:
        def bond_period(entry):
            molecule, bond = tracked[entry]
            return molecule_infos[molecule].bond_periods[bond]

        slowest = max(tracked, key=bond_period)
        period = bond_period(slowest)
        window = (phase_space_frames - 1) * dt
        if window < period:
            messages.append(
                f"The phase space covers {window * 1000:g} fs, less than one "
                f"vibration of the {slowest} bond ({period * 1000:.1f} fs): "
                f"use a longer phase_space_window.")

    for message in messages:
        warnings.warn(message, JostleWarning, stacklevel=2)
    return messages

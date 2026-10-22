"""Simulation settings: defaults and validation.

A Config holds three groups of settings plus run control:
- system: what is simulated (composition, box, temperature, ensemble);
- numerics: how it is approximated (integrator, time step, LJ cutoff);
- recording: what is measured during the run.
Plotting is not configured here: Results makes figures on request. Numerical
constants that need no tuning (packing, neighbour list skin, RDF confidence)
are module constants next to the code that uses them.
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .constants import BUILTIN_MOLECULE_DIR, MOLAR_VOLUMES


@dataclass
class SystemSettings:
    """What is simulated. Units: nm, K."""
    # Relative amounts of each molecule; normalised, so any positive scale
    # works. Built-in: Water, Ethanol, Oxygen, Hydrogen.
    weights: dict = field(default_factory=lambda: {"Water": 1.0,
                                                   "Ethanol": 1.0})
    # Edge length of the cubic periodic box (nm)
    box_length: float = 4.0
    temperature: float = 298.15  # initial (and target) temperature (K)
    # Rescale velocities to temperature every step (NVT); False: NVE
    thermostat: bool = True
    # Folder with <Name>.xyz and <Name>.itp files (default: the built-in ones)
    molecule_dir: Path = BUILTIN_MOLECULE_DIR

    def __post_init__(self):
        self.molecule_dir = Path(self.molecule_dir)

    def validate(self):
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("All weights must be >= 0.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("The sum of weights must be > 0.")
        # Assert the xyz and itp data exists
        for mol, w in self.weights.items():
            if w == 0:
                continue
            for ext in ("xyz", "itp"):
                path = self.molecule_dir / f"{mol}.{ext}"
                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist.")
            if mol not in MOLAR_VOLUMES:
                raise ValueError(f"No molar volume known for {mol}; add it "
                                 f"to MOLAR_VOLUMES in constants.py.")
        for name in ("box_length", "temperature"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0.")
        if not isinstance(self.thermostat, bool):
            raise ValueError("thermostat must be a boolean.")

    def largest_lj_sigma(self):
        """Largest sigma (nm) of the LJ atoms (epsilon > 0) in the system;
        with Lorentz-Berthelot mixing, no pair has a larger sigma."""
        from .molecule_files import read_itp

        sigmas = [0.]
        for mol, w in self.weights.items():
            if w > 0:
                itp = read_itp(self.molecule_dir / f"{mol}.itp")
                sigmas.extend(itp["lj_sigma"][itp["lj_epsilon"] > 0])
        return max(sigmas)


@dataclass
class NumericsSettings:
    """How the system is approximated. Units: ps."""
    # euler, verlet, velocity_verlet or beeman
    integrator: str = "velocity_verlet"
    dt: float = 0.00025           # time step (ps)
    # LJ cutoff as a multiple of sigma (shifted force: the force goes to 0
    # at the cutoff). The largest cutoff must fit in half the box.
    lj_cutoff: float = 2.5

    def validate(self):
        from .integrators import INTEGRATORS

        if self.integrator not in INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {sorted(INTEGRATORS)}.")
        for name in ("dt", "lj_cutoff"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0.")


@dataclass
class RecordingSettings:
    """What is measured during the run. Temperature, computation times and
    the trajectory are always recorded."""
    # Each measurement is recorded every this many steps, plus the start.
    # Energies, kinetic energy and temperature (5 fs at the default dt):
    energy_interval: int = 20
    # Trajectory frames (25 fs; the default run writes 400 frames)
    trajectory_interval: int = 100
    # RDF histograms (25 fs; 40 frames per 1 ps block of the RDF statistics).
    # Closer frames are nearly identical: they cost time, not accuracy.
    rdf_interval: int = 100
    # Potential energies per term (costs a little time on energy steps)
    record_potential_energies: bool = True
    # Bond types for the phase space diagrams: "<atom>-<atom> (<molecule>)".
    # An empty list records none.
    phase_space_bonds: list = field(default_factory=lambda: [
        'O-H (Water)', 'O-H (Ethanol)', 'C-O (Ethanol)', 'C-H (Ethanol)',
        'O-O (Oxygen)'])
    # The phase space is recorded every step (about 34 points per O-H
    # vibration at the default dt), but only over the last phase_space_window
    # ps of the run: the orbits repeat, so a short window shows them all.
    # None (or a time longer than the run) records the whole run.
    phase_space_window: float | None = 1.0
    # Pairs for the RDF: "<atom>~<atom> (<molecule>~<molecule>)", where the
    # first atom belongs to the first molecule. An empty list records none.
    rdf_pairs: list = field(default_factory=lambda: [
        'O~H (Water~Ethanol)', 'O~H (Ethanol~Ethanol)', 'O~H (Water~Water)',
        'O~O (Water~Water)', 'O~O (Water~Ethanol)', 'O~O (Ethanol~Ethanol)'])
    # Fraction of pairs to sample (0 < fraction <= 1)
    rdf_pair_fraction: float = 1.0
    rdf_bin_width: float = 0.005  # histogram bin width (nm)

    def validate(self):
        from .observables import _parse_species_pair

        for name in ("energy_interval", "trajectory_interval", "rdf_interval"):
            steps = getattr(self, name)
            if not (isinstance(steps, int) and steps >= 1):
                raise ValueError(f"{name} must be an integer >= 1.")
        if not (self.phase_space_window is None or self.phase_space_window > 0):
            raise ValueError("phase_space_window must be None or > 0.")
        if not isinstance(self.record_potential_energies, bool):
            raise ValueError("record_potential_energies must be a boolean.")
        # Fail on a malformed entry now rather than after building the box
        for entries, separator in ((self.phase_space_bonds, '-'),
                                   (self.rdf_pairs, '~')):
            for entry in entries:
                _parse_species_pair(entry, separator)
        if self.rdf_pairs:
            if not 0 < self.rdf_pair_fraction <= 1:
                raise ValueError("rdf_pair_fraction must be > 0 and <= 1.")
            if not self.rdf_bin_width > 0:
                raise ValueError("rdf_bin_width must be > 0.")


GROUPS = {"system": SystemSettings, "numerics": NumericsSettings,
          "recording": RecordingSettings}


@dataclass(init=False)
class Config:
    """All simulation settings, with their defaults. Units: nm, ps, K.

    Settings are grouped (config.system.box_length, config.numerics.dt, ...),
    but can be given by name: Config(box_length=2.0, dt=0.001, seed=1). Groups
    can also be passed whole: Config(system=SystemSettings(...)). Change a
    copy with config.replace().
    """
    system: SystemSettings
    numerics: NumericsSettings
    recording: RecordingSettings
    end_time: float     # total simulated time (ps)
    seed: int | None    # random seed; None: every run is different
    output_dir: Path    # where run folders are created (relative to the
                        # working directory)

    def __init__(self, system=None, numerics=None, recording=None,
                 end_time=10.0, seed=None, output_dir="trajectories",
                 **settings):
        groups = dict(system=system, numerics=numerics, recording=recording)
        # Each setting given by name goes to the group that has it
        for group, cls in GROUPS.items():
            names = {f.name for f in fields(cls)}
            given = {k: settings.pop(k) for k in list(settings) if k in names}
            base = groups[group] if groups[group] is not None else cls()
            groups[group] = cls(**{**asdict(base), **given})
        if settings:
            raise TypeError(f"Unknown setting(s): {', '.join(settings)}.")
        self.system = groups["system"]
        self.numerics = groups["numerics"]
        self.recording = groups["recording"]
        self.end_time = end_time
        self.seed = seed
        self.output_dir = Path(output_dir)

    @property
    def nr_steps(self):
        """Number of integration steps of the run."""
        return int(np.ceil(self.end_time / self.numerics.dt))

    def replace(self, **changes):
        """A validated copy with some settings changed (by name or as whole
        groups)."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        return Config(**{**current, **changes}).validate()

    def to_dict(self):
        """The settings as plain values (paths as strings), e.g. for JSON."""
        d = asdict(self)
        d["system"]["molecule_dir"] = str(self.system.molecule_dir)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        groups = {g: GROUPS[g](**d.pop(g)) for g in GROUPS}
        return cls(**groups, **d)

    def validate(self):
        """Validates all given parameters to prevent any weird user inputs."""
        for group in GROUPS:
            getattr(self, group).validate()
        if not self.end_time > 0:
            raise ValueError("end_time must be > 0.")
        # Minimum image: no pair may interact with two images of an atom
        largest = self.numerics.lj_cutoff * self.system.largest_lj_sigma()
        if largest > self.system.box_length / 2:
            raise ValueError(
                f"The largest LJ cutoff ({largest:.3f} nm = lj_cutoff "
                f"* largest sigma) must be at most half the box length; use a "
                f"larger box or a smaller lj_cutoff.")
        return self

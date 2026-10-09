"""Simulation settings: defaults, loading from TOML and validation."""
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .constants import MOLAR_VOLUMES, MOLECULE_DIR


@dataclass
class Config:
    """All simulation settings. See configs/default.toml for documentation
    of each setting. None means 'not set'."""
    # [system]
    weights: dict = field(default_factory=lambda: {"Water": 1.0,
                                                   "Ethanol": 1.0})
    boxsize: float = 4.0
    random_grid_offset: float = 0.0
    r_threshold_scalar: float | None = 0.65
    molecule_dir: Path = MOLECULE_DIR
    seed: int | None = None

    # [integration]
    method: str = "velocity_verlet"
    dt: float = 0.001
    end_time: float = 200.0
    save_steps: int = 100
    initial_temp: float = 298.15
    temp_control: bool = True

    # [lennard_jones]: exactly one of the two must be set
    r_cutoff_scalar: float | None = 2.0
    f_threshold: float | None = None
    neighbour_skin: float = 0.1

    # [output]
    output_dir: Path = Path("trajectories")
    show_plots: bool = True
    energy_graph: bool = True
    temp_graph: bool = True
    phase_space_diagram: bool = True
    comptime_vis: bool = True
    RDF_vis: bool = True
    VMD_vis: bool = False
    vmd_path: str = r"C:\Program Files\VMD\vmd.exe"
    bonds_of_interest: list = field(default_factory=lambda: [
        'O-H (Water)', 'O-H (Ethanol)', 'C-O (Ethanol)', 'C-H (Ethanol)',
        'O-O (Oxygen)'])
    RDF_thin: float = 1.0
    RDF_dr: float = 0.005
    RDF_gif_frames: int = 200
    pairs_of_interest: list = field(default_factory=lambda: [
        'O~H (Water~Ethanol)', 'O~H (Ethanol~Ethanol)', 'O~H (Water~Water)',
        'O~O (Water~Water)', 'O~O (Water~Ethanol)', 'O~O (Ethanol~Ethanol)'])

    def validate(self):
        """Validates all given parameters to prevent any weird user inputs."""
        from .integrators import METHODS

        if any(w < 0 for w in self.weights.values()):
            raise ValueError("All weights must be >= 0.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("The sum of weights must be > 0.")

        # Assert the xyz and itp data exists
        for mol, w in self.weights.items():
            if w == 0:
                continue
            for ext in ("xyz", "itp"):
                path = Path(self.molecule_dir) / f"{mol}.{ext}"
                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist.")
            if mol not in MOLAR_VOLUMES:
                raise ValueError(f"No molar volume known for {mol}; add it "
                                 f"to MOLAR_VOLUMES in constants.py.")

        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}.")
        if not (isinstance(self.save_steps, int) and self.save_steps >= 1):
            raise ValueError("save_steps must be an integer >= 1.")
        for name in ("boxsize", "dt", "end_time", "initial_temp"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0.")
        if not -0.5 <= self.random_grid_offset <= 0.5:
            raise ValueError("random_grid_offset must be between -0.5 and "
                             "0.5.")
        if self.r_threshold_scalar is not None and self.r_threshold_scalar <= 0:
            raise ValueError("r_threshold_scalar must be > 0.")

        # Ensure only one of f_threshold or r_cutoff_scalar is specified
        if (self.f_threshold is None) == (self.r_cutoff_scalar is None):
            raise ValueError("Exactly one of f_threshold or r_cutoff_scalar "
                             "must be specified.")
        if self.r_cutoff_scalar is not None and self.r_cutoff_scalar < 0:
            raise ValueError("r_cutoff_scalar must be >= 0.")
        if self.f_threshold is not None and self.f_threshold < 0:
            raise ValueError("f_threshold must be >= 0.")
        if not self.neighbour_skin > 0:
            raise ValueError("neighbour_skin must be > 0.")

        for f in fields(self):
            if f.type is bool and not isinstance(getattr(self, f.name), bool):
                raise ValueError(f"{f.name} must be a boolean.")

        if self.phase_space_diagram and not self.bonds_of_interest:
            raise ValueError("bonds_of_interest must be a list with at least "
                             "one element when phase_space_diagram is True.")
        if self.RDF_vis:
            if not self.pairs_of_interest:
                raise ValueError("pairs_of_interest must be a list with at "
                                 "least one element when RDF_vis is True.")
            if not 0 < self.RDF_thin <= 1:
                raise ValueError("RDF_thin must be > 0 and <= 1.")
            if self.RDF_dr <= 0:
                raise ValueError("RDF_dr must be > 0.")
            if not (isinstance(self.RDF_gif_frames, int)
                    and self.RDF_gif_frames >= 2):
                raise ValueError("RDF_gif_frames must be an integer >= 2.")
        return self


# Where each setting lives in the TOML file: section -> {toml key: field}
TOML_LAYOUT = {
    None: {"seed": "seed"},
    "system": {"boxsize": "boxsize", "weights": "weights",
               "random_grid_offset": "random_grid_offset",
               "r_threshold_scalar": "r_threshold_scalar",
               "molecule_dir": "molecule_dir"},
    "integration": {"method": "method", "dt": "dt", "end_time": "end_time",
                    "save_steps": "save_steps",
                    "initial_temp": "initial_temp",
                    "temp_control": "temp_control"},
    "lennard_jones": {"r_cutoff_scalar": "r_cutoff_scalar",
                      "f_threshold": "f_threshold",
                      "neighbour_skin": "neighbour_skin"},
    "output": {"directory": "output_dir", "show_plots": "show_plots",
               "vmd_path": "vmd_path"},
    "plots": {"energy": "energy_graph", "temperature": "temp_graph",
              "phase_space": "phase_space_diagram",
              "computation_time": "comptime_vis", "rdf": "RDF_vis",
              "vmd": "VMD_vis"},
    "phase_space": {"bonds": "bonds_of_interest"},
    "rdf": {"pairs": "pairs_of_interest", "thin": "RDF_thin",
            "dr": "RDF_dr", "gif_frames": "RDF_gif_frames"},
}


def load_config(path=None, **overrides):
    """Reads a TOML config file on top of the defaults. Paths in the file are
    relative to the file. Keyword overrides are applied last."""
    values = {}
    if path is not None:
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        for section, keys in TOML_LAYOUT.items():
            table = data.pop(section, {}) if section else data
            for key, name in keys.items():
                if key in table:
                    values[name] = table.pop(key)
            if section and table:
                raise ValueError(f"Unknown setting(s) in [{section}]: "
                                 f"{', '.join(table)}")
        unknown = [k for k in data if k not in TOML_LAYOUT[None]]
        if unknown:
            raise ValueError(f"Unknown setting(s): {', '.join(unknown)}")

        # Setting f_threshold in the file replaces the default cutoff mode
        if "f_threshold" in values and "r_cutoff_scalar" not in values:
            values["r_cutoff_scalar"] = None
        for name in ("molecule_dir", "output_dir"):
            if name in values:
                values[name] = path.parent / values[name]

    values.update(overrides)
    for name in ("molecule_dir", "output_dir"):
        if name in values:
            values[name] = Path(values[name])
    return Config(**values).validate()

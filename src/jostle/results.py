"""The outcome of a run: the recorded measurements, with methods to plot them
(each returns a matplotlib Figure, which a notebook shows inline, and saves
it on request), and to save and load them (results.npz + config.json)."""
import json
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .config import Config
from .constants import BOLTZMANN_CONSTANT, TRAJECTORY_FILE
from .forcefield import POTENTIAL_TERMS
from .observables import RDF, PhaseSpace
from .plotting import (
    RDF_BLOCK_TIME,
    energy_figure,
    phase_space_figure,
    rdf_figure,
    rdf_statistics,
    save_rdf_gif,
    temperature_figure,
    timings_figure,
)
from .system import Topology
from .thermo import degrees_of_freedom


@dataclass
class Results:
    """Measurements of one run. energy_times, potential_energies and
    kinetic_energies have one entry per energy frame (every
    config.recording.energy_interval steps, plus the initial state);
    phase_space and rdf have their own times. A measurement that was
    switched off in the config is None.

    Plots are made on request. Each plot_* method (and animate_rdf) takes
    save: False (just return the figure), True (also save it in the run
    folder) or a path."""
    config: Config
    folder: Path                 # where the files of this run were written
    topology: Topology
    energy_times: np.ndarray     # time of each energy frame (ps)
    potential_energies: dict | None  # POTENTIAL_TERMS -> array (kJ/mol)
    kinetic_energies: np.ndarray  # kinetic energy (kJ/mol)
    phase_space: PhaseSpace | None
    rdf: RDF | None
    timings: dict                # label -> seconds spent in the main loop
    wall_time: float             # seconds spent in the main loop

    @property
    def trajectory(self):
        """Path of the trajectory (.xyz, in Angstrom)."""
        return self.folder / TRAJECTORY_FILE

    @property
    def temperature(self):
        """Temperature at each energy frame (K)."""
        return 2 * self.kinetic_energies / (
            BOLTZMANN_CONSTANT * degrees_of_freedom(self.topology.nr_atoms))

    def _recorded(self, value, setting):
        if value is None:
            raise ValueError(f"Not recorded in this run; set {setting} in "
                             f"the Config.")
        return value

    def _save_figure(self, fig, save, default_name, **savefig_kwargs):
        """Saves fig to the run folder (save=True) or to the path save."""
        if save is not False:
            path = (self.folder / default_name if save is True
                    else Path(save))
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, **savefig_kwargs)
        return fig

    # ---- Plots ----
    def plot_energy(self, save=False):
        """Each potential energy term, the kinetic energy and the total over
        time (energy.png)."""
        potential = self._recorded(self.potential_energies,
                                   "record_potential_energies=True")
        fig = energy_figure(self.energy_times,
                            {**potential, "Kinetic": self.kinetic_energies})
        return self._save_figure(fig, save, "energy.png")

    def plot_temperature(self, save=False):
        """Temperature over time (temperature.png)."""
        return self._save_figure(
            temperature_figure(self.energy_times, self.temperature),
            save, "temperature.png")

    def plot_phase_space(self, bond_type=None, save=False):
        """Phase space diagram of one bond type, named as in
        phase_space_bonds (e.g. 'O-H (Water)'), or a list of figures for all
        tracked bond types if bond_type is None. Saved as
        phase_space/phase_space_<bond type>.png; for all bond types, a save
        path is the folder to save them in."""
        phase_space = self._recorded(self.phase_space, "phase_space_bonds")
        if bond_type is None:
            return [self.plot_phase_space(
                        name, save if save in (True, False)
                        else Path(save) / f"phase_space_{name}.png")
                    for name in phase_space.bond_types]
        if bond_type not in phase_space.bond_types:
            raise ValueError(f"Unknown bond type {bond_type!r}; tracked: "
                             f"{', '.join(phase_space.bond_types)}")
        k = phase_space.bond_types.index(bond_type)
        fig = phase_space_figure(bond_type, phase_space.lengths[k],
                                 phase_space.length_rates[k])
        return self._save_figure(
            fig, save, f"phase_space/phase_space_{bond_type}.png",
            bbox_inches='tight')

    def _checked_rdf_statistics(self, pairs, block_time):
        """The recorded RDF and its statistics (see plotting.rdf_statistics),
        after checking that the given pairs (None: all) were recorded."""
        rdf = self._recorded(self.rdf, "rdf_pairs")
        unknown = [p for p in pairs or [] if p not in rdf.pair_labels]
        if unknown:
            raise ValueError(f"Unknown pair(s) {', '.join(unknown)}; "
                             f"recorded: {', '.join(rdf.pair_labels)}")
        return rdf, rdf_statistics(rdf, self._rdf_frame_interval, block_time)

    def plot_rdf(self, time=None, pairs=None, block_time=RDF_BLOCK_TIME,
                 save=False):
        """Running mean g(r) up to the given time (ps; default: the end) of
        the given pairs (default: all recorded ones), saved as
        RDF/RDF_t=<time>.png. The 90% confidence intervals treat blocks of
        block_time (ps) as independent, so they appear once two blocks are
        complete."""
        rdf, stats = self._checked_rdf_statistics(pairs, block_time)
        # The RDF frame closest to time
        frame = (len(rdf.times) - 1 if time is None
                 else int(np.argmin(np.abs(rdf.times - time))))
        fig = rdf_figure(rdf, stats, frame, rdf.times[frame], pairs)
        return self._save_figure(fig, save,
                                 f"RDF/RDF_t={rdf.times[frame]:.4f}.png")

    def plot_timings(self, save=False):
        """Where the computation time went (timings.png)."""
        fig = timings_figure(self.timings, self.wall_time,
                             self.config.nr_steps)
        return self._save_figure(fig, save, "timings.png", dpi=300,
                                 bbox_inches='tight')

    def animate_rdf(self, pair=None, max_frames=200,
                    block_time=RDF_BLOCK_TIME, save=False):
        """Makes and displays an animation of the RDF over time: of one pair
        (with its confidence interval, from blocks of block_time ps), or of
        the means of all pairs if pair is None. It shows at most max_frames
        of the RDF frames. save=True keeps it (with pngs of its first and last
        frames) in the run folder (RDF/), and a path keeps them in that
        folder instead."""
        from IPython.display import Image

        rdf, stats = self._checked_rdf_statistics(
            None if pair is None else [pair], block_time)

        def make(folder):
            path = save_rdf_gif(rdf, stats, self._rdf_frame_interval, folder,
                                pair, max_frames)
            # Image reads the file right away, so it may be deleted after
            return Image(filename=str(path))

        if save is False:
            # A notebook can only show a gif from a file: use a temporary one
            with tempfile.TemporaryDirectory() as folder:
                return make(Path(folder))
        return make(self.folder / "RDF" if save is True else Path(save))

    @property
    def _rdf_frame_interval(self):
        """Time between RDF frames (ps)."""
        return self.config.numerics.dt * self.config.recording.rdf_interval

    # ---- Saving and loading ----
    def save(self):
        """Writes the recorded data to results.npz (NumPy arrays, by name)
        and the settings to config.json, in the run folder. run() does this
        at the end; Results.load(folder) reads them back."""
        arrays = dict(energy_times=self.energy_times,
                      kinetic_energies=self.kinetic_energies,
                      wall_time=self.wall_time,
                      timing_labels=list(self.timings),
                      timing_seconds=list(self.timings.values()))
        for f in fields(Topology):
            arrays[f"topology/{f.name}"] = getattr(self.topology, f.name)
        if self.potential_energies is not None:
            for name in POTENTIAL_TERMS:
                arrays[f"potential_energies/{name}"] = (
                    self.potential_energies[name])
        if self.phase_space is not None:
            phase_space = self.phase_space
            arrays["phase_space/bond_types"] = np.array(
                phase_space.bond_types, dtype=str)
            arrays["phase_space/times"] = phase_space.times
            for k in range(len(phase_space.bond_types)):
                arrays[f"phase_space/lengths/{k}"] = phase_space.lengths[k]
                arrays[f"phase_space/length_rates/{k}"] = (
                    phase_space.length_rates[k])
        if self.rdf is not None:
            arrays["rdf/pair_labels"] = np.array(self.rdf.pair_labels,
                                                 dtype=str)
            arrays["rdf/times"] = self.rdf.times
            arrays["rdf/bin_edges"] = self.rdf.bin_edges
            arrays["rdf/histograms"] = self.rdf.histograms
            arrays["rdf/nr_pairs"] = self.rdf.nr_pairs
        np.savez_compressed(self.folder / "results.npz", **arrays)
        with open(self.folder / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, folder):
        """The Results of an earlier run, from its run folder (written by
        save()). Everything can be plotted again, as after the run."""
        folder = Path(folder)
        with open(folder / "config.json") as f:
            config_dict = json.load(f)
        try:
            config = Config.from_dict(config_dict)
        except TypeError as error:
            raise ValueError(f"{folder} was written by an older version of "
                             f"jostle ({error}); run it again.") from None
        with np.load(folder / "results.npz") as file:
            data = dict(file)
        topology = Topology(**{f.name: data[f"topology/{f.name}"]
                               for f in fields(Topology)})
        potential_energies = phase_space = rdf = None
        if "potential_energies/Bond" in data:
            potential_energies = {
                name: data[f"potential_energies/{name}"]
                for name in POTENTIAL_TERMS}
        if "phase_space/bond_types" in data:
            bond_types = data["phase_space/bond_types"]
            phase_space = PhaseSpace.from_arrays(
                bond_types, data["phase_space/times"],
                [data[f"phase_space/lengths/{k}"]
                 for k in range(len(bond_types))],
                [data[f"phase_space/length_rates/{k}"]
                 for k in range(len(bond_types))])
        if "rdf/pair_labels" in data:
            rdf = RDF.from_arrays(data["rdf/pair_labels"], data["rdf/times"],
                                  data["rdf/bin_edges"],
                                  data["rdf/histograms"],
                                  data["rdf/nr_pairs"])
        return cls(
            config=config, folder=folder, topology=topology,
            energy_times=data["energy_times"],
            potential_energies=potential_energies,
            kinetic_energies=data["kinetic_energies"],
            phase_space=phase_space, rdf=rdf,
            timings=dict(zip(data["timing_labels"].tolist(),
                             data["timing_seconds"].tolist(), strict=True)),
            wall_time=float(data["wall_time"]))

    # ---- VMD ----
    @property
    def vmd_script(self):
        """Path of the Tcl script that view_in_vmd() runs in VMD."""
        return self.folder / "view_in_vmd.tcl"

    def view_in_vmd(self, vmd_path=None, show_periodic_images=True,
                    show_box=True, show_image_outlines=True):
        """Opens the trajectory in VMD, in its own window (the notebook stays
        usable): licorice on white, fading with distance from the camera.
        show_periodic_images shows the 26 periodic images of the box,
        show_image_outlines their dashed grey outlines, and show_box the red
        outline of the box itself. VMD is vmd_path, else found by
        find_vmd(). The script it runs is kept as `vmd_script`, so the run
        can be opened again later with `vmd -e view_in_vmd.tcl`."""
        from . import vmd

        executable = vmd.find_vmd(vmd_path)
        if executable is None:
            raise FileNotFoundError(
                f"VMD not found at {str(vmd_path)!r}." if vmd_path is not None
                else f"VMD not found. Install it from "
                     f"https://www.ks.uiuc.edu/Research/vmd/, or give its "
                     f"executable with view_in_vmd(vmd_path=...) or the "
                     f"{vmd.VMD_ENV_VAR} environment variable.")
        if not self.trajectory.exists():
            raise FileNotFoundError(f"{self.trajectory} does not exist.")
        vmd.write_vmd_script(
            self.vmd_script, self.trajectory,
            10 * self.config.system.box_length, self.topology.bonds,
            show_periodic_images, show_box, show_image_outlines)
        vmd.launch_vmd(executable, self.vmd_script)
        print(f"Opened {self.trajectory.name} in VMD ({executable})")

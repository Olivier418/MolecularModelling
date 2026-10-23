"""End-to-end checks of system construction and short simulation runs."""
from dataclasses import fields

import numpy as np
import pytest
from matplotlib.figure import Figure

from helpers import forcefield, make_topology, water_ethanol_pair
from jostle.config import Config
from jostle.constants import BUILTIN_MOLECULE_DIR
from jostle.geometry import pair_vectors
from jostle.minimise import minimise_energy, thermal_force
from jostle.observables import RDF, PhaseSpace, accumulate_pair_histogram
from jostle.results import Results
from jostle.simulation import run
from jostle.system import Topology, build_initial_box, load_molecule


def test_build_initial_box_counts():
    t, x = build_initial_box(2.0, {"Water": 1, "Ethanol": 1, "Oxygen": 0},
                             BUILTIN_MOLECULE_DIR, np.random.default_rng(0))
    n = t.nr_atoms
    assert x.shape == (n, 3) and len(t.masses) == n == len(t.molecule_names)
    assert t.bonds.max() < n and (t.bonds[:, 0] < t.bonds[:, 1]).all()
    # Bonds, angles and dihedrals never cross molecules
    for group in (t.bonds, t.angles, t.dihedrals):
        assert (t.molecule_index[group] == t.molecule_index[group[:, :1]]).all()
    n_water = np.sum((t.molecule_names == "Water") & (t.elements == "O"))
    n_ethanol = np.sum((t.molecule_names == "Ethanol") & (t.elements == "O"))
    assert n_water == n_ethanol > 0
    assert "Oxygen" not in t.molecule_names


@pytest.mark.parametrize("weights", [{"Water": 1, "Ethanol": 1},
                                     {"Water": 1, "Hydrogen": 1},
                                     {"Oxygen": 1}])
def test_minimisation_brings_forces_down_to_thermal_size(weights):
    L, T = 2.0, 298.15
    t, x = build_initial_box(L, weights, BUILTIN_MOLECULE_DIR,
                             np.random.default_rng(2))
    ff = forcefield(t, L, 2.5)
    x_min, _ = minimise_energy(x, ff, T)
    energies = [sum(ff.compute_forces(y, True).potential_energies.values())
                for y in (x, x_min)]
    forces = ff.compute_forces(x_min).forces
    assert energies[1] < energies[0]
    assert np.linalg.norm(forces, axis=1).max() <= thermal_force(t, T)


def _config(tmp_path, **kwargs):
    settings = dict(box_length=1.5, lj_cutoff=2.0, dt=0.001,
                    end_time=0.02, energy_interval=5, trajectory_interval=10,
                    rdf_interval=4, seed=0, output_dir=tmp_path,
                    record_potential_energies=False,
                    phase_space_bonds=[], rdf_pairs=[],
                    rdf_bin_width=0.05)
    settings.update(kwargs)
    return Config(**settings).validate()


ALL_MEASUREMENTS = dict(
    record_potential_energies=True,
    phase_space_bonds=['O-H (Water)', 'C-O (Ethanol)'],
    rdf_pairs=['O~O (Water~Water)', 'O~H (Ethanol~Water)'])


def _xyz_times(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return [float(lines[k].split("t=")[1]) for k in range(1, len(lines), n + 2)]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """A short run that records everything."""
    return run(_config(tmp_path_factory.mktemp("full"), **ALL_MEASUREMENTS))


@pytest.fixture(scope="module")
def bare_run(tmp_path_factory):
    """The same run, recording only what is always recorded."""
    return run(_config(tmp_path_factory.mktemp("bare")))


def test_each_measurement_has_its_own_frames(full_run):
    # 20 steps of 1 fs: every 10 steps (trajectory), 5 (energies), 4 (RDF)
    # and every step (phase space, whole run), plus the initial frame
    np.testing.assert_allclose(_xyz_times(full_run.trajectory),
                               [0, 0.01, 0.02])
    np.testing.assert_allclose(full_run.energy_times,
                               [0, 0.005, 0.01, 0.015, 0.02])
    assert len(full_run.temperature) == 5
    assert len(full_run.potential_energies["LJ"]) == 5
    np.testing.assert_allclose(full_run.rdf.times, 0.004 * np.arange(6))
    assert full_run.rdf.histograms.shape[0] == 6
    np.testing.assert_allclose(full_run.phase_space.times,
                               0.001 * np.arange(21))
    assert all(len(d) == 21 for d in full_run.phase_space.lengths)


def test_trajectory_is_reproducible_and_independent_of_recording(full_run,
                                                                 bare_run):
    assert full_run.trajectory.read_text() == bare_run.trajectory.read_text()


def test_figures_are_made_on_request(full_run):
    results, folder = full_run, full_run.folder
    for fig in (results.plot_energy(), results.plot_temperature(),
                results.plot_timings(), results.plot_rdf(time=0.01),
                results.plot_phase_space("O-H (Water)")):
        assert isinstance(fig, Figure)
    assert results.animate_rdf(max_frames=3).data[:3] == b"GIF"
    # Nothing is saved unless asked for
    assert not list(folder.rglob("*.png")) and not list(folder.rglob("*.gif"))


def test_results_load_gives_the_same_data(full_run):
    results = full_run
    loaded = Results.load(results.folder)
    assert loaded.config == results.config
    for name in ("energy_times", "kinetic_energies", "temperature"):
        np.testing.assert_array_equal(getattr(loaded, name),
                                      getattr(results, name))
    for name, values in results.potential_energies.items():
        np.testing.assert_array_equal(loaded.potential_energies[name], values)
    np.testing.assert_array_equal(loaded.lj_correction, results.lj_correction)
    np.testing.assert_array_equal(loaded.rdf.g_of_r(), results.rdf.g_of_r())
    np.testing.assert_array_equal(loaded.rdf.nr_pairs, results.rdf.nr_pairs)
    assert loaded.rdf.pair_labels == results.rdf.pair_labels
    assert loaded.phase_space.bond_types == results.phase_space.bond_types
    for name in ("lengths", "length_rates"):
        for a, b in zip(getattr(loaded.phase_space, name),
                        getattr(results.phase_space, name), strict=True):
            np.testing.assert_array_equal(a, b)
    for f in fields(Topology):
        np.testing.assert_array_equal(getattr(loaded.topology, f.name),
                                      getattr(results.topology, f.name))
    assert loaded.timings == results.timings
    assert loaded.wall_time == results.wall_time
    np.testing.assert_array_equal(loaded.rdf.times, results.rdf.times)
    np.testing.assert_array_equal(loaded.phase_space.times,
                                  results.phase_space.times)
    assert isinstance(loaded.plot_rdf(), Figure)


# ------------------------------- Observables ---------------------------------
def test_RDF_counts_every_intermolecular_pair_once():
    x_e, ethanol = load_molecule("Ethanol", BUILTIN_MOLECULE_DIR)
    t = Topology.concatenate([ethanol, ethanol])
    x = np.vstack([x_e, x_e + 0.5])
    rdf = RDF(t, ["H~H (Ethanol~Ethanol)"], 4.0, 1.0, 0.01, np.zeros(1),
              np.random.default_rng(0))
    rdf.record(0, x)
    # 6 H atoms per ethanol: 6 x 6 pairs between the molecules, all within
    # half the box; the 2 x 15 pairs within a molecule are left out
    assert rdf.nr_pairs[0] == 36
    assert rdf.histograms[0, 0].sum() == 36


def test_RDF_is_one_for_an_ideal_gas():
    # Single-atom molecules at uniformly random positions: no structure
    n, box_length, nr_saves = 200, 2.0, 200
    t = make_topology(n, molecule_index=np.arange(n))
    rng = np.random.default_rng(0)
    rdf = RDF(t, ["H~H (Test~Test)"], box_length, 1.0, 0.1,
              0.1 * np.arange(nr_saves), rng)
    assert rdf.nr_pairs[0] == n * (n - 1) // 2
    for k in range(nr_saves):
        rdf.record(k, rng.uniform(0, box_length, (n, 3)))
    np.testing.assert_allclose(rdf.g_of_r().mean(axis=0), 1, rtol=0.1)


def test_phase_space_velocity_is_rate_of_change_of_bond_length():
    x, t = water_ethanol_pair()
    v = np.random.default_rng(0).normal(0, 1, x.shape)
    ff = forcefield(t)
    ps = PhaseSpace(t, ["O-H (Water)", "C-H (Ethanol)"], np.zeros(1))
    ps.record(0, ff.compute_forces(x), v)

    h = 1e-7
    later = PhaseSpace(t, ["O-H (Water)", "C-H (Ethanol)"], np.zeros(1))
    later.record(0, ff.compute_forces(x + h * v), v)
    for k in range(2):
        finite_difference = (later.lengths[k][0] - ps.lengths[k][0]) / h
        np.testing.assert_allclose(ps.length_rates[k][0], finite_difference,
                                   rtol=1e-5)


def test_accumulate_pair_histogram_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 3., (300, 3))
    i, j = np.triu_indices(300, k=1)
    edges = np.linspace(0, 1.5, 151)
    counts = np.zeros(150)
    accumulate_pair_histogram(x, i, j, 3., edges, counts)
    _, D = pair_vectors(x, i, j, 3.)
    np.testing.assert_array_equal(counts, np.histogram(D, edges)[0])

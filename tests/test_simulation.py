"""End-to-end checks of system construction and short simulation runs."""
import numpy as np

from mdsim.config import Config
from mdsim.constants import MOLECULE_DIR
from mdsim.observables import RDF, PhaseSpace
from mdsim.simulation import run
from mdsim.system import build_system

from helpers import forcefield, water_ethanol_pair


def test_build_system_counts():
    t, x = build_system(2.0, 0.1, {"Water": 1, "Ethanol": 1, "Oxygen": 0},
                        MOLECULE_DIR, np.random.default_rng(0))
    n = t.nr_atoms
    assert x.shape == (n, 3) and len(t.masses) == n == len(t.molec_names)
    assert t.bonds.max() < n and (t.bonds[:, 0] < t.bonds[:, 1]).all()
    # Bonds, angles and dihedrals never cross molecules
    for group in (t.bonds, t.angles, t.dihedrals):
        assert (t.molec_index[group] == t.molec_index[group[:, :1]]).all()
    n_water = np.sum((t.molec_names == "Water") & (t.atoms == "O"))
    n_ethanol = np.sum((t.molec_names == "Ethanol") & (t.atoms == "O"))
    assert n_water == n_ethanol > 0
    assert "Oxygen" not in t.molec_names


def test_build_system_preserves_molecule_shapes():
    t, x = build_system(2.0, 0.2, {"Ethanol": 1}, MOLECULE_DIR,
                        np.random.default_rng(1))
    # Rigid placement keeps every bond at its length in the template
    lengths = np.linalg.norm(x[t.bonds[:, 0]] - x[t.bonds[:, 1]], axis=1)
    per_molecule = lengths.reshape(-1, 8)
    np.testing.assert_allclose(
        per_molecule, np.broadcast_to(per_molecule[0], per_molecule.shape),
        atol=1e-6)


def _config(tmp_path, **kwargs):
    settings = dict(boxsize=1.5, end_time=0.02, save_steps=5, seed=0,
                    output_dir=tmp_path, show_plots=False, energy_graph=False,
                    temp_graph=False, phase_space_diagram=False,
                    comptime_vis=False, RDF_vis=False, RDF_dr=0.05,
                    pairs_of_interest=['O~O (Water~Water)',
                                       'O~H (Ethanol~Water)'])
    settings.update(kwargs)
    return Config(**settings).validate()


def _xyz_times(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return [float(lines[k].split("t=")[1]) for k in range(1, len(lines), n + 2)]


def test_short_run_writes_trajectory(tmp_path):
    folder = run(_config(tmp_path))
    assert (folder / "INITIAL.itp").exists()
    # 20 steps saved every 5 steps, plus the initial frame
    times = _xyz_times(folder / "INTEGRATED.xyz")
    np.testing.assert_allclose(times, [0, 0.005, 0.01, 0.015, 0.02])


def test_short_run_is_reproducible_with_seed(tmp_path):
    a = run(_config(tmp_path / "a"))
    b = run(_config(tmp_path / "b"))
    assert ((a / "INTEGRATED.xyz").read_text()
            == (b / "INTEGRATED.xyz").read_text())


def test_short_run_with_all_plots(tmp_path):
    folder = run(_config(tmp_path, energy_graph=True, temp_graph=True,
                         phase_space_diagram=True, comptime_vis=True,
                         RDF_vis=True))
    for name in ["energy.png", "temperature.png", "comptimes_piechart.png",
                 "RDF/RDF.gif", "ps_diagram/ps_diag_H~O (Water).png"]:
        assert (folder / name).exists(), name
    np.testing.assert_allclose(_xyz_times(folder / "INTEGRATED.xyz"),
                               [0, 0.005, 0.01, 0.015, 0.02])


def test_soft_start_terminates_for_oxygen(tmp_path):
    folder = run(_config(tmp_path, weights={"Oxygen": 1}, boxsize=2.0,
                         end_time=0.002, save_steps=1))
    assert (folder / "INTEGRATED.xyz").exists()


# ------------------------------- Observables ---------------------------------
def test_RDF_pair_keeps_atoms_with_their_molecules():
    x, t = water_ethanol_pair()
    rdf = RDF(t, ["O~H (Ethanol~Water)"], 2.0, 1.0, 0.1, 1,
              np.random.default_rng(0))
    assert rdf.names == ["O~H (Ethanol~Water)"]
    i, j = rdf.pairs[0]
    species = {frozenset({(t.atoms[a], t.molec_names[a]),
                          (t.atoms[b], t.molec_names[b])})
               for a, b in zip(i, j)}
    # The ethanol O with each of the two water H atoms
    assert species == {frozenset({("O", "Ethanol"), ("H", "Water")})}
    assert len(i) == 2


def test_RDF_counts_every_pair_once():
    x, t = water_ethanol_pair()
    rdf = RDF(t, ["H~H (Ethanol~Ethanol)"], 4.0, 1.0, 0.01, 1,
              np.random.default_rng(0))
    rdf.record(0, x)
    # 6 ethanol H atoms make 15 pairs, all within half the box
    assert rdf.histograms[0, 0].sum() == 15


def test_phase_space_tracks_requested_bonds():
    x, t = water_ethanol_pair()
    ps = PhaseSpace(t, ["O-H (Water)", "C-C (Ethanol)", "O-O (Oxygen)"], 1)
    assert ps.names == ["H~O (Water)", "C~C (Ethanol)"]
    ps.record(0, forcefield(t).compute(x))
    expected = np.linalg.norm(x[[1, 2]] - x[0], axis=1)  # water O-H bonds
    np.testing.assert_allclose(ps.distances[0][0], expected, atol=1e-12)

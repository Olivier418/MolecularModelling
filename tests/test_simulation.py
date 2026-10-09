"""End-to-end checks of system construction and short simulation runs."""
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from auxiliary import compute_LJ_indices  # noqa: E402
from initial_config_generation import random_config  # noqa: E402
from integrators import integrate, velo_verlet  # noqa: E402

from helpers import load_molecule  # noqa: E402


def test_random_config_counts():
    np.random.seed(0)
    (atoms, x, bonds, *_, masses, molec_index, molec_names) = random_config(
        2.0, 0.1, ["Water", "Ethanol"], [1, 1])
    n = len(atoms)
    assert x.shape == (n, 3) and len(masses) == n == len(molec_names)
    assert bonds.max() < n and (bonds[:, 0] < bonds[:, 1]).all()
    # Every molecule is internally consistent: bonds never cross molecules
    assert (molec_index[bonds[:, 0]] == molec_index[bonds[:, 1]]).all()
    n_water = np.sum((molec_names == "Water") & (atoms == "O"))
    n_ethanol = np.sum((molec_names == "Ethanol") & (atoms == "O"))
    assert n_water == n_ethanol > 0


@pytest.mark.xfail(strict=True, reason="diatomics have no angles, so the "
                                       "bonded pair is not excluded")
def test_bonded_pairs_excluded_from_LJ():
    o2 = load_molecule("Oxygen")
    i_LJ, j_LJ = compute_LJ_indices(2, o2["dihedrals"], o2["angles"],
                                    o2["epsilon"], o2["sigma"], np.zeros(2))
    assert len(i_LJ) == 0


def _short_run(tmp_path, **kwargs):
    np.random.seed(0)
    boxsize = 1.5
    (atoms, x, bonds, kb, r0, angles, ktheta, theta0, epsilon, sigma,
     dihedrals, C1, C2, C3, C4, masses, molec_index,
     molec_names) = random_config(boxsize, 0, ["Water", "Ethanol"], [1, 1])
    options = dict(VMD_vis=False, energy_graph=False, temp_graph=False,
                   phase_space_diagram=False, RDF_vis=False,
                   comptime_vis=False, r_threshold_scalar=0.65,
                   bonds_of_interest=['O-H (Water)'],
                   pairs_of_interest=['O~O (Water~Water)'],
                   RDF_dr=0.05)
    options.update(kwargs)
    integrate(x, bonds, atoms, kb, r0, angles, ktheta, theta0, epsilon, sigma,
              dihedrals, C1, C2, C3, C4, masses, velo_verlet, 0.001, 0.02, 5,
              boxsize, str(tmp_path), molec_index, molec_names, 2, None,
              298.15, True, **options)
    (run_dir,) = [os.path.join(tmp_path, d) for d in os.listdir(tmp_path)]
    return run_dir, len(atoms)


def _xyz_times(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    return [float(lines[k].split("t=")[1]) for k in range(1, len(lines), n + 2)]


def test_short_run_writes_trajectory(tmp_path):
    run_dir, n = _short_run(tmp_path)
    assert os.path.exists(os.path.join(run_dir, "INITIAL.itp"))
    # 20 steps saved every 5 steps, plus the initial frame
    times = _xyz_times(os.path.join(run_dir, "INTEGRATED.xyz"))
    np.testing.assert_allclose(times, [0, 0.005, 0.01, 0.015, 0.02])


def test_short_run_with_all_plots(tmp_path):
    run_dir, _ = _short_run(tmp_path, energy_graph=True, temp_graph=True,
                            phase_space_diagram=True, comptime_vis=True)
    for name in ["energy.png", "temperature.png", "comptimes_piechart.png"]:
        assert os.path.exists(os.path.join(run_dir, name))


@pytest.mark.xfail(strict=True, reason="np.in1d was removed in NumPy 2")
def test_short_run_with_RDF(tmp_path):
    run_dir, _ = _short_run(tmp_path, RDF_vis=True)
    assert os.path.exists(os.path.join(run_dir, "RDF", "RDF.gif"))
    # The RDF loop must not clobber the time written to the trajectory
    times = _xyz_times(os.path.join(run_dir, "INTEGRATED.xyz"))
    np.testing.assert_allclose(times, [0, 0.005, 0.01, 0.015, 0.02])

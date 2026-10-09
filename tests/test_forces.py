import numpy as np
import pytest

from auxiliary import diff_and_dist_arr, find_bond_indices
from energy import angular_energy, bond_energy, dihedral_energy, LJ_energy
from forces import (ang_force, bond_force, dihedral_force, LJ_force,
                    net_force)
from parse_functions import itp_to_data

from helpers import water_ethanol_pair

eps = 1e-12


# ------------------------------- Single terms --------------------------------
def test_bond():
    """Bond forces for stretched, compressed and equilibrium bonds."""
    X = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    bonds = np.array([[0, 1]])
    diff, D = diff_and_dist_arr(X, bonds[:, 0], bonds[:, 1], 0.)

    # H-H bond separated by more than equilibrium distance
    F1 = bond_force(diff, D, bonds, np.array([24531]), np.array([0.074]), 2)
    assert F1[0, 0] > 0 and F1[1, 0] < 0 and F1[0, 0] == -F1[1, 0]

    # O-O bond separated by less than equilibrium distance
    F2 = bond_force(diff, D, bonds, np.array([49836]), np.array([0.12075]), 2)
    assert F2[0, 0] < 0 and F2[1, 0] > 0 and F2[0, 0] == -F2[1, 0]

    # C-O bond at equilibrium distance
    X = np.array([[0.11282, 0.0, 0.0], [0.0, 0.0, 0.0]])
    diff, D = diff_and_dist_arr(X, bonds[:, 0], bonds[:, 1], 0.)
    F3 = bond_force(diff, D, bonds, np.array([107650]), np.array([0.11282]), 2)
    assert (np.abs(F3) < eps).all()


@pytest.mark.parametrize("delta, F_target", [
    (0.0, np.zeros((3, 3))),
    (0.1, np.array([[-1, -1, 0], [0, 1, 0], [1, 1, 0]])),
    (-0.1, np.array([[1, 1, 0], [0, -1, 0], [-1, -1, 0]])),
])
def test_ang(delta, F_target):
    """Angular force directions at, above and below the equilibrium angle."""
    ktheta = np.array([628.02])
    theta0 = np.array([np.deg2rad(104.52)])
    theta = theta0[0] + delta
    X = np.array([[0., 0., 0.], [1., 0., 0.],
                  [np.cos(theta), np.sin(theta), 0.]])
    bonds = np.array([[0, 1], [0, 2]])
    angles = np.array([[1, 0, 2]])

    diff_bonds, D_bonds = diff_and_dist_arr(X, bonds[:, 0], bonds[:, 1], 0.)
    F = ang_force(diff_bonds, D_bonds, find_bond_indices(angles, bonds),
                  angles, ktheta, theta0, 3)

    if delta == 0.0:
        assert (np.abs(F) < eps).all()
    else:
        assert (np.abs(np.sign(F) - F_target) < eps).all()


@pytest.mark.parametrize("separation, sign", [(0.1, 1), (100., -1)])
def test_LJ(separation, sign):
    """LJ is repulsive at short range and attractive at long range."""
    X = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    i_LJ, j_LJ = np.array([0]), np.array([1])
    diff_LJ, D_LJ = diff_and_dist_arr(X, i_LJ, j_LJ, 0.)

    F = LJ_force(diff_LJ, D_LJ, np.array([0.66386]), np.array([0.315061]),
                 i_LJ, j_LJ, 2, np.array([np.inf]))
    assert np.array_equal(np.sign(F), sign * np.array([[-1, 0, 0],
                                                       [1, 0, 0]]))


def test_dihedral_planar():
    """A planar trans dihedral experiences no out-of-plane force."""
    dihedrals = np.array([[0, 1, 2, 3]])
    X = np.array([[1.2699, 0, 0.1772], [-0.2033, 0, 0.5345],
                  [-0.9394, 0, -0.6674], [-1.8540, 0, -0.4252]])
    bonds = np.array([[0, 1], [1, 2], [2, 3]])
    diff_bonds, D_bonds = diff_and_dist_arr(X, bonds[:, 0], bonds[:, 1], 0.)

    F = dihedral_force(diff_bonds, D_bonds, find_bond_indices(dihedrals, bonds),
                       dihedrals, np.array([-0.44310]), np.array([3.83255]),
                       np.array([0.72801]), np.array([-4.11705]), 4)
    assert np.array_equal(np.sign(F), np.zeros((4, 3)))


def test_net_water():
    """Net force on the reference water geometry matches known values."""
    X = 0.1 * np.array([[1.93617934, 2.31884508, 1.72261570],
                        [1.78931374, 3.24075634, 1.51114298],
                        [2.30448689, 1.98045541, 0.90160232]])
    F_example = np.array([[16.0, 13.66, -55.08],
                          [-10.08, 4.37, 25.81],
                          [-5.92, -18.04, 29.28]])

    (bonds, kb, r0, angles, ktheta, theta0, _, _, dihedrals,
     C1, C2, C3, C4, _) = itp_to_data("molecule_data/Water.itp")
    diff_bonds, D_bonds = diff_and_dist_arr(X, bonds[:, 0], bonds[:, 1], 0.)
    empty = np.array([])
    F = net_force(diff_bonds, D_bonds, empty.reshape(0, 3), empty, bonds, 3,
                  find_bond_indices(angles, bonds),
                  find_bond_indices(dihedrals, bonds), kb, r0, angles, ktheta,
                  theta0, empty, empty, dihedrals, C1, C2, C3, C4,
                  empty.astype(int), empty.astype(int))

    assert (np.abs(F_example - np.round(0.1 * F, 2)) < 1e-9).all()


# ------------------------ Forces are -grad(energy) ---------------------------
def _terms(s):
    """(energy(x), force(x)) callables for each force field term."""
    bonds, n = s["bonds"], len(s["masses"])

    def bonded(x):
        return diff_and_dist_arr(x, bonds[:, 0], bonds[:, 1], 0.)

    def lj(x):
        return diff_and_dist_arr(x, s["i_LJ"], s["j_LJ"], 0.)

    C = (s["C1"], s["C2"], s["C3"], s["C4"])
    rc = np.full(len(s["i_LJ"]), np.inf)
    return {
        "bond": (lambda x: bond_energy(bonded(x)[1], s["kb"], s["r0"]),
                 lambda x: bond_force(*bonded(x), bonds, s["kb"], s["r0"], n)),
        "angle": (lambda x: angular_energy(*bonded(x), s["angle_mapping"],
                                           s["ktheta"], s["theta0"]),
                  lambda x: ang_force(*bonded(x), s["angle_mapping"],
                                      s["angles"], s["ktheta"], s["theta0"],
                                      n)),
        "dihedral": (lambda x: dihedral_energy(*bonded(x),
                                               s["dihedral_mapping"], *C),
                     lambda x: dihedral_force(*bonded(x),
                                              s["dihedral_mapping"],
                                              s["dihedrals"], *C, n)),
        "LJ": (lambda x: LJ_energy(lj(x)[1], s["epsilon_LJ"], s["sigma_LJ"],
                                   rc),
               lambda x: LJ_force(*lj(x), s["epsilon_LJ"], s["sigma_LJ"],
                                  s["i_LJ"], s["j_LJ"], n, rc)),
    }


@pytest.mark.parametrize("term", ["bond", "angle", "dihedral", "LJ"])
def test_force_is_negative_energy_gradient(term):
    """Each force term equals minus the numerical gradient of its energy."""
    rng = np.random.default_rng(0)
    s = water_ethanol_pair()
    assert len(s["i_LJ"]) > 0 and len(s["dihedrals"]) > 0
    x = s["x"] + rng.normal(0, 0.005, s["x"].shape)

    energy, force = _terms(s)[term]
    h = 1e-6
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        grad[idx] = (energy(xp) - energy(xm)) / (2 * h)

    F = force(x)
    np.testing.assert_allclose(F, -grad, rtol=1e-5,
                               atol=1e-5 * np.abs(grad).max())

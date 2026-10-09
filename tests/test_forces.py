from dataclasses import replace

import numpy as np
import pytest

from mdsim.constants import MOLECULE_DIR
from mdsim.forcefield import lj_cutoffs, lj_pairs
from mdsim.geometry import map_to_bonds, pair_vectors
from mdsim.system import load_molecule

from helpers import (forcefield, forces_and_energy, make_topology,
                     water_ethanol_pair)

eps = 1e-12


# ------------------------------- Geometry ------------------------------------
def test_pair_vectors():
    X = np.array([[1., 0., 0.], [0., 1., 0.]])
    diff, dist = pair_vectors(X, np.array([0]), np.array([1]), 0.)
    np.testing.assert_allclose(diff, [[1, -1, 0]], atol=eps)
    np.testing.assert_allclose(dist, [2 ** 0.5], atol=eps)


def test_pair_vectors_minimum_image():
    X = np.array([[0.1, 0., 0.], [3.9, 0., 0.]])
    diff, dist = pair_vectors(X, np.array([0]), np.array([1]), 4.)
    np.testing.assert_allclose(diff, [[0.2, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(dist, [0.2], atol=1e-12)


def test_map_to_bonds():
    bonds = np.array([[0, 1], [1, 2], [2, 3]])
    indices, signs = map_to_bonds(np.array([[3, 2, 1, 0]]), bonds)
    np.testing.assert_array_equal(indices, [[2, 1, 0]])
    np.testing.assert_array_equal(signs, [[1, 1, 1]])


# ------------------------------- Single terms --------------------------------
@pytest.mark.parametrize("kb, r0, x1, sign", [
    (24531, 0.074, 0.1, 1),       # H-H stretched: atoms pulled together
    (49836, 0.12075, 0.1, -1),    # O-O compressed: atoms pushed apart
    (107650, 0.11282, 0.11282, 0),  # C-O at equilibrium: no force
])
def test_bond(kb, r0, x1, sign):
    X = np.array([[0.0, 0.0, 0.0], [x1, 0.0, 0.0]])
    t = make_topology(2, bonds=np.array([[0, 1]]), kb=np.array([kb]),
                      r0=np.array([r0]))
    F = forcefield(t).compute(X).forces
    assert F[0, 0] == -F[1, 0]
    if sign == 0:
        assert (np.abs(F) < eps).all()
    else:
        assert np.sign(F[0, 0]) == sign


@pytest.mark.parametrize("delta, F_target", [
    (0.0, np.zeros((3, 3))),
    (0.1, np.array([[-1, -1, 0], [0, 1, 0], [1, 1, 0]])),
    (-0.1, np.array([[1, 1, 0], [0, -1, 0], [-1, -1, 0]])),
])
def test_ang(delta, F_target):
    """Angular force directions at, above and below the equilibrium angle."""
    theta0 = np.deg2rad(104.52)
    theta = theta0 + delta
    X = np.array([[0., 0., 0.], [1., 0., 0.],
                  [np.cos(theta), np.sin(theta), 0.]])
    # Bonds with kb=0 are only there to define the angle
    t = make_topology(3, bonds=np.array([[0, 1], [0, 2]]), kb=np.zeros(2),
                      r0=np.ones(2), angles=np.array([[1, 0, 2]]),
                      ktheta=np.array([628.02]), theta0=np.array([theta0]))
    F = forcefield(t).compute(X).forces

    if delta == 0.0:
        assert (np.abs(F) < eps).all()
    else:
        assert (np.abs(np.sign(F) - F_target) < eps).all()


@pytest.mark.parametrize("separation, sign", [(0.1, 1), (100., -1)])
def test_LJ(separation, sign):
    """LJ is repulsive at short range and attractive at long range."""
    X = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    t = make_topology(2, epsilon=np.full(2, 0.66386),
                      sigma=np.full(2, 0.315061))
    F = forcefield(t).compute(X).forces
    assert np.array_equal(np.sign(F), sign * np.array([[-1, 0, 0],
                                                       [1, 0, 0]]))


def test_dihedral_planar():
    """A planar trans dihedral at this geometry experiences no force."""
    X = np.array([[1.2699, 0, 0.1772], [-0.2033, 0, 0.5345],
                  [-0.9394, 0, -0.6674], [-1.8540, 0, -0.4252]])
    t = make_topology(4, bonds=np.array([[0, 1], [1, 2], [2, 3]]),
                      kb=np.zeros(3), r0=np.ones(3),
                      dihedrals=np.array([[0, 1, 2, 3]]),
                      C=np.array([[-0.44310, 3.83255, 0.72801, -4.11705]]))
    F = forcefield(t).compute(X).forces
    assert np.array_equal(np.sign(F), np.zeros((4, 3)))


def test_net_water():
    """Net force on the reference water geometry matches known values."""
    X = 0.1 * np.array([[1.93617934, 2.31884508, 1.72261570],
                        [1.78931374, 3.24075634, 1.51114298],
                        [2.30448689, 1.98045541, 0.90160232]])
    F_example = np.array([[16.0, 13.66, -55.08],
                          [-10.08, 4.37, 25.81],
                          [-5.92, -18.04, 29.28]])
    _, water = load_molecule("Water", MOLECULE_DIR)
    F = forcefield(water).compute(X).forces
    assert (np.abs(F_example - np.round(0.1 * F, 2)) < 1e-9).all()


# ------------------------------ LJ bookkeeping -------------------------------
def test_bonded_pairs_excluded_from_LJ():
    _, o2 = load_molecule("Oxygen", MOLECULE_DIR)
    i_LJ, _ = lj_pairs(o2)
    assert len(i_LJ) == 0


def test_LJ_pairs_exclude_up_to_1_4_and_zero_epsilon():
    x, t = water_ethanol_pair()
    i, j = lj_pairs(t)
    assert len(i) > 0
    assert (t.epsilon[i] > 0).all() and (t.epsilon[j] > 0).all()
    # Within a molecule, every LJ pair is at least 5 atoms apart
    excluded = {tuple(sorted(p)) for group in (t.bonds, t.angles, t.dihedrals)
                for row in group for p in zip(row, row[1:])}
    excluded |= {(min(r[a], r[b]), max(r[a], r[b]))
                 for group in (t.angles, t.dihedrals) for r in group
                 for a in range(len(r)) for b in range(a + 1, len(r))}
    assert not excluded & set(zip(i.tolist(), j.tolist()))


def test_f_threshold_cutoff_has_requested_force():
    eps_LJ, sig_LJ = np.array([0.66386]), np.array([0.315061])
    rc = lj_cutoffs(None, 0.05, eps_LJ, sig_LJ, 4.0)
    r = rc[0]
    F = 24 * eps_LJ[0] * (2 * (sig_LJ[0] / r)**12 - (sig_LJ[0] / r)**6) / r
    assert abs(abs(F) - 0.05) < 1e-9


# ------------------------ Forces are -grad(energy) ---------------------------
def _only(topology, term):
    """The topology with every force field term except `term` switched off."""
    zero = {"bond": dict(kb=np.zeros_like(topology.kb)),
            "angle": dict(ktheta=np.zeros_like(topology.ktheta)),
            "dihedral": dict(C=np.zeros_like(topology.C)),
            "LJ": dict(epsilon=np.zeros_like(topology.epsilon))}
    changes = {}
    for other, values in zero.items():
        if other != term:
            changes.update(values)
    return replace(topology, **changes)


@pytest.mark.parametrize("term", ["bond", "angle", "dihedral", "LJ"])
def test_force_is_negative_energy_gradient(term):
    """Each force term equals minus the numerical gradient of its energy."""
    rng = np.random.default_rng(0)
    x, t = water_ethanol_pair()
    ff = forcefield(_only(t, term))
    if term == "LJ":
        assert len(ff.i_LJ) > 0
    x = x + rng.normal(0, 0.005, x.shape)

    h = 1e-6
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        grad[idx] = (forces_and_energy(ff, xp)[1]
                     - forces_and_energy(ff, xm)[1]) / (2 * h)

    F = forces_and_energy(ff, x)[0]
    assert np.abs(grad).max() > 1
    np.testing.assert_allclose(F, -grad, rtol=1e-5,
                               atol=1e-5 * np.abs(grad).max())

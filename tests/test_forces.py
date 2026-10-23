from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from helpers import (
    forcefield,
    forces_and_energy,
    make_topology,
    water_ethanol_pair,
)
from jostle.constants import BUILTIN_MOLECULE_DIR
from jostle.forcefield import ForceField
from jostle.geometry import pair_vectors
from jostle.integrators import (
    initial_state,
    integration_step,
    velocity_verlet,
)
from jostle.minimise import minimise_energy
from jostle.neighbours import lj_pairs
from jostle.system import build_initial_box, initial_velocities, load_molecule

eps = 1e-12
# LJ parameters of oxygen
EPSILON_O, SIGMA_O = 0.66386, 0.315061


# ------------------------------- Geometry ------------------------------------
@pytest.mark.parametrize("X, box_length, diff, dist", [
    ([[1., 0., 0.], [0., 1., 0.]], 0., [[1, -1, 0]], [2 ** 0.5]),
    # Minimum image
    ([[0.1, 0., 0.], [3.9, 0., 0.]], 4., [[0.2, 0, 0]], [0.2]),
])
def test_pair_vectors(X, box_length, diff, dist):
    d, D = pair_vectors(np.array(X), np.array([0]), np.array([1]), box_length)
    np.testing.assert_allclose(d, diff, atol=eps)
    np.testing.assert_allclose(D, dist, atol=eps)


# ------------------------------- Single terms --------------------------------
def test_net_water():
    """Net force on the reference water geometry matches known values."""
    X = 0.1 * np.array([[1.93617934, 2.31884508, 1.72261570],
                        [1.78931374, 3.24075634, 1.51114298],
                        [2.30448689, 1.98045541, 0.90160232]])
    F_example = np.array([[16.0, 13.66, -55.08],
                          [-10.08, 4.37, 25.81],
                          [-5.92, -18.04, 29.28]])
    _, water = load_molecule("Water", BUILTIN_MOLECULE_DIR)
    F = forcefield(water).compute_forces(X).forces
    assert (np.abs(F_example - np.round(0.1 * F, 2)) < 1e-9).all()


# ------------------------------ LJ bookkeeping -------------------------------
def test_LJ_pairs_exclude_up_to_1_4_and_zero_epsilon():
    x, t = water_ethanol_pair()
    i, j = lj_pairs(t)
    assert len(i) > 0
    assert (t.lj_epsilon[i] > 0).all() and (t.lj_epsilon[j] > 0).all()
    # Within a molecule, every LJ pair is at least 5 atoms apart
    excluded = {tuple(sorted(p)) for group in (t.bonds, t.angles, t.dihedrals)
                for row in group for p in zip(row, row[1:], strict=False)}
    excluded |= {(min(r[a], r[b]), max(r[a], r[b]))
                 for group in (t.angles, t.dihedrals) for r in group
                 for a in range(len(r)) for b in range(a + 1, len(r))}
    assert not excluded & set(zip(i.tolist(), j.tolist(), strict=True))


def _LJ_pair(D, **kwargs):
    """Force on the second atom (along x) and LJ energy of an O-O pair."""
    t = make_topology(2, lj_epsilon=np.full(2, EPSILON_O),
                      lj_sigma=np.full(2, SIGMA_O))
    result = forcefield(t, **kwargs).compute_forces(
        np.array([[0., 0., 0.], [D, 0., 0.]]), compute_energy=True)
    return result.forces[1, 0], result.potential_energies["LJ"]


def _plain_LJ(D):
    """Unshifted radial force and energy of the same O-O pair."""
    SIGovrD6 = (SIGMA_O / D)**6
    return (24 * EPSILON_O * (2 * SIGovrD6**2 - SIGovrD6) / D,
            4 * EPSILON_O * (SIGovrD6**2 - SIGovrD6))


RC = 2.5 * SIGMA_O


@pytest.mark.parametrize("D", [0.4, RC - 1e-9, RC + 1e-9])
def test_shifted_force(D):
    """The force and energy at the cutoff are subtracted, so both go to 0
    there, and beyond it there is no interaction."""
    F, E = _plain_LJ(D)
    F_rc, E_rc = _plain_LJ(RC)
    expected = ((F - F_rc, E - E_rc + (D - RC) * F_rc) if D < RC
                else (0., 0.))
    F_sf, E_sf = _LJ_pair(D, lj_cutoff=2.5)
    np.testing.assert_allclose(F_sf, expected[0], rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(E_sf, expected[1], rtol=1e-9, atol=1e-9)


def test_LJ_correction_gives_the_full_potential():
    """The shifted-force LJ energy plus lj_correction is the full LJ energy
    of a pair inside the cutoff, plus the tail beyond it for a uniform
    fluid: 2 pi / V N^2 int_rc^inf r^2 E(r) dr (2 atoms of one type)."""
    D, box_length = 0.4, 3.
    t = make_topology(2, lj_epsilon=np.full(2, EPSILON_O),
                      lj_sigma=np.full(2, SIGMA_O))
    result = forcefield(t, box_length, lj_cutoff=2.5).compute_forces(
        np.array([[0., 0., 0.], [D, 0., 0.]]), compute_energy=True)
    tail = (2 * np.pi / box_length**3 * 2**2
            * quad(lambda r: r**2 * _plain_LJ(r)[1], RC, np.inf)[0])
    np.testing.assert_allclose(
        result.potential_energies["LJ"] + result.lj_correction,
        _plain_LJ(D)[1] + tail, rtol=1e-9)


# ------------------------ Forces are -grad(energy) ---------------------------
def _only(topology, term):
    """The topology with every force field term except `term` switched off."""
    zero = {"bond": dict(k_bond=np.zeros_like(topology.k_bond)),
            "angle": dict(k_angle=np.zeros_like(topology.k_angle)),
            "dihedral": dict(rb_coefficients=np.zeros_like(
                topology.rb_coefficients)),
            "LJ": dict(lj_epsilon=np.zeros_like(topology.lj_epsilon))}
    changes = {}
    for other, values in zero.items():
        if other != term:
            changes.update(values)
    return replace(topology, **changes)


@pytest.mark.parametrize("term", ["bond", "angle", "dihedral", "LJ",
                                  "LJ with cutoff"])
def test_force_is_negative_energy_gradient(term):
    """Each force term equals minus the numerical gradient of its energy."""
    rng = np.random.default_rng(0)
    x, t = water_ethanol_pair()
    if term == "LJ with cutoff":
        # Shifted force at a finite cutoff
        term = "LJ"
        ff = forcefield(_only(t, term), lj_cutoff=2.5)
    else:
        ff = forcefield(_only(t, term))
    x = x + rng.normal(0, 0.005, x.shape)
    if term == "LJ":
        ff.compute_forces(x)
        assert len(ff.pair_i) > 0

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


# ----------------------------- Neighbour list --------------------------------
def test_neighbour_list_matches_all_pairs_over_a_run():
    """Forces with the neighbour list equal the all-pairs forces (up to
    rounding), also after the list has been rebuilt several times."""
    rng = np.random.default_rng(0)
    L = 2.0
    t, x = build_initial_box(L, {"Water": 1, "Ethanol": 1},
                             BUILTIN_MOLECULE_DIR, rng)
    listed = ForceField(t, L, lj_cutoff=2.0, skin=0.05)
    x, _ = minimise_energy(x, listed, 600.)
    # A skin reaching half the box makes the list hold all pairs
    everything = ForceField(t, L, lj_cutoff=2.0, skin=L)
    assert not listed.neighbour_list.holds_all_pairs
    assert everything.neighbour_list.holds_all_pairs

    v = initial_velocities(t.masses, 600., rng)
    state, _ = initial_state(x, v, listed.compute_forces, t.masses)
    for _ in range(60):
        state, result = integration_step(
            velocity_verlet, state, lambda x: listed.compute_forces(x, True),
            t.masses, 0.0005, target_temperature=600.)
        reference = everything.compute_forces(state.x, True)
        # Only the order of summation differs
        np.testing.assert_allclose(result.forces, reference.forces,
                                   rtol=1e-12, atol=1e-9)
        for name, value in reference.potential_energies.items():
            np.testing.assert_allclose(result.potential_energies[name], value,
                                       rtol=1e-12)
    assert listed.neighbour_list.nr_builds > 1
    assert len(listed.pair_i) < len(everything.pair_i) / 2

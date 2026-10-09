import numpy as np
import pytest

from auxiliary import diff_and_dist_arr, find_bond_indices
from config import kboltz
from energy import (angular_energy, bond_energy, dihedral_energy,
                    kinetic_energy, LJ_energy)
from forces import net_force
from initial_config_generation import initializeVelocity
from integrators import beeman, euler, integrator, velo_verlet, verlet

from helpers import random_molecule, water_ethanol_pair

eps = 1e-12
nr_tests = 15


def _step(method, s, x, x_prev, v, a, a_prev, dt, temp=0., temp_control=False):
    """One integrator step on the system dict s, without periodic boundaries."""
    n = len(s["masses"])
    rc = np.full(len(s["i_LJ"]), np.inf)
    force_parameters = (s["angle_mapping"], s["dihedral_mapping"], s["kb"],
                        s["r0"], s["angles"], s["ktheta"], s["theta0"],
                        s["epsilon_LJ"], s["sigma_LJ"], s["dihedrals"],
                        s["C1"], s["C2"], s["C3"], s["C4"], s["i_LJ"],
                        s["j_LJ"], rc, None)
    return integrator(method, x, x_prev, v, a, a_prev, force_parameters,
                      s["bonds"], n, dt, s["masses"], 0., temp, temp_control,
                      s["i_LJ"], s["j_LJ"], False)


def _random_system(rng, n=10):
    s = random_molecule(n, rng)
    s["angle_mapping"] = find_bond_indices(s["angles"], s["bonds"])
    s["dihedral_mapping"] = find_bond_indices(s["dihedrals"], s["bonds"])
    s["i_LJ"] = s["j_LJ"] = np.array([], dtype=int)
    s["epsilon_LJ"] = s["sigma_LJ"] = np.array([])
    return s


@pytest.mark.parametrize("method", [euler, verlet, beeman, velo_verlet])
def test_stationary_without_velocity_or_acceleration(method):
    rng = np.random.default_rng(1)
    for _ in range(nr_tests):
        s = _random_system(rng)
        zeros = np.zeros_like(s["x"])
        x_new = _step(method, s, s["x"], s["x"], zeros, zeros, zeros, 1.)[0]
        assert (np.abs(s["x"] - x_new) < eps).all()


@pytest.mark.parametrize("method", [euler, beeman, velo_verlet])
def test_positions_move_along_velocity(method):
    """Not applicable to Verlet, which has no explicit velocity."""
    rng = np.random.default_rng(2)
    for _ in range(nr_tests):
        s = _random_system(rng)
        zeros = np.zeros_like(s["x"])
        v = 2 * rng.random(s["x"].shape) - 1
        x_new = _step(method, s, s["x"], s["x"], v, zeros, zeros, 1.)[0]
        assert (np.abs(np.sign(x_new - s["x"]) - np.sign(v)) < eps).all()


# ----------------------------- Physical behaviour ----------------------------
def _total_energy(s, x, v):
    b = s["bonds"]
    diff_b, D_b = diff_and_dist_arr(x, b[:, 0], b[:, 1], 0.)
    _, D_LJ = diff_and_dist_arr(x, s["i_LJ"], s["j_LJ"], 0.)
    C = (s["C1"], s["C2"], s["C3"], s["C4"])
    return (bond_energy(D_b, s["kb"], s["r0"])
            + angular_energy(diff_b, D_b, s["angle_mapping"], s["ktheta"],
                             s["theta0"])
            + dihedral_energy(diff_b, D_b, s["dihedral_mapping"], *C)
            + LJ_energy(D_LJ, s["epsilon_LJ"], s["sigma_LJ"])
            + kinetic_energy(s["masses"], v))


def _run(method, s, v0, dt, steps, temp=0., temp_control=False):
    """Bootstrap with one Euler step (as the simulation does), then iterate.
    Returns the list of (x, v) per step."""
    n = len(s["masses"])
    b = s["bonds"]
    diff_b, D_b = diff_and_dist_arr(s["x"], b[:, 0], b[:, 1], 0.)
    diff_LJ, D_LJ = diff_and_dist_arr(s["x"], s["i_LJ"], s["j_LJ"], 0.)
    F = net_force(diff_b, D_b, diff_LJ, D_LJ, b, n, s["angle_mapping"],
                  s["dihedral_mapping"], s["kb"], s["r0"], s["angles"],
                  s["ktheta"], s["theta0"], s["epsilon_LJ"], s["sigma_LJ"],
                  s["dihedrals"], s["C1"], s["C2"], s["C3"], s["C4"],
                  s["i_LJ"], s["j_LJ"])
    a_prev = F / s["masses"][:, None]
    x_prev, v = s["x"], v0
    x, v, a = _step(euler, s, x_prev, x_prev, v, a_prev, a_prev, dt,
                    temp, temp_control)[:3]
    history = [(x_prev, v0), (x, v)]
    for _ in range(steps - 1):
        x_new, v, a_new = _step(method, s, x, x_prev, v, a, a_prev, dt,
                                temp, temp_control)[:3]
        x_prev, x, a_prev, a = x, x_new, a, a_new
        history.append((x, v))
    return history


def test_velocity_verlet_conserves_energy():
    np.random.seed(0)
    s = water_ethanol_pair()
    v0 = initializeVelocity(s["masses"], 300.)
    history = _run(velo_verlet, s, v0, dt=0.0002, steps=2000)

    E = np.array([_total_energy(s, x, v) for x, v in history])
    KE0 = kinetic_energy(s["masses"], v0)
    # Fluctuations of the total energy are small compared to kinetic energy
    assert np.ptp(E) < 0.01 * KE0
    # ...and there is no systematic drift
    assert abs(E[-200:].mean() - E[:200].mean()) < 0.001 * KE0


@pytest.mark.parametrize("method", [
    velo_verlet, beeman,
    pytest.param(verlet, marks=pytest.mark.xfail(
        strict=True, reason="Verlet ignores rescaled velocities")),
])
def test_temperature_control_holds_temperature(method):
    np.random.seed(0)
    s = water_ethanol_pair()
    target = 300.
    v0 = initializeVelocity(s["masses"], 3 * target)  # start far too hot
    history = _run(method, s, v0, dt=0.0005, steps=400, temp=target,
                   temp_control=True)

    # Measure the temperature from positions, so that it doesn't depend on
    # how each method estimates velocities.
    dt = 0.0005
    x = np.array([h[0] for h in history[-100:]])
    v_fd = (x[2:] - x[:-2]) / (2 * dt)
    KE = np.mean([kinetic_energy(s["masses"], v) for v in v_fd])
    T = 2 * KE / (3 * len(s["masses"]) * kboltz)
    assert abs(T - target) < 0.1 * target

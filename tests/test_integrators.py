import numpy as np
import pytest

from mdsim.constants import KBOLTZ
from mdsim.integrators import (METHODS, State, beeman, euler, initial_state,
                               step, velocity_verlet, verlet)
from mdsim.system import initial_velocities, kinetic_energy

from helpers import (forcefield, forces_and_energy, random_molecule,
                     water_ethanol_pair)

eps = 1e-12
nr_tests = 15


def test_methods_registry():
    assert set(METHODS) == {"euler", "verlet", "velocity_verlet", "beeman"}


@pytest.mark.parametrize("method", [euler, verlet, beeman, velocity_verlet])
def test_stationary_without_velocity_or_acceleration(method):
    rng = np.random.default_rng(1)
    for _ in range(nr_tests):
        x, t = random_molecule(10, rng)
        zeros = np.zeros_like(x)
        state = State(x, zeros, zeros, x, zeros)
        new, _ = step(method, state, forcefield(t).compute, t.masses, 1.)
        assert (np.abs(x - new.x) < eps).all()


@pytest.mark.parametrize("method", [euler, beeman, velocity_verlet])
def test_positions_move_along_velocity(method):
    """Not applicable to Verlet, which has no explicit velocity."""
    rng = np.random.default_rng(2)
    for _ in range(nr_tests):
        x, t = random_molecule(10, rng)
        zeros = np.zeros_like(x)
        v = 2 * rng.random(x.shape) - 1
        state = State(x, v, zeros, x, zeros)
        new, _ = step(method, state, forcefield(t).compute, t.masses, 1.)
        assert (np.abs(np.sign(new.x - x) - np.sign(v)) < eps).all()


# ----------------------------- Physical behaviour ----------------------------
def _run(method, ff, x, v0, dt, steps, temp=None):
    """Bootstrap with one Euler step (as the simulation does), then iterate.
    Returns the states of all steps."""
    masses = ff.topology.masses
    state, _ = initial_state(x, v0, ff.compute, masses)
    states = [state]
    for i in range(steps):
        state, _ = step(euler if i == 0 else method, state, ff.compute,
                        masses, dt, temp)
        states.append(state)
    return states


def test_velocity_verlet_conserves_energy():
    x, t = water_ethanol_pair()
    ff = forcefield(t)
    v0 = initial_velocities(t.masses, 300., np.random.default_rng(0))
    states = _run(velocity_verlet, ff, x, v0, dt=0.0002, steps=2000)

    E = np.array([forces_and_energy(ff, s.x)[1] + kinetic_energy(t.masses, s.v)
                  for s in states])
    KE0 = kinetic_energy(t.masses, v0)
    # Fluctuations of the total energy are small compared to kinetic energy
    assert np.ptp(E) < 0.01 * KE0
    # ...and there is no systematic drift
    assert abs(E[-200:].mean() - E[:200].mean()) < 0.001 * KE0


@pytest.mark.parametrize("method", [velocity_verlet, beeman, verlet])
def test_temperature_control_holds_temperature(method):
    x, t = water_ethanol_pair()
    target, dt = 300., 0.0005
    # Start far too hot
    v0 = initial_velocities(t.masses, 3 * target, np.random.default_rng(0))
    states = _run(method, forcefield(t), x, v0, dt, 400, temp=target)

    # Measure the temperature from positions, so that it doesn't depend on
    # how each method estimates velocities.
    x = np.array([s.x for s in states[-100:]])
    v_fd = (x[2:] - x[:-2]) / (2 * dt)
    KE = np.mean([kinetic_energy(t.masses, v) for v in v_fd])
    T = 2 * KE / (3 * len(t.masses) * KBOLTZ)
    assert abs(T - target) < 0.1 * target


@pytest.mark.parametrize("temp", [None, 300.])
def test_state_kinetic_energy_matches_velocities(temp):
    x, t = water_ethanol_pair()
    v0 = initial_velocities(t.masses, 900., np.random.default_rng(0))
    for state in _run(velocity_verlet, forcefield(t), x, v0, 0.0005, 5, temp):
        np.testing.assert_allclose(state.KE, kinetic_energy(t.masses, state.v),
                                   rtol=1e-12)
    if temp is not None:
        T = 2 * state.KE / (3 * len(t.masses) * KBOLTZ)
        np.testing.assert_allclose(T, temp, rtol=1e-12)

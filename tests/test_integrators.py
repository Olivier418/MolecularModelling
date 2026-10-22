import numpy as np
import pytest

from helpers import forcefield, forces_and_energy, water_ethanol_pair
from jostle.constants import BOLTZMANN_CONSTANT
from jostle.integrators import (
    beeman,
    initial_state,
    integration_step,
    startup_integrator,
    velocity_verlet,
    verlet,
)
from jostle.system import initial_velocities
from jostle.thermo import degrees_of_freedom, kinetic_energy


# ----------------------------- Physical behaviour ----------------------------
def _run(integrator, ff, x, v0, dt, steps, target_temperature=None):
    """Bootstrap with a self-starting step (as the simulation does), then
    iterate. Returns the states of all steps."""
    masses = ff.topology.masses
    state, _ = initial_state(x, v0, ff.compute_forces, masses)
    states = [state]
    for i in range(steps):
        step_integrator = (startup_integrator(integrator) if i == 0
                           else integrator)
        state, _ = integration_step(step_integrator, state, ff.compute_forces,
                                    masses, dt, target_temperature)
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


@pytest.mark.parametrize("integrator", [velocity_verlet, beeman, verlet])
def test_temperature_control_holds_temperature(integrator):
    x, t = water_ethanol_pair()
    target, dt = 300., 0.0005
    # Start far too hot
    v0 = initial_velocities(t.masses, 3 * target, np.random.default_rng(0))
    states = _run(integrator, forcefield(t), x, v0, dt, 400,
                  target_temperature=target)

    # Measure the temperature from positions, so that it doesn't depend on
    # how each integrator estimates velocities.
    x = np.array([s.x for s in states[-100:]])
    v_fd = (x[2:] - x[:-2]) / (2 * dt)
    KE = np.mean([kinetic_energy(t.masses, v) for v in v_fd])
    T = 2 * KE / (degrees_of_freedom(len(t.masses)) * BOLTZMANN_CONSTANT)
    assert abs(T - target) < 0.1 * target


def test_initial_velocities_maxwell_boltzmann():
    # Light and heavy atoms each get kT/2 per degree of freedom
    masses = np.repeat([1.008, 15.999], 20000)
    v = initial_velocities(masses, 300., np.random.default_rng(0))
    for m in (1.008, 15.999):
        sel = masses == m
        T = m * np.mean(np.sum(v[sel]**2, axis=1)) / (3 * BOLTZMANN_CONSTANT)
        assert abs(T - 300.) < 0.03 * 300.
    # Zero centre-of-mass momentum and exactly the target temperature
    np.testing.assert_allclose(masses @ v, 0, atol=1e-9)
    T = 2 * kinetic_energy(masses, v) / (degrees_of_freedom(len(masses))
                                         * BOLTZMANN_CONSTANT)
    np.testing.assert_allclose(T, 300., rtol=1e-12)

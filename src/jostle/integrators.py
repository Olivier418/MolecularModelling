"""Time integration schemes and a single integration step."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .thermo import kinetic_energy, velocity_scale_factor


# --------------------------- Integrator Functions ----------------------------
@dataclass(frozen=True)
class Integrator:
    """Update rules for positions and velocities. Suffixes: x_curr and x_prev
    are the positions at the current and previous step, x_new those at the
    step being computed; a_new is the acceleration at the new positions."""
    name: str
    # (x_curr, x_prev, v_curr, a_curr, a_prev, dt) -> x_new
    update_positions: Callable
    # (x_new, x_curr, x_prev, v_curr, a_new, a_curr, a_prev, dt) -> v_new
    update_velocities: Callable


def _taylor_positions(x_curr, x_prev, v_curr, a_curr, a_prev, dt):
    """Second-order Taylor step, shared by Euler and velocity Verlet."""
    return x_curr + dt * v_curr + (dt**2) / 2 * a_curr


euler = Integrator(
    "euler",
    _taylor_positions,
    lambda x_new, x_curr, x_prev, v_curr, a_new, a_curr, a_prev, dt:
        v_curr + dt * a_curr,
)

verlet = Integrator(
    "verlet",
    lambda x_curr, x_prev, v_curr, a_curr, a_prev, dt:
        2 * x_curr - x_prev + (dt**2) * a_curr,
    lambda x_new, x_curr, x_prev, v_curr, a_new, a_curr, a_prev, dt:
        1 / (2 * dt) * (x_new - x_prev),
)

velocity_verlet = Integrator(
    "velocity_verlet",
    _taylor_positions,
    lambda x_new, x_curr, x_prev, v_curr, a_new, a_curr, a_prev, dt:
        v_curr + dt / 2 * (a_curr + a_new),
)

beeman = Integrator(
    "beeman",
    lambda x_curr, x_prev, v_curr, a_curr, a_prev, dt:
        x_curr + dt * v_curr + (4 * a_curr - a_prev) * (1 / 6) * dt**2,
    lambda x_new, x_curr, x_prev, v_curr, a_new, a_curr, a_prev, dt:
        v_curr + (2 * a_new + 5 * a_curr - a_prev) * dt / 6,
)

INTEGRATORS = {m.name: m for m in (euler, verlet, velocity_verlet, beeman)}


def startup_integrator(integrator):
    """The integrator for the first step, which has no history yet: velocity
    Verlet needs only x, v and a at t=0, and leaves the real previous step for
    Verlet and Beeman. Euler runs stay Euler."""
    return euler if integrator is euler else velocity_verlet


# ------------------------------ Integration step -----------------------------
@dataclass
class State:
    """Positions, velocities and accelerations at the current step (x, v, a)
    and the previous step (x_prev, a_prev), and the kinetic energy."""
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    x_prev: np.ndarray
    a_prev: np.ndarray
    kinetic_energy: float


def initial_state(x, v, compute_forces, masses):
    """State at t=0. The previous step is set equal to the current one, which
    only the self-starting integrators (see startup_integrator) can handle."""
    evaluation = compute_forces(x)
    a = evaluation.forces / masses[:, None]
    return State(x, v, a, x, a, kinetic_energy(masses, v)), evaluation


def integration_step(integrator, state, compute_forces, masses, dt,
                     target_temperature=None):
    """
    Advances state by one step using compute_forces(x) -> ForceEvaluation. If
    target_temperature is given, velocities are rescaled to achieve it.
    Returns the new state and the ForceEvaluation at the new positions.
    """
    s = state
    x = integrator.update_positions(s.x, s.x_prev, s.v, s.a, s.a_prev, dt)
    evaluation = compute_forces(x)
    a = evaluation.forces / masses[:, None]
    v = integrator.update_velocities(x, s.x, s.x_prev, s.v, a, s.a, s.a_prev,
                                     dt)
    x_prev = s.x
    energy = kinetic_energy(masses, v)

    # Optional: Temperature scaling
    if target_temperature is not None:
        scale = velocity_scale_factor(energy, len(masses), target_temperature)
        v = scale * v
        energy *= scale**2
        # Verlet has no explicit velocity: rescale the last displacement
        if integrator is verlet:
            x_prev = x - scale * (x - x_prev)

    return State(x, v, a, x_prev, s.a, energy), evaluation

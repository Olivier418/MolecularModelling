"""Time integration schemes and a single integration step."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .system import kinetic_energy, temperature_scale


# --------------------------- Integrator Functions ----------------------------
@dataclass(frozen=True)
class Method:
    """Update rules for positions and velocities. Suffix im1/im2 means the
    value at step i-1/i-2; a_i is the acceleration at the new positions."""
    name: str
    positions: Callable    # (x_im1, x_im2, v_im1, a_im1, a_im2, dt) -> x_i
    velocities: Callable   # (x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt)


euler = Method(
    "euler",
    lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt:
        x_im1 + dt * v_im1 + (dt**2) / 2 * a_im1,
    lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt:
        v_im1 + dt * a_im1,
)

verlet = Method(
    "verlet",
    lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt:
        2 * x_im1 - x_im2 + (dt**2) * a_im1,
    lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt:
        1 / (2 * dt) * (x_i - x_im2),
)

velocity_verlet = Method(
    "velocity_verlet",
    lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt:
        x_im1 + dt * v_im1 + (dt**2) / 2 * a_im1,
    lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt:
        v_im1 + dt / 2 * (a_im1 + a_i),
)

beeman = Method(
    "beeman",
    lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt:
        x_im1 + dt * v_im1 + (4 * a_im1 - a_im2) * (1 / 6) * dt**2,
    lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt:
        v_im1 + (2 * a_i + 5 * a_im1 - a_im2) * dt / 6,
)

METHODS = {m.name: m for m in (euler, verlet, velocity_verlet, beeman)}


# ------------------------------ Integration step -----------------------------
@dataclass
class State:
    """Positions, velocities and accelerations at the current step (x, v, a)
    and the previous step (x_prev, a_prev), and the kinetic energy KE."""
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    x_prev: np.ndarray
    a_prev: np.ndarray
    KE: float | None = None


def initial_state(x, v, compute_forces, masses):
    """State at t=0. The previous step is set equal to the current one, which
    only Euler (used for the first step) can handle."""
    result = compute_forces(x)
    a = result.forces / masses[:, None]
    return State(x, v, a, x, a, kinetic_energy(masses, v)), result


def step(method, state, compute_forces, masses, dt, temp=None):
    """
    Advances state by one step using compute_forces(x) -> ForceResult. If temp
    is given, velocities are rescaled to achieve that temperature.
    Returns the new state and the ForceResult at the new positions.
    """
    s = state
    x = method.positions(s.x, s.x_prev, s.v, s.a, s.a_prev, dt)
    result = compute_forces(x)
    a = result.forces / masses[:, None]
    v = method.velocities(x, s.x, s.x_prev, s.v, a, s.a, s.a_prev, dt)
    x_prev = s.x
    KE = kinetic_energy(masses, v)

    # Optional: Temperature scaling
    if temp is not None:
        scale = temperature_scale(KE, len(masses), temp)
        v = scale * v
        KE *= scale**2
        # Verlet has no explicit velocity: rescale the last displacement
        if method is verlet:
            x_prev = x - scale * (x - x_prev)

    return State(x, v, a, x_prev, s.a, KE), result

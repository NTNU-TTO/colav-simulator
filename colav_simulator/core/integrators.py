"""integrators.py.

Summary:
Contains ODE integrator functionality used in the simulator.

Author: Trym Tengesdal
"""

from collections.abc import Callable
from typing import Any

import numpy as np

import colav_simulator.common.math_functions as mf


def erk4_integration_step(f: Callable, b: Callable, x: np.ndarray, u: np.ndarray, w: Any | None, dt: float) -> np.ndarray:
    """Performs a (saturated) single step of a 4th order Runge-Kutta integration scheme.

    Args:
        f (Callable): Function to be integrated.
        b (Callable): Bounds of states and inputs considered in the model dynamics to
            be integrated.
        x (np.ndarray): State vector.
        u (np.ndarray): Input vector.
        w (Any | None): Disturbance data.
        dt (float): Time step.

    Returns:
        np.ndarray: State vector at time t + dt.
    """
    _, _, lbx, ubx = b()
    k1 = f(x, u, w)
    k2 = f(x + 0.5 * dt * k1, u, w)
    k3 = f(x + 0.5 * dt * k2, u, w)
    k4 = f(x + dt * k3, u, w)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = mf.sat(x_next, lbx, ubx)
    return x_next


def euler_integration_step(f: Callable, b: Callable, x: np.ndarray, u: np.ndarray, w: Any | None, dt: float) -> np.ndarray:
    """Performs a (saturated) single step of a Euler integration scheme.

    Args:
        f (Callable): Function to be integrated.
        b (Callable): Bounds of states and inputs considered in the model dynamics to
            be integrated.
        x (np.ndarray): State vector.
        u (np.ndarray): Input vector.
        w (Any | None): Disturbance data.
        dt (float): Time step.

    Returns:
        np.ndarray: State vector at time t + dt.
    """
    _, _, lbx, ubx = b()
    x_next = x + dt * f(x, u, w)
    x_next = mf.sat(x_next, lbx, ubx)
    return x_next

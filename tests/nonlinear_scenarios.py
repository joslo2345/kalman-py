"""Nonlinear test problems. Model functions use jax.numpy so they work on both backends and
under jax.jacfwd; import this module only after checking JAX is installed."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from tests.scenarios import LinearScenario, cv_model_2d

SENSOR_R = np.diag([0.5**2, math.radians(0.5) ** 2])  # range (m), bearing (rad)


def cv_f(x: ArrayLike, dt: ArrayLike) -> jax.Array:
    x = jnp.asarray(x)
    """Constant-velocity motion for state (px, py, vx, vy), written for any dt."""
    return jnp.array([x[0] + dt * x[2], x[1] + dt * x[3], x[2], x[3]])


def cv_f_jacobian(x: ArrayLike, dt: ArrayLike) -> np.ndarray:
    return cv_model_2d(float(np.asarray(dt)), 0.0)[0]


def range_bearing_h(x: ArrayLike) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.array([jnp.hypot(x[0], x[1]), jnp.arctan2(x[1], x[0])])


def range_bearing_h_jacobian(x: ArrayLike) -> np.ndarray:
    px, py = float(np.asarray(x)[0]), float(np.asarray(x)[1])
    r2 = px**2 + py**2
    r = math.sqrt(r2)
    return np.array([[px / r, py / r, 0.0, 0.0], [-py / r2, px / r2, 0.0, 0.0]])


def wrap_bearing_residual(z: ArrayLike, z_pred: ArrayLike) -> jax.Array:
    d = jnp.asarray(z) - jnp.asarray(z_pred)
    return d.at[1].set((d[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)


def make_range_bearing(seed: int = 0, steps: int = 300, dt: float = 1.0) -> LinearScenario:
    """A target passing behind the sensor, so its bearing wraps from +pi to -pi."""
    rng = np.random.default_rng(seed)
    F, Q = cv_model_2d(dt=dt, q=0.01)
    x0 = np.array([-200.0, 60.0, 0.5, -0.8])
    P0 = np.diag([25.0, 25.0, 0.04, 0.04])
    x = rng.multivariate_normal(x0, P0)
    truth = np.empty((steps, 4))
    for k in range(steps):
        x = F @ x + rng.multivariate_normal(np.zeros(4), Q)
        truth[k] = x
    noise = rng.multivariate_normal(np.zeros(2), SENSOR_R, size=steps)
    zs = np.column_stack((np.hypot(truth[:, 0], truth[:, 1]), np.arctan2(truth[:, 1], truth[:, 0])))
    zs = zs + noise
    zs[:, 1] = (zs[:, 1] + np.pi) % (2 * np.pi) - np.pi
    H = np.zeros((2, 4))  # unused: the measurement model is nonlinear
    return LinearScenario(F, H, Q, SENSOR_R, x0, P0, zs, truth)


# Vectorized versions: accept a stack of points (leading axis) as well as a single point.
def cv_f_vectorized(x: ArrayLike, dt: ArrayLike) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.stack(
        [x[..., 0] + dt * x[..., 2], x[..., 1] + dt * x[..., 3], x[..., 2], x[..., 3]], axis=-1
    )


def range_bearing_h_vectorized(x: ArrayLike) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.stack([jnp.hypot(x[..., 0], x[..., 1]), jnp.arctan2(x[..., 1], x[..., 0])], axis=-1)


def wrap_bearing_residual_vectorized(z: ArrayLike, z_pred: ArrayLike) -> jax.Array:
    d = jnp.asarray(z) - jnp.asarray(z_pred)
    return d.at[..., 1].set((d[..., 1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

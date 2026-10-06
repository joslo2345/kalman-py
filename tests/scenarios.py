"""Synthetic test problems shared across the test suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LinearScenario:
    F: Array
    H: Array
    Q: Array
    R: Array
    x0: Array
    P0: Array
    zs: Array  # (T, m) measurements
    truth: Array  # (T, n) true states

    @property
    def params(self) -> dict[str, Any]:
        return {"F": self.F, "H": self.H, "Q": self.Q, "R": self.R, "x0": self.x0, "P0": self.P0}

    @property
    def n(self) -> int:
        return int(self.F.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])


def cv_model_2d(dt: float, q: float) -> tuple[Array, Array]:
    """2-D constant-velocity transition matrix and process noise; state is (px, py, vx, vy)."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    # Discretized continuous white-noise acceleration, per axis.
    block = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    Q = np.zeros((4, 4))
    for axis in (0, 1):
        idx = np.ix_([axis, axis + 2], [axis, axis + 2])
        Q[idx] = block
    return F, Q


def simulate(
    F: Array, H: Array, Q: Array, R: Array, x0: Array, P0: Array, steps: int, seed: int
) -> LinearScenario:
    rng = np.random.default_rng(seed)
    n, m = F.shape[0], H.shape[0]
    x = rng.multivariate_normal(x0, P0)
    process_noise = rng.multivariate_normal(np.zeros(n), Q, size=steps)
    truth = np.empty((steps, n))
    for k in range(steps):
        x = F @ x + process_noise[k]
        truth[k] = x
    zs = truth @ H.T + rng.multivariate_normal(np.zeros(m), R, size=steps)
    return LinearScenario(F, H, Q, R, x0, P0, zs, truth)


def make_cv_2d(seed: int = 0, steps: int = 200) -> LinearScenario:
    """A typical 2-D tracking problem with position measurements."""
    F, Q = cv_model_2d(dt=0.1, q=0.5)
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    R = np.diag([0.5, 0.8])
    x0 = np.array([0.0, 0.0, 1.0, -1.0])
    P0 = np.diag([1.0, 1.0, 0.5, 0.5])
    return simulate(F, H, Q, R, x0, P0, steps, seed)


def make_random_linear(seed: int, r_scale: float, steps: int) -> LinearScenario:
    """A random 4-state / 2-measurement problem; tiny ``r_scale`` makes P ill-conditioned."""
    rng = np.random.default_rng(seed)
    F, Q = cv_model_2d(dt=rng.uniform(0.01, 1.0), q=10.0 ** rng.uniform(-4, 1))
    H = rng.normal(size=(2, 4))
    B = rng.normal(size=(2, 2))
    R = r_scale * (B @ B.T + 0.1 * np.eye(2))
    x0 = rng.normal(size=4)
    P0 = 10.0 ** rng.uniform(-2, 4) * np.eye(4)
    return simulate(F, H, Q, R, x0, P0, steps, seed)

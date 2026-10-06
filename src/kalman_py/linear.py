"""Linear Kalman filter."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from kalman_py._typing import Array
from kalman_py.backends import numpy_backend
from kalman_py.result import FilterResult, SmootherResult

Backend = Literal["numpy", "jax"]


def _as_float_arrays(*arrays: ArrayLike) -> list[Array]:
    """Convert to arrays of one common float dtype, keeping float32 if all inputs are float32."""
    converted = [np.asarray(a) for a in arrays]
    dtype = np.result_type(*converted)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    return [a.astype(dtype, copy=True) for a in converted]


def _check_shape(name: str, a: Array, shape: tuple[int, ...]) -> None:
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")


class KalmanFilter:
    """Linear Kalman filter for ``x_k = F x_{k-1} + w``, ``z_k = H x_k + v``.

    ``w ~ N(0, Q)`` and ``v ~ N(0, R)``. ``(x0, P0)`` is the prior *before* the first
    prediction, so every measurement is preceded by a predict step (the FilterPy
    convention).

    Use :meth:`predict` and :meth:`update` in real-time loops, or :meth:`filter` to process a
    whole measurement sequence at once, then :meth:`smooth` to refine it with future data.
    """

    def __init__(
        self,
        F: ArrayLike,
        H: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
    ) -> None:
        self.F, self.H, self.Q, self.R, self.x0, self.P0 = _as_float_arrays(F, H, Q, R, x0, P0)
        if self.H.ndim != 2:
            raise ValueError(f"H must be 2-D, got shape {self.H.shape}")
        m, n = self.H.shape
        _check_shape("F", self.F, (n, n))
        _check_shape("Q", self.Q, (n, n))
        _check_shape("R", self.R, (m, m))
        _check_shape("x0", self.x0, (n,))
        _check_shape("P0", self.P0, (n, n))
        self.x = self.x0.copy()
        self.P = self.P0.copy()

    @property
    def dim_x(self) -> int:
        return int(self.x0.shape[0])

    @property
    def dim_z(self) -> int:
        return int(self.H.shape[0])

    def predict(self) -> None:
        """Advance the current estimate ``(x, P)`` by one time step."""
        self.x, self.P = numpy_backend.predict(self.x, self.P, self.F, self.Q)

    def update(self, z: ArrayLike) -> None:
        """Condition the current estimate on one measurement."""
        z_arr = np.asarray(z, dtype=self.P.dtype).reshape(-1)
        _check_shape("z", z_arr, (self.dim_z,))
        self.x, self.P, _, _ = numpy_backend.update(self.x, self.P, z_arr, self.H, self.R)

    def filter(self, zs: ArrayLike, backend: Backend = "numpy") -> FilterResult:
        """Filter a measurement sequence of shape ``(T, dim_z)``, starting from ``(x0, P0)``.

        This does not change the step-by-step state ``(x, P)``. A 1-D ``zs`` is accepted when
        ``dim_z == 1``.
        """
        zs_arr = np.asarray(zs, dtype=self.P0.dtype)
        if zs_arr.ndim == 1 and self.dim_z == 1:
            zs_arr = zs_arr[:, None]
        if zs_arr.ndim != 2 or zs_arr.shape[1] != self.dim_z:
            raise ValueError(f"zs must have shape (T, {self.dim_z}), got {zs_arr.shape}")

        if backend == "numpy":
            return numpy_backend.kalman_filter(
                self.F, self.H, self.Q, self.R, self.x0, self.P0, zs_arr
            )
        if backend == "jax":
            raise NotImplementedError("the JAX backend is not implemented yet")
        raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'jax'")

    def smooth(self, result: FilterResult) -> SmootherResult:
        """Run the RTS smoother over the output of :meth:`filter`."""
        if result.means.ndim != 2 or result.means.shape[1] != self.dim_x:
            raise ValueError(
                f"result does not match this filter: expected means of shape (T, {self.dim_x}), "
                f"got {result.means.shape}"
            )
        return numpy_backend.rts_smoother(self.F, result)

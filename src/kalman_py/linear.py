"""Linear Kalman filter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from kalman_py._common import (
    Backend,
    as_float_arrays,
    check_filter_result,
    check_measurements,
    check_shape,
    jax_backend,
)
from kalman_py._typing import Array
from kalman_py.backends import numpy_backend
from kalman_py.result import FilterResult, SmootherResult

if TYPE_CHECKING:
    import jax


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
        self.F, self.H, self.Q, self.R, self.x0, self.P0 = as_float_arrays(F, H, Q, R, x0, P0)
        if self.H.ndim != 2:
            raise ValueError(f"H must be 2-D, got shape {self.H.shape}")
        m, n = self.H.shape
        check_shape("F", self.F, (n, n))
        check_shape("Q", self.Q, (n, n))
        check_shape("R", self.R, (m, m))
        check_shape("x0", self.x0, (n,))
        check_shape("P0", self.P0, (n, n))
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
        check_shape("z", z_arr, (self.dim_z,))
        self.x, self.P, _, _ = numpy_backend.update(self.x, self.P, z_arr, self.H, self.R)

    @overload
    def filter(self, zs: ArrayLike, backend: Literal["numpy"] = ...) -> FilterResult[Array]: ...
    @overload
    def filter(
        self, zs: ArrayLike | jax.Array, backend: Literal["jax"]
    ) -> FilterResult[jax.Array]: ...

    def filter(
        self, zs: ArrayLike | jax.Array, backend: Backend = "numpy"
    ) -> FilterResult[Array] | FilterResult[jax.Array]:
        """Filter a measurement sequence of shape ``(T, dim_z)``, starting from ``(x0, P0)``.

        This does not change the step-by-step state ``(x, P)``. A 1-D ``zs`` is accepted when
        ``dim_z == 1``. With ``backend="jax"`` the result holds JAX arrays, and JAX arrays
        passed as ``zs`` stay on their device.
        """
        if backend == "numpy":
            zs_np = check_measurements(np.asarray(zs, dtype=self.P0.dtype), self.dim_z)
            return numpy_backend.kalman_filter(
                self.F, self.H, self.Q, self.R, self.x0, self.P0, zs_np
            )
        if backend == "jax":
            jb = jax_backend()
            zs_jax = check_measurements(jb.asarray(zs, self.P0.dtype), self.dim_z)
            result: FilterResult[jax.Array] = jb.kalman_filter(
                self.F, self.H, self.Q, self.R, self.x0, self.P0, zs_jax
            )
            return result
        raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'jax'")

    @overload
    def smooth(self, result: FilterResult[Array]) -> SmootherResult[Array]: ...
    @overload
    def smooth(self, result: FilterResult[jax.Array]) -> SmootherResult[jax.Array]: ...

    def smooth(
        self, result: FilterResult[Array] | FilterResult[jax.Array]
    ) -> SmootherResult[Array] | SmootherResult[jax.Array]:
        """Run the RTS smoother over the output of :meth:`filter`, on the same backend."""
        check_filter_result(result.means.shape, self.dim_x)
        if isinstance(result.means, np.ndarray):
            return numpy_backend.rts_smoother(self.F, result)
        smoothed: SmootherResult[jax.Array] = jax_backend().rts_smoother(self.F, result)
        return smoothed
